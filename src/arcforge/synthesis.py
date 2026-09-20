from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

from .symbolic import (
    Grid,
    _anti_transpose,
    _copy,
    _bbox_crop,
    _component_by_rank,
    _flip_h,
    _flip_v,
    _infer_color_map,
    _mode,
    _rot180,
    _rot270,
    _rot90,
    _scale,
    _tile,
    _transpose,
    _apply_color_map,
)


@dataclass(frozen=True)
class Primitive:
    name: str
    apply: Callable[[Grid], Grid | None]


@dataclass(frozen=True)
class SynthesizedProgram:
    name: str
    primitives: tuple[Primitive, ...]
    color_map: dict[int, int] | None = None

    def apply(self, grid: Grid) -> Grid | None:
        value: Grid | None = grid
        for primitive in self.primitives:
            if value is None:
                return None
            value = primitive.apply(value)
        if value is not None and self.color_map is not None:
            value = _apply_color_map(value, self.color_map)
        return value


def _hcat(a: Grid, b: Grid) -> Grid | None:
    if len(a) != len(b):
        return None
    return [ra + rb for ra, rb in zip(a, b)]


def _vcat(a: Grid, b: Grid) -> Grid | None:
    if not a or not b or len(a[0]) != len(b[0]):
        return None
    return [row[:] for row in a] + [row[:] for row in b]


def _mirror_h(g: Grid) -> Grid:
    return [row + row[::-1] for row in g]


def _mirror_h_overlap(g: Grid) -> Grid:
    return [row + row[-2::-1] if len(row) > 1 else row[:] for row in g]


def _mirror_v(g: Grid) -> Grid:
    return [row[:] for row in g] + [row[:] for row in g[::-1]]


def _mirror_v_overlap(g: Grid) -> Grid:
    if len(g) <= 1:
        return [row[:] for row in g]
    return [row[:] for row in g] + [row[:] for row in g[-2::-1]]


def _primitive_library() -> list[Primitive]:
    primitives = [
        Primitive("identity", lambda g: [row[:] for row in g]),
        Primitive("rot90", _rot90),
        Primitive("rot180", _rot180),
        Primitive("rot270", _rot270),
        Primitive("flip_h", _flip_h),
        Primitive("flip_v", _flip_v),
        Primitive("transpose", _transpose),
        Primitive("anti_transpose", _anti_transpose),
        Primitive(
            "crop_non_background",
            lambda g: _bbox_crop(g, lambda v: v != _mode(g)),
        ),
        Primitive("mirror_h", _mirror_h),
        Primitive("mirror_h_overlap", _mirror_h_overlap),
        Primitive("mirror_v", _mirror_v),
        Primitive("mirror_v_overlap", _mirror_v_overlap),
        Primitive("scale_2", lambda g: _scale(g, 2)),
        Primitive("scale_3", lambda g: _scale(g, 3)),
        Primitive("tile_1x2", lambda g: _tile(g, 1, 2)),
        Primitive("tile_2x1", lambda g: _tile(g, 2, 1)),
        Primitive("tile_2x2", lambda g: _tile(g, 2, 2)),
        Primitive(
            "largest_component",
            lambda g: _component_by_rank(g, 0, largest=True),
        ),
        Primitive(
            "smallest_component",
            lambda g: _component_by_rank(g, 0, largest=False),
        ),
    ]

    for color in range(10):
        primitives.append(
            Primitive(
                f"crop_color_{color}",
                lambda g, color=color: _bbox_crop(g, lambda v: v == color),
            )
        )

    # Common composition-like transforms exposed as atomic primitives so depth-2
    # search can combine them with cropping, geometry, scaling, etc.
    primitives.extend(
        [
            Primitive("hcat_flip_h", lambda g: _hcat(g, _flip_h(g))),
            Primitive("hcat_rot180", lambda g: _hcat(g, _rot180(g))),
            Primitive("vcat_flip_v", lambda g: _vcat(g, _flip_v(g))),
            Primitive("vcat_rot180", lambda g: _vcat(g, _rot180(g))),
        ]
    )
    return primitives


def _run_sequence(grid: Grid, sequence: tuple[Primitive, ...]) -> Grid | None:
    value: Grid | None = grid
    for primitive in sequence:
        if value is None:
            return None
        try:
            value = primitive.apply(value)
        except Exception:
            return None
    return value


def _fits_raw(sequence: tuple[Primitive, ...], train: list[dict]) -> bool:
    for pair in train:
        if _run_sequence(pair["input"], sequence) != pair["output"]:
            return False
    return True


def _learn_final_color_map(
    sequence: tuple[Primitive, ...],
    train: list[dict],
) -> dict[int, int] | None:
    merged: dict[int, int] = {}
    for pair in train:
        intermediate = _run_sequence(pair["input"], sequence)
        if intermediate is None:
            return None
        mapping = _infer_color_map(intermediate, pair["output"])
        if mapping is None:
            return None
        for k, v in mapping.items():
            if k in merged and merged[k] != v:
                return None
            merged[k] = v
    return merged


def synthesize_programs(
    train: list[dict],
    max_depth: int = 2,
    max_programs: int = 64,
) -> list[SynthesizedProgram]:
    primitives = _primitive_library()
    found: list[SynthesizedProgram] = []
    seen_behaviors: set[tuple] = set()

    for depth in range(1, max_depth + 1):
        for sequence in product(primitives, repeat=depth):
            # Avoid obvious no-op explosions.
            if depth > 1 and all(p.name == "identity" for p in sequence):
                continue

            if _fits_raw(sequence, train):
                program = SynthesizedProgram(
                    name=" -> ".join(p.name for p in sequence),
                    primitives=sequence,
                )
            else:
                mapping = _learn_final_color_map(sequence, train)
                if mapping is None:
                    continue
                program = SynthesizedProgram(
                    name=" -> ".join(p.name for p in sequence) + " -> color_map",
                    primitives=sequence,
                    color_map=mapping,
                )

            # Deduplicate programs by their behavior on all training inputs.
            behavior = tuple(
                tuple(tuple(row) for row in (program.apply(pair["input"]) or []))
                for pair in train
            )
            if behavior in seen_behaviors:
                continue
            seen_behaviors.add(behavior)
            found.append(program)
            if len(found) >= max_programs:
                return found

    return found


def synthesis_candidate_grids(
    task: dict,
    max_depth: int = 2,
    max_programs: int = 64,
) -> list[list[Grid]]:
    programs = synthesize_programs(
        task["train"],
        max_depth=max_depth,
        max_programs=max_programs,
    )
    pools: list[list[Grid]] = []
    for test_case in task["test"]:
        row: list[Grid] = []
        seen: set[tuple[tuple[int, ...], ...]] = set()
        for program in programs:
            candidate = program.apply(test_case["input"])
            if candidate is None:
                continue
            key = tuple(tuple(r) for r in candidate)
            if key in seen:
                continue
            seen.add(key)
            row.append(candidate)
        pools.append(row)
    return pools
