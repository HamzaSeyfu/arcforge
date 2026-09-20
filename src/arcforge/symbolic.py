from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Iterable

Grid = list[list[int]]


def _shape(g: Grid) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def _copy(g: Grid) -> Grid:
    return [row[:] for row in g]


def _rot90(g: Grid) -> Grid:
    if not g:
        return []
    return [list(row) for row in zip(*g[::-1])]


def _rot180(g: Grid) -> Grid:
    return _rot90(_rot90(g))


def _rot270(g: Grid) -> Grid:
    return _rot90(_rot180(g))


def _flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def _flip_v(g: Grid) -> Grid:
    return [row[:] for row in g[::-1]]


def _transpose(g: Grid) -> Grid:
    if not g:
        return []
    return [list(row) for row in zip(*g)]


def _anti_transpose(g: Grid) -> Grid:
    return _transpose(_rot180(g))


def _mode(g: Grid) -> int:
    counts = Counter(v for row in g for v in row)
    return counts.most_common(1)[0][0]


def _bbox_crop(g: Grid, predicate: Callable[[int], bool]) -> Grid | None:
    cells = [
        (r, c)
        for r, row in enumerate(g)
        for c, value in enumerate(row)
        if predicate(value)
    ]
    if not cells:
        return None
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    return [row[c0 : c1 + 1] for row in g[r0 : r1 + 1]]


def _scale(g: Grid, factor: int) -> Grid:
    out: Grid = []
    for row in g:
        expanded = [v for v in row for _ in range(factor)]
        for _ in range(factor):
            out.append(expanded[:])
    return out


def _tile(g: Grid, vr: int, hc: int) -> Grid:
    out: Grid = []
    for _ in range(vr):
        for row in g:
            out.append(row * hc)
    return out


def _mask_tile(g: Grid, background: int) -> Grid:
    """Kronecker-style expansion: a foreground cell emits the whole source grid."""
    h, w = _shape(g)
    out = [[background for _ in range(w * w)] for _ in range(h * h)]
    for br in range(h):
        for bc in range(w):
            if g[br][bc] == background:
                continue
            for r in range(h):
                for c in range(w):
                    out[br * h + r][bc * w + c] = g[r][c]
    return out


def _connected_components(g: Grid, background: int) -> list[list[tuple[int, int]]]:
    h, w = _shape(g)
    seen = [[False] * w for _ in range(h)]
    components: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or g[sr][sc] == background:
                continue
            q = deque([(sr, sc)])
            seen[sr][sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr][nc]
                        and g[nr][nc] != background
                    ):
                        seen[nr][nc] = True
                        q.append((nr, nc))
            components.append(cells)
    return components


def _crop_component(
    g: Grid,
    cells: list[tuple[int, int]],
    background: int,
) -> Grid:
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    out = [[background for _ in range(c1 - c0 + 1)] for _ in range(r1 - r0 + 1)]
    for r, c in cells:
        out[r - r0][c - c0] = g[r][c]
    return out


def _infer_color_map(source: Grid, target: Grid) -> dict[int, int] | None:
    if _shape(source) != _shape(target):
        return None
    mapping: dict[int, int] = {}
    for sr, tr in zip(source, target):
        for s, t in zip(sr, tr):
            if s in mapping and mapping[s] != t:
                return None
            mapping[s] = t
    return mapping


def _apply_color_map(g: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(v, v) for v in row] for row in g]


@dataclass(frozen=True)
class SymbolicProgram:
    name: str
    apply: Callable[[Grid], Grid | None]


@dataclass(frozen=True)
class SymbolicResult:
    programs: tuple[str, ...]
    candidates: tuple[Grid, ...]


def _merge_color_maps(
    transformed: Callable[[Grid], Grid | None],
    train: list[dict],
) -> dict[int, int] | None:
    merged: dict[int, int] = {}
    for pair in train:
        source = transformed(pair["input"])
        if source is None:
            return None
        mapping = _infer_color_map(source, pair["output"])
        if mapping is None:
            return None
        for k, v in mapping.items():
            if k in merged and merged[k] != v:
                return None
            merged[k] = v
    return merged


def _program_library(train: list[dict]) -> list[SymbolicProgram]:
    programs: list[SymbolicProgram] = []

    geometry: list[tuple[str, Callable[[Grid], Grid]]] = [
        ("identity", _copy),
        ("rot90", _rot90),
        ("rot180", _rot180),
        ("rot270", _rot270),
        ("flip_h", _flip_h),
        ("flip_v", _flip_v),
        ("transpose", _transpose),
        ("anti_transpose", _anti_transpose),
    ]
    programs.extend(SymbolicProgram(name, fn) for name, fn in geometry)

    programs.append(
        SymbolicProgram(
            "crop_non_background",
            lambda g: _bbox_crop(g, lambda v: v != _mode(g)),
        )
    )
    for color in range(10):
        programs.append(
            SymbolicProgram(
                f"crop_color_{color}",
                lambda g, color=color: _bbox_crop(g, lambda v: v == color),
            )
        )

    for factor in range(2, 5):
        programs.append(
            SymbolicProgram(
                f"scale_{factor}",
                lambda g, factor=factor: _scale(g, factor),
            )
        )

    for vr in range(1, 5):
        for hc in range(1, 5):
            if vr == 1 and hc == 1:
                continue
            programs.append(
                SymbolicProgram(
                    f"tile_{vr}x{hc}",
                    lambda g, vr=vr, hc=hc: _tile(g, vr, hc),
                )
            )

    programs.extend(
        [
            SymbolicProgram("mask_tile_mode", lambda g: _mask_tile(g, _mode(g))),
            SymbolicProgram("mask_tile_zero", lambda g: _mask_tile(g, 0)),
        ]
    )

    for rank in range(4):
        programs.append(
            SymbolicProgram(
                f"component_largest_{rank}",
                lambda g, rank=rank: _component_by_rank(g, rank, largest=True),
            )
        )
        programs.append(
            SymbolicProgram(
                f"component_smallest_{rank}",
                lambda g, rank=rank: _component_by_rank(g, rank, largest=False),
            )
        )

    # Learned color maps, optionally after a dihedral transform.
    for name, transform in geometry:
        mapping = _merge_color_maps(transform, train)
        if mapping is not None:
            programs.append(
                SymbolicProgram(
                    f"{name}_color_map",
                    lambda g, transform=transform, mapping=mapping: _apply_color_map(
                        transform(g), mapping
                    ),
                )
            )

    return programs


def _component_by_rank(g: Grid, rank: int, largest: bool) -> Grid | None:
    bg = _mode(g)
    components = _connected_components(g, bg)
    components.sort(key=len, reverse=largest)
    if rank >= len(components):
        return None
    return _crop_component(g, components[rank], bg)


def _fits_training(program: SymbolicProgram, train: list[dict]) -> bool:
    for pair in train:
        try:
            pred = program.apply(pair["input"])
        except Exception:
            return False
        if pred is None or pred != pair["output"]:
            return False
    return True


def solve_symbolic_task(task: dict) -> list[SymbolicResult]:
    """Return all distinct execution-verified candidates for each test output."""
    train = task["train"]
    fitting = [p for p in _program_library(train) if _fits_training(p, train)]

    results: list[list[SymbolicResult]] = []
    for test_case in task["test"]:
        seen: dict[tuple[tuple[int, ...], ...], tuple[Grid, list[str]]] = {}
        for program in fitting:
            try:
                candidate = program.apply(test_case["input"])
            except Exception:
                continue
            if candidate is None:
                continue
            key = tuple(tuple(row) for row in candidate)
            if key not in seen:
                seen[key] = (candidate, [program.name])
            else:
                seen[key][1].append(program.name)

        row_results = [
            SymbolicResult(programs=tuple(names), candidates=(grid,))
            for grid, names in seen.values()
        ]
        results.append(row_results)

    return results


def candidate_grids(task: dict) -> list[list[Grid]]:
    raw = solve_symbolic_task(task)
    return [[result.candidates[0] for result in row] for row in raw]
