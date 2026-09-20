from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .symbolic import Grid, _infer_color_map, _apply_color_map, _mode


@dataclass(frozen=True)
class PanelScheme:
    axis: str  # "v" or "h"
    count: int
    separator: int = 0

    def extract(self, grid: Grid) -> list[Grid] | None:
        h = len(grid)
        w = len(grid[0]) if grid else 0
        if self.axis == "v":
            usable = h - (self.count - 1) * self.separator
            if usable <= 0 or usable % self.count:
                return None
            ph = usable // self.count
            panels = []
            start = 0
            for _ in range(self.count):
                panels.append([row[:] for row in grid[start : start + ph]])
                start += ph + self.separator
            return panels

        usable = w - (self.count - 1) * self.separator
        if usable <= 0 or usable % self.count:
            return None
        pw = usable // self.count
        panels = []
        start = 0
        for _ in range(self.count):
            panels.append([row[start : start + pw] for row in grid])
            start += pw + self.separator
        return panels


@dataclass(frozen=True)
class PanelProgram:
    name: str
    scheme: PanelScheme
    combine: Callable[[list[Grid]], Grid | None]
    color_map: dict[int, int] | None = None

    def apply(self, grid: Grid) -> Grid | None:
        panels = self.scheme.extract(grid)
        if panels is None:
            return None
        result = self.combine(panels)
        if result is not None and self.color_map is not None:
            result = _apply_color_map(result, self.color_map)
        return result


def _same_shapes(panels: list[Grid]) -> bool:
    if not panels:
        return False
    shape = (len(panels[0]), len(panels[0][0]))
    return all((len(p), len(p[0])) == shape for p in panels)


def _select(index: int):
    def fn(panels: list[Grid]) -> Grid | None:
        if index >= len(panels):
            return None
        return [row[:] for row in panels[index]]
    return fn


def _overlay(order: tuple[int, ...]):
    def fn(panels: list[Grid]) -> Grid | None:
        if not _same_shapes(panels) or any(i >= len(panels) for i in order):
            return None
        h, w = len(panels[0]), len(panels[0][0])
        bgs = [_mode(p) for p in panels]
        out = [[bgs[order[-1]] for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                value = out[r][c]
                for idx in order:
                    if panels[idx][r][c] != bgs[idx]:
                        value = panels[idx][r][c]
                        break
                out[r][c] = value
        return out
    return fn


def _consensus(panels: list[Grid]) -> Grid | None:
    if not _same_shapes(panels):
        return None
    h, w = len(panels[0]), len(panels[0][0])
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            vals = [p[r][c] for p in panels]
            # deterministic plurality, tie by earliest panel
            counts = {v: vals.count(v) for v in set(vals)}
            best_count = max(counts.values())
            out[r][c] = next(v for v in vals if counts[v] == best_count)
    return out


def _mask_relation(kind: str):
    def fn(panels: list[Grid]) -> Grid | None:
        if not _same_shapes(panels):
            return None
        h, w = len(panels[0]), len(panels[0][0])
        bgs = [_mode(p) for p in panels]
        out = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                fg = [panels[i][r][c] != bgs[i] for i in range(len(panels))]
                vals = [panels[i][r][c] for i in range(len(panels))]
                if kind == "any_fg":
                    bit = any(fg)
                elif kind == "all_fg":
                    bit = all(fg)
                elif kind == "exactly_one_fg":
                    bit = sum(fg) == 1
                elif kind == "odd_fg":
                    bit = sum(fg) % 2 == 1
                elif kind == "all_equal":
                    bit = len(set(vals)) == 1
                elif kind == "not_all_equal":
                    bit = len(set(vals)) > 1
                else:
                    raise ValueError(kind)
                out[r][c] = 1 if bit else 0
        return out
    return fn


def _combine_library(count: int) -> list[tuple[str, Callable[[list[Grid]], Grid | None]]]:
    ops: list[tuple[str, Callable[[list[Grid]], Grid | None]]] = []
    for i in range(count):
        ops.append((f"select_{i}", _select(i)))

    if count >= 2:
        ops.append(("overlay_forward", _overlay(tuple(range(count)))))
        ops.append(("overlay_reverse", _overlay(tuple(reversed(range(count))))))
        ops.append(("consensus", _consensus))

    for kind in (
        "any_fg",
        "all_fg",
        "exactly_one_fg",
        "odd_fg",
        "all_equal",
        "not_all_equal",
    ):
        ops.append((f"mask_{kind}", _mask_relation(kind)))
    return ops


def _scheme_candidates(train: list[dict]) -> list[PanelScheme]:
    schemes: list[PanelScheme] = []
    for axis in ("v", "h"):
        for count in range(2, 5):
            for separator in (0, 1, 2):
                scheme = PanelScheme(axis, count, separator)
                ok = True
                for pair in train:
                    panels = scheme.extract(pair["input"])
                    if panels is None:
                        ok = False
                        break
                    out_h = len(pair["output"])
                    out_w = len(pair["output"][0])
                    if any((len(p), len(p[0])) != (out_h, out_w) for p in panels):
                        ok = False
                        break
                if ok:
                    schemes.append(scheme)
    return schemes


def _learn_map_for_program(
    scheme: PanelScheme,
    combine: Callable[[list[Grid]], Grid | None],
    train: list[dict],
) -> dict[int, int] | None:
    merged: dict[int, int] = {}
    for pair in train:
        panels = scheme.extract(pair["input"])
        if panels is None:
            return None
        raw = combine(panels)
        if raw is None:
            return None
        mapping = _infer_color_map(raw, pair["output"])
        if mapping is None:
            return None
        for k, v in mapping.items():
            if k in merged and merged[k] != v:
                return None
            merged[k] = v
    return merged


def synthesize_panel_programs(task: dict) -> list[PanelProgram]:
    train = task["train"]
    programs: list[PanelProgram] = []

    for scheme in _scheme_candidates(train):
        for name, combine in _combine_library(scheme.count):
            raw_ok = True
            for pair in train:
                panels = scheme.extract(pair["input"])
                pred = combine(panels) if panels is not None else None
                if pred != pair["output"]:
                    raw_ok = False
                    break
            if raw_ok:
                programs.append(
                    PanelProgram(
                        f"{scheme.axis}{scheme.count}s{scheme.separator}:{name}",
                        scheme,
                        combine,
                    )
                )
                continue

            mapping = _learn_map_for_program(scheme, combine, train)
            if mapping is not None:
                programs.append(
                    PanelProgram(
                        f"{scheme.axis}{scheme.count}s{scheme.separator}:{name}:map",
                        scheme,
                        combine,
                        mapping,
                    )
                )

    # behavior-level dedupe on training pairs
    unique: list[PanelProgram] = []
    seen = set()
    for program in programs:
        behavior = tuple(
            tuple(tuple(r) for r in (program.apply(pair["input"]) or []))
            for pair in train
        )
        if behavior not in seen:
            seen.add(behavior)
            unique.append(program)
    return unique


def panel_candidate_grids(task: dict) -> list[list[Grid]]:
    programs = synthesize_panel_programs(task)
    pools: list[list[Grid]] = []
    for test in task["test"]:
        row: list[Grid] = []
        seen = set()
        for program in programs:
            try:
                candidate = program.apply(test["input"])
            except Exception:
                continue
            if candidate is None:
                continue
            key = tuple(tuple(r) for r in candidate)
            if key not in seen:
                seen.add(key)
                row.append(candidate)
        pools.append(row)
    return pools
