from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

from .symbolic import Grid, _mode


@dataclass(frozen=True)
class RelationalRule:
    name: str
    apply: Callable[[Grid], Grid | None]


def _shape(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0]) if g else 0


def _colors(g: Grid) -> set[int]:
    return {v for row in g for v in row}


def _plus_recolor_rule(task: dict) -> RelationalRule | None:
    introduced: set[int] | None = None
    for pair in task["train"]:
        if _shape(pair["input"]) != _shape(pair["output"]):
            return None
        new = _colors(pair["output"]) - _colors(pair["input"])
        introduced = set(new) if introduced is None else introduced & new
    if not introduced or len(introduced) != 1:
        return None
    target = next(iter(introduced))

    def apply(g: Grid) -> Grid:
        h, w = _shape(g)
        bg = _mode(g)
        out = [row[:] for row in g]
        hits: list[tuple[int, int]] = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                color = g[r][c]
                if color == bg:
                    continue
                pts = ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if all(g[rr][cc] == color for rr, cc in pts):
                    hits.append((r, c))
        for r, c in hits:
            for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                out[rr][cc] = target
        return out

    return RelationalRule(f"plus_recolor_{target}", apply)


def _symmetry_prune(g: Grid, axis: str) -> Grid:
    h, w = _shape(g)
    bg = _mode(g)
    out = [[bg for _ in range(w)] for _ in range(h)]

    for color in sorted(_colors(g) - {bg}):
        points = {(r, c) for r in range(h) for c in range(w) if g[r][c] == color}
        best: set[tuple[int, int]] = set()
        best_axis = -1

        if axis == "vertical":
            for doubled in range(2 * w - 1):
                subset = {(r, c) for r, c in points if (r, doubled - c) in points}
                if len(subset) > len(best) or (len(subset) == len(best) and doubled < best_axis):
                    best, best_axis = subset, doubled
        else:
            for doubled in range(2 * h - 1):
                subset = {(r, c) for r, c in points if (doubled - r, c) in points}
                if len(subset) > len(best) or (len(subset) == len(best) and doubled < best_axis):
                    best, best_axis = subset, doubled

        for r, c in best:
            out[r][c] = color

    return out


def _background_regions(g: Grid, bg: int) -> list[list[tuple[int, int]]]:
    h, w = _shape(g)
    seen = [[False] * w for _ in range(h)]
    regions: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or g[sr][sc] != bg:
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
                        and g[nr][nc] == bg
                    ):
                        seen[nr][nc] = True
                        q.append((nr, nc))
            regions.append(cells)
    return regions


def _candidate_border_maps(g: Grid) -> list[dict[int, int]]:
    h, w = _shape(g)
    if h < 2 or w < 2:
        return []
    bg = _mode(g)
    records: list[tuple[tuple[int, int, int], dict[int, int]]] = []

    for outer, inner in ((0, 1), (h - 1, h - 2)):
        mask = [g[outer][c] != bg and g[inner][c] != bg for c in range(w)]
        c = 0
        while c < w:
            if not mask[c]:
                c += 1
                continue
            end = c
            while end < w and mask[end]:
                end += 1
            if end - c >= 2:
                src = g[outer][c:end]
                dst = g[inner][c:end]
                mapping = dict(zip(src, dst))
                score = (end - c, len(set(src)) + len(set(dst)), sum(k != v for k, v in mapping.items()))
                records.append((score, mapping))
            c = end

    for outer, inner in ((0, 1), (w - 1, w - 2)):
        mask = [g[r][outer] != bg and g[r][inner] != bg for r in range(h)]
        r = 0
        while r < h:
            if not mask[r]:
                r += 1
                continue
            end = r
            while end < h and mask[end]:
                end += 1
            if end - r >= 2:
                src = [g[y][outer] for y in range(r, end)]
                dst = [g[y][inner] for y in range(r, end)]
                mapping = dict(zip(src, dst))
                score = (end - r, len(set(src)) + len(set(dst)), sum(k != v for k, v in mapping.items()))
                records.append((score, mapping))
            r = end

    records.sort(key=lambda x: x[0], reverse=True)
    if not records:
        return []
    best = records[0][0]
    return [mapping for score, mapping in records if score == best]


def _fill_holes(g: Grid, mapping: dict[int, int]) -> Grid:
    h, w = _shape(g)
    bg = _mode(g)
    out = [row[:] for row in g]
    for cells in _background_regions(g, bg):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        neighbors: set[int] = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != bg:
                    neighbors.add(g[nr][nc])
        if len(neighbors) == 1:
            outline = next(iter(neighbors))
            if outline in mapping:
                for r, c in cells:
                    out[r][c] = mapping[outline]
    return out


def _border_palette_rules() -> list[RelationalRule]:
    rules = []
    for idx in range(4):
        def apply(g: Grid, idx=idx) -> Grid | None:
            maps = _candidate_border_maps(g)
            if idx >= len(maps):
                return None
            return _fill_holes(g, maps[idx])
        rules.append(RelationalRule(f"border_palette_hole_fill_{idx}", apply))
    return rules


def _periodic_repair_line(values: list[int], max_period: int = 8) -> list[int]:
    if len(values) < 3:
        return values[:]
    best = values[:]
    best_score = len(values) + 1
    max_p = min(max_period, max(1, len(values) // 2))
    for p in range(1, max_p + 1):
        templates: list[int] = []
        for residue in range(p):
            bucket = values[residue::p]
            templates.append(Counter(bucket).most_common(1)[0][0])
        pred = [templates[i % p] for i in range(len(values))]
        score = sum(a != b for a, b in zip(values, pred))
        if score < best_score:
            best_score = score
            best = pred
    return best


def _periodic_rows(g: Grid) -> Grid:
    return [_periodic_repair_line(row) for row in g]


def _periodic_cols(g: Grid) -> Grid:
    h, w = _shape(g)
    cols = [[g[r][c] for r in range(h)] for c in range(w)]
    repaired = [_periodic_repair_line(col) for col in cols]
    return [[repaired[c][r] for c in range(w)] for r in range(h)]


def relational_rule_library(task: dict) -> list[RelationalRule]:
    rules: list[RelationalRule] = [
        RelationalRule("vertical_symmetry_prune", lambda g: _symmetry_prune(g, "vertical")),
        RelationalRule("horizontal_symmetry_prune", lambda g: _symmetry_prune(g, "horizontal")),
        RelationalRule("periodic_rows", _periodic_rows),
        RelationalRule("periodic_cols", _periodic_cols),
    ]
    plus = _plus_recolor_rule(task)
    if plus is not None:
        rules.append(plus)
    rules.extend(_border_palette_rules())
    return rules


def _fits(rule: RelationalRule, train: list[dict]) -> bool:
    changed = False
    for pair in train:
        try:
            pred = rule.apply(pair["input"])
        except Exception:
            return False
        if pred is None or pred != pair["output"]:
            return False
        changed = changed or pred != pair["input"]
    return changed


def relational_candidate_grids(task: dict) -> list[list[Grid]]:
    rules = [r for r in relational_rule_library(task) if _fits(r, task["train"])]
    pools: list[list[Grid]] = []
    for test in task["test"]:
        row: list[Grid] = []
        seen: set[tuple[tuple[int, ...], ...]] = set()
        for rule in rules:
            try:
                candidate = rule.apply(test["input"])
            except Exception:
                continue
            if candidate is None:
                continue
            key = tuple(tuple(x) for x in candidate)
            if key not in seen:
                seen.add(key)
                row.append(candidate)
        pools.append(row)
    return pools
