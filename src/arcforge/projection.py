from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .symbolic import Grid, _mode


@dataclass(frozen=True)
class ProjectionRule:
    name: str
    apply: Callable[[Grid], Grid | None]


def _shape(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0]) if g else 0


def _detect_dividers(g: Grid) -> list[tuple[str, int, int]]:
    """Detect strong non-background row/column dividers without assuming color 0."""
    h, w = _shape(g)
    bg = _mode(g)
    out: list[tuple[str, int, int]] = []

    for c in range(w):
        vals = [g[r][c] for r in range(h)]
        counts = Counter(vals)
        for color, n in counts.most_common():
            if color == bg:
                continue
            if n >= max(2, h - 2):
                out.append(("col", c, color))
                break

    for r in range(h):
        vals = g[r]
        counts = Counter(vals)
        for color, n in counts.most_common():
            if color == bg:
                continue
            if n >= max(2, w - 2):
                out.append(("row", r, color))
                break
    return out


def _runs_outward(
    line: list[int],
    divider: int,
    side: int,
    background: int,
    divider_color: int,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = divider + side
    current: int | None = None
    length = 0

    while 0 <= i < len(line):
        value = line[i]
        meaningful = value not in (background, divider_color)
        if not meaningful:
            if current is not None:
                runs.append((length, current))
                current = None
                length = 0
        elif current == value:
            length += 1
        else:
            if current is not None:
                runs.append((length, current))
            current = value
            length = 1
        i += side

    if current is not None:
        runs.append((length, current))
    return [(k, c) for k, c in runs if k > 0]


def _periodic_project_line(
    line: list[int],
    divider: int,
    target_side: int,
    bars: list[tuple[int, int]],
    background: int,
) -> list[int]:
    out = line[:]
    i = divider + target_side
    distance = 0
    while 0 <= i < len(out):
        value = background
        for period, color in bars:
            if period > 0 and distance % period == 0:
                value = color
                break
        out[i] = value
        i += target_side
        distance += 1
    return out


def _divider_periodic_variant(
    g: Grid,
    divider: tuple[str, int, int],
    source_side: int,
) -> Grid | None:
    h, w = _shape(g)
    bg = _mode(g)
    kind, idx, divider_color = divider
    target_side = -source_side
    out = [row[:] for row in g]
    generated_any = False

    if kind == "col":
        for r in range(h):
            line = out[r][:]
            bars = _runs_outward(line, idx, source_side, bg, divider_color)
            if not bars:
                continue
            # source side must contain evidence and target side must be mostly empty
            target_idxs = range(idx + 1, w) if target_side == 1 else range(0, idx)
            target_non_bg = sum(
                1 for c in target_idxs if line[c] not in (bg, divider_color)
            )
            if target_non_bg > max(1, len(list(target_idxs)) // 5):
                continue
            out[r] = _periodic_project_line(
                line, idx, target_side, bars, bg
            )
            generated_any = True
    else:
        for c in range(w):
            line = [out[r][c] for r in range(h)]
            bars = _runs_outward(line, idx, source_side, bg, divider_color)
            if not bars:
                continue
            target_idxs = range(idx + 1, h) if target_side == 1 else range(0, idx)
            target_non_bg = sum(
                1 for r in target_idxs if line[r] not in (bg, divider_color)
            )
            if target_non_bg > max(1, len(list(target_idxs)) // 5):
                continue
            line = _periodic_project_line(line, idx, target_side, bars, bg)
            for r, value in enumerate(line):
                out[r][c] = value
            generated_any = True

    return out if generated_any else None


def divider_periodic_rules(task: dict) -> list[ProjectionRule]:
    """Infer divider orientation/side from demonstrations by exact execution fit."""
    if not task["train"]:
        return []
    first = task["train"][0]["input"]
    candidates: list[ProjectionRule] = []
    for divider in _detect_dividers(first):
        for source_side in (-1, 1):
            name = f"divider_periodic:{divider[0]}:{divider[1]}:src{source_side}"
            rule = ProjectionRule(
                name,
                lambda g, divider=divider, source_side=source_side:
                    _divider_periodic_variant(g, divider, source_side),
            )
            if _fits(rule, task["train"]):
                candidates.append(rule)
    return candidates


def _uniform_separator_indices(
    g: Grid,
    axis: str,
) -> list[tuple[int, int]]:
    h, w = _shape(g)
    result = []
    if axis == "row":
        for r in range(h):
            if len(set(g[r])) == 1:
                result.append((r, g[r][0]))
    else:
        for c in range(w):
            vals = [g[r][c] for r in range(h)]
            if len(set(vals)) == 1:
                result.append((c, vals[0]))
    return result


def _split_by_separator(
    g: Grid,
    axis: str,
    separator_color: int,
) -> list[Grid]:
    h, w = _shape(g)
    indices = {
        idx
        for idx, color in _uniform_separator_indices(g, axis)
        if color == separator_color
    }
    blocks: list[Grid] = []

    if axis == "row":
        start = 0
        for r in range(h):
            if r in indices:
                if start < r:
                    blocks.append([row[:] for row in g[start:r]])
                start = r + 1
        if start < h:
            blocks.append([row[:] for row in g[start:]])
    else:
        start = 0
        for c in range(w):
            if c in indices:
                if start < c:
                    blocks.append([row[start:c] for row in g])
                start = c + 1
        if start < w:
            blocks.append([row[start:] for row in g])
    return [b for b in blocks if b and b[0]]


def _solid_color(block: Grid) -> int | None:
    vals = {v for row in block for v in row}
    if len(vals) == 1:
        return next(iter(vals))
    return None


def _trim_background(block: Grid, background: int) -> Grid | None:
    rows = [r for r, row in enumerate(block) if any(v != background for v in row)]
    cols = [
        c
        for c in range(len(block[0]))
        if any(block[r][c] != background for r in range(len(block)))
    ]
    if not rows or not cols:
        return None
    return [
        [block[r][c] for c in range(min(cols), max(cols) + 1)]
        for r in range(min(rows), max(rows) + 1)
    ]


def _block_tiling_variant(
    g: Grid,
    axis: str,
    separator_color: int,
    count_divisor: int,
) -> Grid | None:
    bg = _mode(g)
    blocks = _split_by_separator(g, axis, separator_color)
    if len(blocks) != 4:
        return None

    nonsolid = [b for b in blocks if _solid_color(b) is None]
    solid = [b for b in blocks if _solid_color(b) is not None]
    if len(nonsolid) != 2 or len(solid) != 2:
        return None

    shape_block, counter_block = nonsolid
    color_a = _solid_color(solid[0])
    color_b = _solid_color(solid[1])
    if color_a is None or color_b is None:
        return None

    core = _trim_background(shape_block, bg)
    counter = _trim_background(counter_block, bg)
    if core is None or counter is None:
        return None

    dots = sum(v != bg for row in counter for v in row)
    if count_divisor <= 0 or dots == 0 or dots % count_divisor:
        return None
    count = dots // count_divisor
    if count <= 0 or count > 12:
        return None

    recolored = [
        [color_a if v != bg else color_b for v in row]
        for row in core
    ]

    if axis == "row":
        width = len(recolored[0])
        out: Grid = []
        for i in range(count):
            out.extend(row[:] for row in recolored)
            if i + 1 < count:
                out.append([color_b] * width)
        return out

    height = len(recolored)
    out = [[] for _ in range(height)]
    for i in range(count):
        for r in range(height):
            out[r].extend(recolored[r])
            if i + 1 < count:
                out[r].append(color_b)
    return out


def block_tiling_rules(task: dict) -> list[ProjectionRule]:
    first = task["train"][0]["input"] if task["train"] else []
    candidates: list[ProjectionRule] = []
    for axis in ("row", "col"):
        colors = Counter(
            color for _, color in _uniform_separator_indices(first, axis)
        )
        for separator_color, freq in colors.items():
            if freq < 2:
                continue
            for divisor in range(1, 7):
                rule = ProjectionRule(
                    f"block_tile:{axis}:sep{separator_color}:div{divisor}",
                    lambda g, axis=axis, separator_color=separator_color, divisor=divisor:
                        _block_tiling_variant(g, axis, separator_color, divisor),
                )
                if _fits(rule, task["train"]):
                    candidates.append(rule)
    return candidates


def _cell_layout(
    g: Grid,
    cell_h: int,
    cell_w: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = _shape(g)
    row_sep = [len(set(g[r])) == 1 for r in range(h)]
    col_sep = [
        len({g[r][c] for r in range(h)}) == 1
        for c in range(w)
    ]

    rows = []
    r = 0
    while r + cell_h <= h:
        if row_sep[r]:
            r += 1
            continue
        if any(row_sep[r + k] for k in range(cell_h)):
            r += 1
            continue
        rows.append((r, r + cell_h))
        r += cell_h

    cols = []
    c = 0
    while c + cell_w <= w:
        if col_sep[c]:
            c += 1
            continue
        if any(col_sep[c + k] for k in range(cell_w)):
            c += 1
            continue
        cols.append((c, c + cell_w))
        c += cell_w

    if len(rows) < 2 or len(cols) < 2:
        return None
    return rows, cols


def _cell_template_completion(
    g: Grid,
    cell_h: int,
    cell_w: int,
) -> Grid | None:
    layout = _cell_layout(g, cell_h, cell_w)
    if layout is None:
        return None
    cell_rows, cell_cols = layout
    cells: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}

    for ri, (r0, r1) in enumerate(cell_rows):
        for ci, (c0, c1) in enumerate(cell_cols):
            cells[(ri, ci)] = tuple(
                tuple(g[r][c0:c1])
                for r in range(r0, r1)
            )

    counts = Counter(cells.values())
    normal, normal_count = counts.most_common(1)[0]
    specials = [(pos, cell) for pos, cell in cells.items() if cell != normal]
    if not specials:
        return None

    exemplar = specials[0][1]
    normal_colors = {v for row in normal for v in row}
    exemplar_colors = {v for row in exemplar for v in row}
    marker_candidates = exemplar_colors - normal_colors
    if len(marker_candidates) == 1:
        marker = next(iter(marker_candidates))
    else:
        diffs = Counter(
            exemplar[r][c]
            for r in range(cell_h)
            for c in range(cell_w)
            if exemplar[r][c] != normal[r][c]
        )
        if not diffs:
            return None
        marker = diffs.most_common(1)[0][0]

    template = {
        (r, c)
        for r in range(cell_h)
        for c in range(cell_w)
        if exemplar[r][c] == marker
    }
    if not template:
        return None

    special_positions = {pos for pos, _ in specials}
    nr, nc = len(cell_rows), len(cell_cols)
    best: tuple[int, set[tuple[int, int]]] | None = None

    for dr in range(-cell_h, nr + 1):
        for dc in range(-cell_w, nc + 1):
            projected = {(r + dr, c + dc) for r, c in template}
            if not special_positions.issubset(projected):
                continue
            if not all(0 <= r < nr and 0 <= c < nc for r, c in projected):
                continue
            extra = projected - special_positions
            score = len(extra)
            if best is None or score < best[0]:
                best = (score, extra)

    if best is None:
        return None

    out = [row[:] for row in g]
    for ri, ci in best[1]:
        r0, r1 = cell_rows[ri]
        c0, c1 = cell_cols[ci]
        for r in range(r0, r1):
            for c in range(c0, c1):
                out[r][c] = exemplar[r - r0][c - c0]
    return out


def cell_completion_rules(task: dict) -> list[ProjectionRule]:
    candidates: list[ProjectionRule] = []
    for cell_h in range(2, 6):
        for cell_w in range(2, 6):
            rule = ProjectionRule(
                f"cell_template_completion:{cell_h}x{cell_w}",
                lambda g, cell_h=cell_h, cell_w=cell_w:
                    _cell_template_completion(g, cell_h, cell_w),
            )
            if _fits(rule, task["train"]):
                candidates.append(rule)
    return candidates


def _fits(rule: ProjectionRule, train: list[dict]) -> bool:
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


def projection_rule_library(task: dict) -> list[ProjectionRule]:
    return (
        divider_periodic_rules(task)
        + block_tiling_rules(task)
        + cell_completion_rules(task)
    )


def projection_candidate_grids(task: dict) -> list[list[Grid]]:
    rules = projection_rule_library(task)
    pools: list[list[Grid]] = []
    for test in task["test"]:
        row: list[Grid] = []
        seen = set()
        for rule in rules:
            try:
                candidate = rule.apply(test["input"])
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
