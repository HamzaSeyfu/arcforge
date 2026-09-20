from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

from .symbolic import Grid, _mode


@dataclass(frozen=True)
class KeyLine:
    axis: str
    index: int
    sequence: tuple[int, ...]


@dataclass(frozen=True)
class Frame:
    marker: int
    border: int
    r0: int
    r1: int
    c0: int
    c1: int
    marker_rows: tuple[int, ...]
    marker_cols: tuple[int, ...]


def _shape(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0]) if g else 0


def _parse_alternating(line: list[int], bg: int) -> tuple[int, ...] | None:
    """
    Accept a sequence encoded as bg,color,bg,color,... followed by background.
    Also accept the reverse orientation.
    """
    def parse(values: list[int]) -> tuple[int, ...] | None:
        seq: list[int] = []
        i = 0
        while i < len(values):
            if values[i] != bg:
                return None
            if i + 1 >= len(values):
                break
            if values[i + 1] == bg:
                if any(v != bg for v in values[i:]):
                    return None
                break
            seq.append(values[i + 1])
            i += 2
        return tuple(seq) if len(seq) >= 2 else None

    direct = parse(line)
    if direct is not None:
        return direct
    rev = parse(line[::-1])
    return tuple(reversed(rev)) if rev is not None else None


def _candidate_key_lines(g: Grid) -> list[KeyLine]:
    h, w = _shape(g)
    bg = _mode(g)
    result: list[KeyLine] = []

    for r in range(h):
        seq = _parse_alternating(g[r], bg)
        if seq is not None:
            result.append(KeyLine("row", r, seq))

    for c in range(w):
        col = [g[r][c] for r in range(h)]
        seq = _parse_alternating(col, bg)
        if seq is not None:
            result.append(KeyLine("col", c, seq))

    return result


def _component_cells(
    g: Grid,
    bg: int,
    key: KeyLine,
) -> list[list[tuple[int, int]]]:
    h, w = _shape(g)
    seen = [[False] * w for _ in range(h)]
    comps: list[list[tuple[int, int]]] = []

    def excluded(r: int, c: int) -> bool:
        return (key.axis == "row" and r == key.index) or (
            key.axis == "col" and c == key.index
        )

    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or excluded(sr, sc) or g[sr][sc] == bg:
                continue
            q = deque([(sr, sc)])
            seen[sr][sc] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr][nc] or excluded(nr, nc) or g[nr][nc] == bg:
                        continue
                    seen[nr][nc] = True
                    q.append((nr, nc))
            comps.append(cells)
    return comps


def _frame_from_component(
    g: Grid,
    cells: list[tuple[int, int]],
    bg: int,
) -> Frame | None:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    perimeter = (
        [g[r0][c] for c in range(c0, c1 + 1)]
        + [g[r1][c] for c in range(c0, c1 + 1)]
        + [g[r][c0] for r in range(r0 + 1, r1)]
        + [g[r][c1] for r in range(r0 + 1, r1)]
    )
    border_counts = Counter(v for v in perimeter if v != bg)
    if not border_counts:
        return None
    border, n = border_counts.most_common(1)[0]
    if n < int(0.8 * len(perimeter)):
        return None

    marker_cells: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = g[r][c]
            if value not in (bg, border):
                marker_cells.append((r, c, value))

    marker_colors = {v for _, _, v in marker_cells}
    if len(marker_colors) != 1:
        return None

    marker = next(iter(marker_colors))
    return Frame(
        marker=marker,
        border=border,
        r0=r0,
        r1=r1,
        c0=c0,
        c1=c1,
        marker_rows=tuple(sorted({r for r, _, _ in marker_cells})),
        marker_cols=tuple(sorted({c for _, c, _ in marker_cells})),
    )


def _find_frames(g: Grid, key: KeyLine) -> list[Frame]:
    bg = _mode(g)
    frames = []
    for cells in _component_cells(g, bg, key):
        frame = _frame_from_component(g, cells, bg)
        if frame is not None:
            frames.append(frame)
    return frames


def _adjacency(a: Frame, b: Frame) -> str | None:
    row_overlap = not (a.r1 < b.r0 or b.r1 < a.r0)
    col_overlap = not (a.c1 < b.c0 or b.c1 < a.c0)
    if row_overlap and not col_overlap:
        return "h"
    if col_overlap and not row_overlap:
        return "v"
    return None


def _gap_distance(a: Frame, b: Frame) -> int | None:
    kind = _adjacency(a, b)
    if kind == "h":
        return (
            b.c0 - a.c1 - 1
            if a.c1 < b.c0
            else a.c0 - b.c1 - 1
        )
    if kind == "v":
        return (
            b.r0 - a.r1 - 1
            if a.r1 < b.r0
            else a.r0 - b.r1 - 1
        )
    return None


def _walk(
    frames: list[Frame],
    sequence: tuple[int, ...],
) -> list[Frame] | None:
    by_marker: dict[int, list[Frame]] = {}
    for frame in frames:
        by_marker.setdefault(frame.marker, []).append(frame)

    starts = by_marker.get(sequence[0], [])
    best: list[Frame] | None = None

    for start in starts:
        path = [start]
        used = {start}
        for marker in sequence[1:]:
            choices = []
            for frame in by_marker.get(marker, []):
                if frame in used:
                    continue
                distance = _gap_distance(path[-1], frame)
                if distance is not None and distance >= 0:
                    choices.append((distance, frame.r0, frame.c0, frame))
            if not choices:
                break
            choices.sort(key=lambda x: x[:3])
            nxt = choices[0][3]
            path.append(nxt)
            used.add(nxt)

        if best is None or len(path) > len(best):
            best = path
        if best is not None and len(best) == len(sequence):
            return best

    return best if best and len(best) >= 2 else None


def _project(g: Grid, key: KeyLine) -> Grid | None:
    frames = _find_frames(g, key)
    path = _walk(frames, key.sequence)
    if path is None:
        return None

    bg = _mode(g)
    out = [row[:] for row in g]

    for i, (a, b) in enumerate(zip(path, path[1:])):
        kind = _adjacency(a, b)
        if kind is None:
            continue
        color = key.sequence[min(i, len(key.sequence) - 1)]

        if kind == "h":
            left, right = (a, b) if a.c1 < b.c0 else (b, a)
            cols = range(left.c1 + 1, right.c0)
            marker_rows = a.marker_rows
            for r in marker_rows:
                for c in cols:
                    if out[r][c] == bg:
                        out[r][c] = color
        else:
            top, bottom = (a, b) if a.r1 < b.r0 else (b, a)
            rows = range(top.r1 + 1, bottom.r0)
            marker_cols = a.marker_cols
            for c in marker_cols:
                for r in rows:
                    if out[r][c] == bg:
                        out[r][c] = color

    return out


@dataclass(frozen=True)
class FrameWalkRule:
    key_rank: int

    @property
    def name(self) -> str:
        return f"frame_walk:key_rank_{self.key_rank}"

    def apply(self, g: Grid) -> Grid | None:
        keys = _candidate_key_lines(g)
        if self.key_rank >= len(keys):
            return None
        return _project(g, keys[self.key_rank])


def _fits(rule: FrameWalkRule, train: list[dict]) -> bool:
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


def framewalk_candidate_grids(task: dict) -> list[list[Grid]]:
    if not task["train"]:
        return [[] for _ in task["test"]]

    max_keys = max(
        (len(_candidate_key_lines(pair["input"])) for pair in task["train"]),
        default=0,
    )
    rules = [
        FrameWalkRule(rank)
        for rank in range(max_keys)
        if _fits(FrameWalkRule(rank), task["train"])
    ]

    pools: list[list[Grid]] = []
    for test in task["test"]:
        row: list[Grid] = []
        seen = set()
        for rule in rules:
            candidate = rule.apply(test["input"])
            if candidate is None:
                continue
            key = tuple(tuple(r) for r in candidate)
            if key not in seen:
                seen.add(key)
                row.append(candidate)
        pools.append(row)
    return pools
