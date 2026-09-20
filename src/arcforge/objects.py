from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

from .symbolic import Grid, _mode


@dataclass(frozen=True)
class ArcObject:
    color: int
    cells: tuple[tuple[int, int], ...]
    r0: int
    r1: int
    c0: int
    c1: int
    canvas_h: int
    canvas_w: int

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def bbox_area(self) -> int:
        return self.height * self.width

    @property
    def density(self) -> float:
        return self.size / self.bbox_area

    @property
    def touches_border(self) -> bool:
        return (
            self.r0 == 0
            or self.c0 == 0
            or self.r1 == self.canvas_h - 1
            or self.c1 == self.canvas_w - 1
        )


def extract_objects(g: Grid) -> list[ArcObject]:
    h = len(g)
    w = len(g[0]) if g else 0
    bg = _mode(g)
    seen = [[False] * w for _ in range(h)]
    objects: list[ArcObject] = []

    for sr in range(h):
        for sc in range(w):
            color = g[sr][sc]
            if color == bg or seen[sr][sc]:
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
                        and g[nr][nc] == color
                    ):
                        seen[nr][nc] = True
                        q.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            objects.append(
                ArcObject(
                    color=color,
                    cells=tuple(cells),
                    r0=min(rs),
                    r1=max(rs),
                    c0=min(cs),
                    c1=max(cs),
                    canvas_h=h,
                    canvas_w=w,
                )
            )
    return objects


def _rank_selector(
    key: Callable[[ArcObject], object],
    reverse: bool,
    rank: int = 0,
) -> Callable[[list[ArcObject]], ArcObject | None]:
    def select(objects: list[ArcObject]) -> ArcObject | None:
        if not objects:
            return None
        ordered = sorted(objects, key=key, reverse=reverse)
        return ordered[rank] if rank < len(ordered) else None
    return select


def _unique_property_selector(
    getter: Callable[[ArcObject], object],
) -> Callable[[list[ArcObject]], ArcObject | None]:
    def select(objects: list[ArcObject]) -> ArcObject | None:
        counts = Counter(getter(obj) for obj in objects)
        uniques = [obj for obj in objects if counts[getter(obj)] == 1]
        return uniques[0] if len(uniques) == 1 else None
    return select


def _selector_library():
    return [
        ("largest", _rank_selector(lambda o: o.size, True)),
        ("smallest", _rank_selector(lambda o: o.size, False)),
        ("second_largest", _rank_selector(lambda o: o.size, True, 1)),
        ("largest_bbox", _rank_selector(lambda o: o.bbox_area, True)),
        ("smallest_bbox", _rank_selector(lambda o: o.bbox_area, False)),
        ("densest", _rank_selector(lambda o: o.density, True)),
        ("sparsest", _rank_selector(lambda o: o.density, False)),
        ("topmost", _rank_selector(lambda o: (o.r0, o.c0), False)),
        ("bottommost", _rank_selector(lambda o: (o.r1, o.c1), True)),
        ("leftmost", _rank_selector(lambda o: (o.c0, o.r0), False)),
        ("rightmost", _rank_selector(lambda o: (o.c1, o.r1), True)),
        ("unique_size", _unique_property_selector(lambda o: o.size)),
        ("unique_bbox", _unique_property_selector(lambda o: (o.height, o.width))),
        ("unique_color", _unique_property_selector(lambda o: o.color)),
    ]


def _crop_original(g: Grid, obj: ArcObject) -> Grid:
    return [row[obj.c0 : obj.c1 + 1] for row in g[obj.r0 : obj.r1 + 1]]


def _crop_isolated(g: Grid, obj: ArcObject) -> Grid:
    bg = _mode(g)
    out = [[bg] * obj.width for _ in range(obj.height)]
    for r, c in obj.cells:
        out[r - obj.r0][c - obj.c0] = obj.color
    return out


def _keep_on_canvas(g: Grid, obj: ArcObject) -> Grid:
    bg = _mode(g)
    out = [[bg] * len(g[0]) for _ in range(len(g))]
    for r, c in obj.cells:
        out[r][c] = obj.color
    return out


def _fill_bbox_on_canvas(g: Grid, obj: ArcObject) -> Grid:
    out = [row[:] for row in g]
    for r in range(obj.r0, obj.r1 + 1):
        for c in range(obj.c0, obj.c1 + 1):
            out[r][c] = obj.color
    return out


def _outline_bbox_on_canvas(g: Grid, obj: ArcObject) -> Grid:
    out = [row[:] for row in g]
    for c in range(obj.c0, obj.c1 + 1):
        out[obj.r0][c] = obj.color
        out[obj.r1][c] = obj.color
    for r in range(obj.r0, obj.r1 + 1):
        out[r][obj.c0] = obj.color
        out[r][obj.c1] = obj.color
    return out


@dataclass(frozen=True)
class ObjectProgram:
    name: str
    selector: Callable[[list[ArcObject]], ArcObject | None]
    renderer: Callable[[Grid, ArcObject], Grid]
    recolor: int | None = None

    def apply(self, g: Grid) -> Grid | None:
        obj = self.selector(extract_objects(g))
        if obj is None:
            return None
        result = self.renderer(g, obj)
        if self.recolor is not None:
            result = [
                [self.recolor if v == obj.color else v for v in row]
                for row in result
            ]
        return result


def synthesize_object_programs(task: dict) -> list[ObjectProgram]:
    train = task["train"]
    renderers = [
        ("crop_original", _crop_original),
        ("crop_isolated", _crop_isolated),
        ("keep_on_canvas", _keep_on_canvas),
        ("fill_bbox", _fill_bbox_on_canvas),
        ("outline_bbox", _outline_bbox_on_canvas),
    ]

    introduced: set[int] | None = None
    for pair in train:
        new = {v for row in pair["output"] for v in row} - {v for row in pair["input"] for v in row}
        introduced = set(new) if introduced is None else introduced & new
    recolors = [None] + sorted(introduced or set())

    programs: list[ObjectProgram] = []
    for selector_name, selector in _selector_library():
        for renderer_name, renderer in renderers:
            for recolor in recolors:
                program = ObjectProgram(
                    name=f"{selector_name}:{renderer_name}"
                    + (f":recolor_{recolor}" if recolor is not None else ""),
                    selector=selector,
                    renderer=renderer,
                    recolor=recolor,
                )
                good = True
                changed = False
                for pair in train:
                    try:
                        pred = program.apply(pair["input"])
                    except Exception:
                        pred = None
                    if pred != pair["output"]:
                        good = False
                        break
                    changed = changed or pred != pair["input"]
                if good and changed:
                    programs.append(program)
    return programs


def object_candidate_grids(task: dict) -> list[list[Grid]]:
    programs = synthesize_object_programs(task)
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
