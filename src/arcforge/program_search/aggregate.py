from __future__ import annotations

from .types import Grid, ProgramArtifact


def _grid_key(grid: Grid):
    return tuple(tuple(row) for row in grid)


def candidate_pools_from_exact(
    programs: list[ProgramArtifact],
) -> list[list[Grid]]:
    exact = [p for p in programs if p.exact_train and p.report is not None]
    if not exact:
        return []

    num_tests = max(len(p.report.test_outputs) for p in exact)
    pools = [[] for _ in range(num_tests)]

    for test_index in range(num_tests):
        scored = {}
        for p in exact:
            if test_index >= len(p.report.test_outputs):
                continue
            grid = p.report.test_outputs[test_index]
            if grid is None:
                continue
            key = _grid_key(grid)
            rec = scored.setdefault(
                key,
                {
                    "grid": grid,
                    "support": 0,
                    "families": set(),
                    "review": 0.0,
                    "complexity": 0,
                },
            )
            rec["support"] += 1
            rec["families"].add(p.family)
            rec["review"] += p.reviewer_score
            rec["complexity"] += len(p.source)

        ranked = sorted(
            scored.values(),
            key=lambda r: (
                len(r["families"]),
                r["support"],
                r["review"],
                -r["complexity"],
            ),
            reverse=True,
        )
        pools[test_index] = [r["grid"] for r in ranked]

    return pools
