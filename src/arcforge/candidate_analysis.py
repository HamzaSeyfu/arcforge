from __future__ import annotations

from dataclasses import dataclass

from .evaluation import Grid, Solutions


CandidatePools = dict[str, list[list[Grid]]]


@dataclass(frozen=True)
class CandidateCoverageReport:
    rows: int
    rows_with_any_candidate: int
    oracle_solved_rows: int
    oracle_row_accuracy: float
    missing_candidate_rows: int


def candidate_oracle_coverage(
    candidate_pools: CandidatePools,
    solutions: Solutions,
) -> CandidateCoverageReport:
    """
    Measure whether the correct answer exists anywhere in each candidate pool.

    candidate_pools format:
      {
        task_id: [
          [candidate_grid_1, candidate_grid_2, ...],  # test output 0
          [candidate_grid_1, ...],                    # test output 1
        ]
      }

    This is diagnostic only. It is an oracle upper bound and must never be used
    as a hidden-set selector.
    """
    rows = 0
    rows_with_any_candidate = 0
    oracle_solved_rows = 0

    for task_id, truths in solutions.items():
        task_pools = candidate_pools.get(task_id, [])
        for i, truth in enumerate(truths):
            rows += 1
            candidates = task_pools[i] if i < len(task_pools) else []
            if candidates:
                rows_with_any_candidate += 1
            if any(candidate == truth for candidate in candidates):
                oracle_solved_rows += 1

    return CandidateCoverageReport(
        rows=rows,
        rows_with_any_candidate=rows_with_any_candidate,
        oracle_solved_rows=oracle_solved_rows,
        oracle_row_accuracy=oracle_solved_rows / rows if rows else 0.0,
        missing_candidate_rows=rows - rows_with_any_candidate,
    )


def oracle_union_coverage(
    pools: list[CandidatePools],
    solutions: Solutions,
) -> CandidateCoverageReport:
    merged: CandidatePools = {}

    for task_id, truths in solutions.items():
        merged[task_id] = [[] for _ in truths]

    for pool in pools:
        for task_id, rows in pool.items():
            if task_id not in merged:
                continue
            for i, candidates in enumerate(rows):
                if i >= len(merged[task_id]):
                    break
                seen = {tuple(map(tuple, g)) for g in merged[task_id][i]}
                for candidate in candidates:
                    key = tuple(map(tuple, candidate))
                    if key not in seen:
                        merged[task_id][i].append(candidate)
                        seen.add(key)

    return candidate_oracle_coverage(merged, solutions)
