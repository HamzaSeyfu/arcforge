from __future__ import annotations

from dataclasses import dataclass

from .evaluation import Submission, Solutions, solved_row_keys


@dataclass(frozen=True)
class ComplementarityReport:
    solved_a: int
    solved_b: int
    shared: int
    unique_a: int
    unique_b: int
    oracle_union: int


def complementarity(
    a: Submission,
    b: Submission,
    solutions: Solutions,
) -> ComplementarityReport:
    sa = solved_row_keys(a, solutions)
    sb = solved_row_keys(b, solutions)
    return ComplementarityReport(
        solved_a=len(sa),
        solved_b=len(sb),
        shared=len(sa & sb),
        unique_a=len(sa - sb),
        unique_b=len(sb - sa),
        oracle_union=len(sa | sb),
    )
