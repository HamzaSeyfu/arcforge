from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

from .evaluation import Submission, Solutions, solved_row_keys


@dataclass(frozen=True)
class ComplementarityReport:
    solved_a: int
    solved_b: int
    shared: int
    unique_a: int
    unique_b: int
    oracle_union: int


@dataclass(frozen=True)
class PortfolioStep:
    solver: str
    added: int
    cumulative: int


@dataclass(frozen=True)
class PortfolioCoverageReport:
    total_rows: int
    per_solver: dict[str, int]
    oracle_union: int
    greedy_order: tuple[PortfolioStep, ...]

    @property
    def oracle_union_accuracy(self) -> float:
        return self.oracle_union / self.total_rows if self.total_rows else 0.0


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


def solved_sets(
    submissions: Mapping[str, Submission],
    solutions: Solutions,
) -> dict[str, set[str]]:
    """Return exact-solved output-row keys for every named solver."""
    if not submissions:
        raise ValueError("at least one solver submission is required")
    return {
        name: solved_row_keys(submission, solutions)
        for name, submission in submissions.items()
    }


def pairwise_complementarity(
    submissions: Mapping[str, Submission],
    solutions: Solutions,
) -> dict[tuple[str, str], ComplementarityReport]:
    """Compute complementarity for every unordered pair of named solvers."""
    names = sorted(submissions)
    return {
        (a, b): complementarity(submissions[a], submissions[b], solutions)
        for a, b in combinations(names, 2)
    }


def greedy_solver_order(
    submissions: Mapping[str, Submission],
    solutions: Solutions,
) -> tuple[PortfolioStep, ...]:
    """Order solvers by marginal exact coverage.

    At each step, pick the solver adding the largest number of previously unsolved
    output rows. Ties are broken lexicographically for reproducibility.
    """
    coverage = solved_sets(submissions, solutions)
    remaining = set(coverage)
    solved: set[str] = set()
    steps: list[PortfolioStep] = []

    while remaining:
        ranked = sorted(
            remaining,
            key=lambda name: (-len(coverage[name] - solved), name),
        )
        chosen = ranked[0]
        new_rows = coverage[chosen] - solved
        solved |= coverage[chosen]
        steps.append(
            PortfolioStep(
                solver=chosen,
                added=len(new_rows),
                cumulative=len(solved),
            )
        )
        remaining.remove(chosen)

    return tuple(steps)


def portfolio_coverage(
    submissions: Mapping[str, Submission],
    solutions: Solutions,
) -> PortfolioCoverageReport:
    """Summarize exact coverage of an arbitrary solver portfolio."""
    coverage = solved_sets(submissions, solutions)
    union: set[str] = set()
    for rows in coverage.values():
        union |= rows

    total_rows = sum(len(truths) for truths in solutions.values())
    return PortfolioCoverageReport(
        total_rows=total_rows,
        per_solver={name: len(rows) for name, rows in sorted(coverage.items())},
        oracle_union=len(union),
        greedy_order=greedy_solver_order(submissions, solutions),
    )
