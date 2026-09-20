from __future__ import annotations

from dataclasses import dataclass

Grid = list[list[int]]
Submission = dict[str, list[dict[str, Grid]]]
Solutions = dict[str, list[Grid]]


def exact_grid(a: Grid, b: Grid) -> bool:
    return a == b


@dataclass(frozen=True)
class ScoreReport:
    rows: int
    solved_rows: int
    row_accuracy: float
    tasks: int
    task_weight_score: float


def score_submission(submission: Submission, solutions: Solutions) -> ScoreReport:
    """Score exact pass@2 using both row-weighted and task-weighted views."""
    rows = 0
    solved_rows = 0
    task_weight_total = 0.0

    missing = set(solutions) - set(submission)
    if missing:
        raise ValueError(f"Missing {len(missing)} task ids, e.g. {sorted(missing)[:5]}")

    for task_id, truths in solutions.items():
        preds = submission[task_id]
        if len(preds) != len(truths):
            raise ValueError(
                f"{task_id}: expected {len(truths)} output rows, got {len(preds)}"
            )

        per_row_weight = 1.0 / len(truths)
        for pred, truth in zip(preds, truths):
            rows += 1
            hit = pred["attempt_1"] == truth or pred["attempt_2"] == truth
            if hit:
                solved_rows += 1
                task_weight_total += per_row_weight

    tasks = len(solutions)
    return ScoreReport(
        rows=rows,
        solved_rows=solved_rows,
        row_accuracy=solved_rows / rows if rows else 0.0,
        tasks=tasks,
        task_weight_score=task_weight_total / tasks if tasks else 0.0,
    )


def solved_row_keys(submission: Submission, solutions: Solutions) -> set[str]:
    solved: set[str] = set()
    for task_id, truths in solutions.items():
        preds = submission[task_id]
        for i, (pred, truth) in enumerate(zip(preds, truths)):
            if pred["attempt_1"] == truth or pred["attempt_2"] == truth:
                solved.add(f"{task_id}:{i}")
    return solved
