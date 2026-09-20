from __future__ import annotations

from typing import Any


def validate_grid(grid: Any) -> None:
    if not isinstance(grid, list) or not grid:
        raise ValueError("grid must be a non-empty list of rows")
    if len(grid) > 30:
        raise ValueError("grid has more than 30 rows")
    if not all(isinstance(row, list) and row for row in grid):
        raise ValueError("each row must be a non-empty list")
    width = len(grid[0])
    if width > 30 or any(len(row) != width for row in grid):
        raise ValueError("grid must be rectangular and <= 30 columns")
    for row in grid:
        for value in row:
            if type(value) is not int or not (0 <= value <= 9):
                raise ValueError("ARC cells must be integer colors in [0, 9]")


def validate_submission(submission: dict, challenges: dict) -> None:
    if set(submission) != set(challenges):
        missing = set(challenges) - set(submission)
        extra = set(submission) - set(challenges)
        raise ValueError(f"task ids differ: missing={len(missing)} extra={len(extra)}")

    for task_id, task in challenges.items():
        rows = submission[task_id]
        if len(rows) != len(task["test"]):
            raise ValueError(f"{task_id}: wrong number of output rows")
        for row in rows:
            if set(row) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{task_id}: each row needs attempt_1 and attempt_2")
            validate_grid(row["attempt_1"])
            validate_grid(row["attempt_2"])
