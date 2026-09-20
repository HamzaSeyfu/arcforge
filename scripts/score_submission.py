#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcforge.evaluation import score_submission


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--solutions", required=True)
    p.add_argument("--submission", required=True)
    args = p.parse_args()

    solutions = json.loads(Path(args.solutions).read_text())
    submission = json.loads(Path(args.submission).read_text())
    report = score_submission(submission, solutions)

    print(f"Rows: {report.solved_rows}/{report.rows}")
    print(f"Row exact pass@2: {100 * report.row_accuracy:.4f}%")
    print(f"Task-weight exact pass@2: {100 * report.task_weight_score:.4f}%")


if __name__ == "__main__":
    main()
