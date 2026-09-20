#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcforge.symbolic import candidate_grids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-dir", required=True)
    p.add_argument("--task-list", required=True)
    p.add_argument("--qwen-oracle-rows")
    p.add_argument("--output", default="symbolic_benchmark.json")
    args = p.parse_args()

    task_ids = [
        line.strip()
        for line in Path(args.task_list).read_text().splitlines()
        if line.strip()
    ]
    qwen_oracle: set[str] = set()
    if args.qwen_oracle_rows:
        qwen_oracle = set(json.loads(Path(args.qwen_oracle_rows).read_text()))

    total = 0
    solved: list[str] = []
    unique: list[str] = []
    generated_rows = 0
    candidate_counts: dict[str, int] = {}

    for task_id in task_ids:
        task = json.loads((Path(args.evaluation_dir) / f"{task_id}.json").read_text())
        pools = candidate_grids(task)

        for i, test_case in enumerate(task["test"]):
            total += 1
            key = f"{task_id}:{i}"
            candidates = pools[i]
            candidate_counts[key] = len(candidates)
            if candidates:
                generated_rows += 1

            truth = test_case.get("output")
            if truth is not None and any(candidate == truth for candidate in candidates):
                solved.append(key)
                if key not in qwen_oracle:
                    unique.append(key)

    report = {
        "tasks": len(task_ids),
        "rows": total,
        "rows_with_candidates": generated_rows,
        "symbolic_solved_rows": len(solved),
        "symbolic_row_accuracy": len(solved) / total if total else 0.0,
        "unique_over_qwen_oracle": len(unique),
        "unique_rows": unique,
        "solved_rows": solved,
        "candidate_counts": candidate_counts,
        "qwen_oracle_rows": len(qwen_oracle),
        "oracle_union_rows": len(qwen_oracle | set(solved)),
        "oracle_union_accuracy": len(qwen_oracle | set(solved)) / total if total else 0.0,
    }

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"candidate_counts"}}, indent=2))


if __name__ == "__main__":
    main()
