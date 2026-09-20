#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcforge.symbolic import candidate_grids
from arcforge.synthesis import synthesis_candidate_grids


def _dedupe(grids):
    out = []
    seen = set()
    for grid in grids:
        key = tuple(tuple(row) for row in grid)
        if key not in seen:
            seen.add(key)
            out.append(grid)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-dir", required=True)
    p.add_argument("--task-list", required=True)
    p.add_argument("--qwen-oracle-rows", required=True)
    p.add_argument("--output", default="experiments/latest_portfolio_benchmark.json")
    args = p.parse_args()

    ids = [x.strip() for x in Path(args.task_list).read_text().splitlines() if x.strip()]
    qwen = set(json.loads(Path(args.qwen_oracle_rows).read_text()))

    total = 0
    symbolic_hits = set()
    synthesis_hits = set()
    union_hits = set()
    candidate_counts = {}

    for task_id in ids:
        task = json.loads((Path(args.evaluation_dir) / f"{task_id}.json").read_text())
        symbolic = candidate_grids(task)
        synthesis = synthesis_candidate_grids(task, max_depth=2)

        for i, test_case in enumerate(task["test"]):
            total += 1
            key = f"{task_id}:{i}"
            truth = test_case.get("output")
            sp = symbolic[i]
            sy = synthesis[i]
            merged = _dedupe(sp + sy)
            candidate_counts[key] = {
                "symbolic": len(sp),
                "synthesis": len(sy),
                "local_union": len(merged),
            }

            if truth is not None and any(g == truth for g in sp):
                symbolic_hits.add(key)
            if truth is not None and any(g == truth for g in sy):
                synthesis_hits.add(key)
            if truth is not None and any(g == truth for g in merged):
                union_hits.add(key)

    full_oracle = qwen | union_hits
    report = {
        "tasks": len(ids),
        "rows": total,
        "qwen_oracle_rows": len(qwen),
        "qwen_oracle_accuracy": len(qwen) / total,
        "symbolic_rows": len(symbolic_hits),
        "synthesis_rows": len(synthesis_hits),
        "local_union_rows": len(union_hits),
        "unique_symbolic_over_qwen": sorted(symbolic_hits - qwen),
        "unique_synthesis_over_qwen": sorted(synthesis_hits - qwen),
        "unique_local_union_over_qwen": sorted(union_hits - qwen),
        "qwen_plus_local_oracle_rows": len(full_oracle),
        "qwen_plus_local_oracle_accuracy": len(full_oracle) / total,
        "candidate_counts": candidate_counts,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "candidate_counts"}, indent=2))


if __name__ == "__main__":
    main()
