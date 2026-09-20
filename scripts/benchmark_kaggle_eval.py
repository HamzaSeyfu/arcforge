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
    p.add_argument("--challenges", required=True)
    p.add_argument("--solutions", required=True)
    p.add_argument("--qwen-oracle-rows", required=True)
    p.add_argument("--output", default="experiments/latest_kaggle_eval_benchmark.json")
    args = p.parse_args()

    challenges = json.loads(Path(args.challenges).read_text())
    solutions = json.loads(Path(args.solutions).read_text())
    qwen = set(json.loads(Path(args.qwen_oracle_rows).read_text()))

    total = 0
    symbolic_hits = set()
    synthesis_hits = set()
    local_union_hits = set()
    rows_with_symbolic_candidates = 0
    rows_with_synthesis_candidates = 0
    candidate_counts = {}

    for task_id, task in challenges.items():
        symbolic = candidate_grids(task)
        synthesis = synthesis_candidate_grids(task, max_depth=2)

        truths = solutions[task_id]
        if len(truths) != len(task["test"]):
            raise ValueError(f"{task_id}: solution/test row mismatch")

        for i, truth in enumerate(truths):
            total += 1
            key = f"{task_id}:{i}"
            sp = symbolic[i]
            sy = synthesis[i]
            merged = _dedupe(sp + sy)

            if sp:
                rows_with_symbolic_candidates += 1
            if sy:
                rows_with_synthesis_candidates += 1

            candidate_counts[key] = {
                "symbolic": len(sp),
                "synthesis": len(sy),
                "local_union": len(merged),
            }

            if any(g == truth for g in sp):
                symbolic_hits.add(key)
            if any(g == truth for g in sy):
                synthesis_hits.add(key)
            if any(g == truth for g in merged):
                local_union_hits.add(key)

    full_oracle = qwen | local_union_hits
    report = {
        "tasks": len(challenges),
        "rows": total,
        "qwen_oracle_rows": len(qwen),
        "qwen_oracle_accuracy": len(qwen) / total,
        "rows_with_symbolic_candidates": rows_with_symbolic_candidates,
        "rows_with_synthesis_candidates": rows_with_synthesis_candidates,
        "symbolic_rows": len(symbolic_hits),
        "synthesis_rows": len(synthesis_hits),
        "local_union_rows": len(local_union_hits),
        "unique_symbolic_over_qwen": sorted(symbolic_hits - qwen),
        "unique_synthesis_over_qwen": sorted(synthesis_hits - qwen),
        "unique_local_union_over_qwen": sorted(local_union_hits - qwen),
        "qwen_plus_local_oracle_rows": len(full_oracle),
        "qwen_plus_local_oracle_accuracy": len(full_oracle) / total,
        "candidate_counts": candidate_counts,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "candidate_counts"}, indent=2))


if __name__ == "__main__":
    main()
