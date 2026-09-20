#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arcforge.program_search.aggregate import candidate_pools_from_exact
from arcforge.program_search.engine import ProgramSearchEngine
from arcforge.program_search.generator import TextModelProgramGenerator
from arcforge.program_search.local_qwen import LocalQwenTextRuntime
from arcforge.program_search.reviewer import structural_generalization_score
from arcforge.program_search.types import SearchConfig


QWEN_ORACLE_ROWS = {
    "135a2760:0","142ca369:0","1818057f:0","1ae2feb7:2","20270e3b:0",
    "2b83f449:0","2ba387bc:0","2d0172a1:0","2d0172a1:1","36a08778:0",
    "38007db0:0","38007db0:1","3a25b0d8:0","3a25b0d8:1","45a5af55:0",
    "4a21e3da:1","4c3d4a41:0","4c3d4a41:1","4c7dc4dd:0","53fb4810:0",
    "58f5dbd5:0","65b59efc:0","6e453dd6:0","71e489b6:0","71e489b6:1",
    "7666fa5d:0","78332cb0:0","78332cb0:1","7b0280bc:0","7b80bb43:0",
    "7c66cb00:0","7ed72f31:0","7ed72f31:1","800d221b:0","8b9c3697:0",
    "8e5c0c38:1","9385bd28:1","981571dc:0","9aaea919:0","a251c730:0",
    "a6f40cea:0","aa4ec2a5:0","b5ca7ac4:0","b6f77b65:0","b6f77b65:1",
    "b6f77b65:2","bf45cf4b:0","d35bdbdc:1","d35bdbdc:2","d8e07eb2:0",
    "d8e07eb2:1","db0c5428:0","de809cff:0","dfadab01:0","e8686506:0",
    "eee78d87:0","fc7cae8d:0"
}
ARCFORGE_UNIQUE_ROWS = {"8e5c0c38:0", "b99e7126:0"}
BASELINE_POOL_ROWS = QWEN_ORACLE_ROWS | ARCFORGE_UNIQUE_ROWS


def _task_uncovered(task_id: str, solutions: dict) -> bool:
    return all(
        f"{task_id}:{i}" not in BASELINE_POOL_ROWS
        for i in range(len(solutions[task_id]))
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenges", required=True)
    p.add_argument("--solutions", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--initial", type=int, default=2)
    p.add_argument("--mutations", type=int, default=2)
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=900)
    p.add_argument("--output", default="arcforge_v2_killtest_report.json")
    args = p.parse_args()

    challenges = json.loads(Path(args.challenges).read_text())
    solutions = json.loads(Path(args.solutions).read_text())

    task_ids = [
        tid for tid in challenges
        if _task_uncovered(tid, solutions)
    ][:args.limit]

    print("ARCForge v2 — 51% agentic kill test")
    print("Hard-slice tasks:", task_ids)
    print("Previous audited combined oracle: 59/172 = 34.30%")
    print()

    runtime = LocalQwenTextRuntime(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=0.85,
        top_p=0.92,
    )
    generator = TextModelProgramGenerator(runtime, family="qwen3-coder")
    engine = ProgramSearchEngine(
        generator,
        config=SearchConfig(
            initial_candidates=args.initial,
            mutations_per_generation=args.mutations,
            generations=args.generations,
            parent_pool_size=3,
            max_exact_programs=12,
            max_programs_total=32,
            timeout_seconds=1.5,
        ),
        reviewer=structural_generalization_score,
    )

    total_rows = 0
    gen0_oracle = 0
    final_oracle = 0
    final_selected = 0
    refinement_rescued_rows = 0
    refinement_rescued_tasks = 0
    unique_rows: set[str] = set()
    task_report = {}

    started = time.time()

    for task_number, task_id in enumerate(task_ids, start=1):
        task_started = time.time()
        task = challenges[task_id]
        truths = solutions[task_id]

        result = engine.search(
            task,
            seed=20260920 + task_number * 10007,
        )

        gen0_programs = [
            program for program in result.exact_programs
            if program.generation == 0
        ]
        gen0_pools = candidate_pools_from_exact(gen0_programs)
        final_pools = result.candidate_pools

        task_gen0 = 0
        task_final = 0
        task_selected = 0
        task_rescues = 0

        for test_index, truth in enumerate(truths):
            total_rows += 1
            row_id = f"{task_id}:{test_index}"

            pool0 = (
                gen0_pools[test_index]
                if test_index < len(gen0_pools)
                else []
            )
            poolf = (
                final_pools[test_index]
                if test_index < len(final_pools)
                else []
            )

            hit0 = any(candidate == truth for candidate in pool0)
            hitf = any(candidate == truth for candidate in poolf)
            hits = any(candidate == truth for candidate in poolf[:2])

            if hit0:
                gen0_oracle += 1
                task_gen0 += 1
            if hitf:
                final_oracle += 1
                task_final += 1
                if row_id not in BASELINE_POOL_ROWS:
                    unique_rows.add(row_id)
            if hits:
                final_selected += 1
                task_selected += 1
            if hitf and not hit0:
                refinement_rescued_rows += 1
                task_rescues += 1

        if task_rescues:
            refinement_rescued_tasks += 1

        first_exact_generation = min(
            (program.generation for program in result.exact_programs),
            default=None,
        )

        task_report[task_id] = {
            "rows": len(truths),
            "generation0_oracle_hits": task_gen0,
            "final_oracle_hits": task_final,
            "selected_hits": task_selected,
            "refinement_rescued_rows": task_rescues,
            "programs_evaluated": result.diagnostics["programs_evaluated"],
            "exact_programs": len(result.exact_programs),
            "first_exact_generation": first_exact_generation,
            "best_soft_score": result.diagnostics["best_soft_score"],
            "distinct_test_candidates": result.diagnostics["distinct_test_candidates"],
            "elapsed_seconds": time.time() - task_started,
        }

        print(
            f"[{task_number}/{len(task_ids)}] {task_id} "
            f"best={result.diagnostics['best_soft_score']:.3f} "
            f"exact={len(result.exact_programs)} "
            f"first_exact_gen={first_exact_generation} "
            f"gen0={task_gen0}/{len(truths)} "
            f"final={task_final}/{len(truths)} "
            f"selected={task_selected}/{len(truths)} "
            f"elapsed={time.time()-task_started:.1f}s"
        )

    elapsed = time.time() - started
    summary = {
        "tasks": len(task_ids),
        "task_ids": task_ids,
        "outputs": total_rows,
        "generation0_oracle_hits": gen0_oracle,
        "after_refinement_oracle_hits": final_oracle,
        "after_refinement_selected_hits": final_selected,
        "refinement_rescued_rows": refinement_rescued_rows,
        "refinement_rescued_tasks": refinement_rescued_tasks,
        "unique_beyond_previous_59_pool": sorted(unique_rows),
        "elapsed_seconds": elapsed,
        "config": {
            "initial_candidates": args.initial,
            "mutations_per_generation": args.mutations,
            "generations": args.generations,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    report = {"summary": summary, "tasks": task_report}
    Path(args.output).write_text(json.dumps(report, indent=2))

    print()
    print("=" * 76)
    print("ARCForge v2 — 51% AGENTIC KILL TEST RESULT")
    print("=" * 76)
    print(f"Tasks:                          {len(task_ids)}")
    print(f"Outputs:                        {total_rows}")
    print(
        f"Generation-0 oracle:            "
        f"{gen0_oracle}/{total_rows} = "
        f"{100*gen0_oracle/total_rows if total_rows else 0:.2f}%"
    )
    print(
        f"After-refinement oracle:        "
        f"{final_oracle}/{total_rows} = "
        f"{100*final_oracle/total_rows if total_rows else 0:.2f}%"
    )
    print(
        f"After-refinement selected p@2:  "
        f"{final_selected}/{total_rows} = "
        f"{100*final_selected/total_rows if total_rows else 0:.2f}%"
    )
    print(f"Rows rescued by refinement:     {refinement_rescued_rows}")
    print(f"Tasks rescued by refinement:    {refinement_rescued_tasks}")
    print(f"Unique beyond old 59-row pool:  {len(unique_rows)}")
    print(f"Unique row IDs:                  {sorted(unique_rows)}")
    print(f"Elapsed:                         {elapsed/60:.1f} minutes")
    print("=" * 76)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
