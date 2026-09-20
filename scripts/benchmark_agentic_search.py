#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arcforge.program_search import ProgramSearchEngine, SearchConfig, TextModelProgramGenerator
from arcforge.program_search.local_qwen import LocalQwenTextRuntime
from arcforge.program_search.vllm_runtime import VLLMTextRuntime
from arcforge.program_search.reviewer import structural_generalization_score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenges", required=True)
    p.add_argument("--solutions", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--initial", type=int, default=4)
    p.add_argument("--mutations", type=int, default=4)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--runtime", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--tensor-parallel-size", type=int, default=4)
    p.add_argument("--output", default="agentic_search_report.json")
    args = p.parse_args()

    challenges = json.loads(Path(args.challenges).read_text())
    solutions = json.loads(Path(args.solutions).read_text())
    task_ids = list(challenges)[args.offset : args.offset + args.limit]

    if args.runtime == "vllm":
        vllm_runtime = VLLMTextRuntime(
            args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=8192,
            gpu_memory_utilization=0.90,
            max_num_seqs=max(4, args.initial, args.mutations),
        )

        def generate_text(prompts, seed):
            return vllm_runtime.generate(
                prompts,
                seed=seed,
                max_tokens=args.max_new_tokens,
                temperature=0.8,
                top_p=0.9,
                top_k=30,
            )

        family = "qwen-coder-vllm"
    else:
        hf_runtime = LocalQwenTextRuntime(
            args.model_path,
            max_new_tokens=args.max_new_tokens,
        )

        def generate_text(prompts, seed):
            return hf_runtime(prompts, seed)

        family = "qwen-coder-transformers"

    generator = TextModelProgramGenerator(generate_text, family=family)
    engine = ProgramSearchEngine(
        generator,
        config=SearchConfig(
            initial_candidates=args.initial,
            mutations_per_generation=args.mutations,
            generations=args.generations,
            parent_pool_size=4,
            max_exact_programs=12,
            max_programs_total=64,
            timeout_seconds=1.5,
        ),
        reviewer=structural_generalization_score,
    )

    rows = 0
    oracle_hits = 0
    selected_hits = 0
    task_reports = {}

    started = time.time()
    for index, task_id in enumerate(task_ids, start=1):
        task = challenges[task_id]
        result = engine.search(task, seed=12345 + index * 997)
        truths = solutions[task_id]

        task_oracle = 0
        task_selected = 0
        for test_index, truth in enumerate(truths):
            rows += 1
            pool = (
                result.candidate_pools[test_index]
                if test_index < len(result.candidate_pools)
                else []
            )
            if any(candidate == truth for candidate in pool):
                oracle_hits += 1
                task_oracle += 1
            if any(candidate == truth for candidate in pool[:2]):
                selected_hits += 1
                task_selected += 1

        task_reports[task_id] = {
            "rows": len(truths),
            "oracle_hits": task_oracle,
            "selected_hits": task_selected,
            "exact_programs": len(result.exact_programs),
            **result.diagnostics,
        }
        print(
            f"[{index}/{len(task_ids)}] {task_id} "
            f"oracle={task_oracle}/{len(truths)} "
            f"exact_programs={len(result.exact_programs)} "
            f"elapsed={time.time()-started:.1f}s"
        )

    report = {
        "tasks": len(task_ids),
        "rows": rows,
        "oracle_hits": oracle_hits,
        "oracle_accuracy": oracle_hits / rows if rows else 0.0,
        "selected_hits": selected_hits,
        "selected_accuracy": selected_hits / rows if rows else 0.0,
        "elapsed_seconds": time.time() - started,
        "config": {
            "initial": args.initial,
            "mutations": args.mutations,
            "generations": args.generations,
            "max_new_tokens": args.max_new_tokens,
            "runtime": args.runtime,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "task_reports": task_reports,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "task_reports"}, indent=2))


if __name__ == "__main__":
    main()
