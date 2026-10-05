#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcforge.evaluation import score_submission
from arcforge.portfolio import pairwise_complementarity, portfolio_coverage


def parse_solver(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/submission.json")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("solver name cannot be empty")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze exact pass@2 coverage, pairwise complementarity, and "
            "greedy marginal coverage for an ARC solver portfolio."
        )
    )
    parser.add_argument("--solutions", required=True, type=Path)
    parser.add_argument(
        "--solver",
        action="append",
        required=True,
        type=parse_solver,
        metavar="NAME=PATH",
        help="Named submission. Repeat for multiple solvers.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for a machine-readable summary.",
    )
    args = parser.parse_args()

    solutions = json.loads(args.solutions.read_text())
    submissions = {
        name: json.loads(path.read_text())
        for name, path in args.solver
    }

    if len(submissions) != len(args.solver):
        raise SystemExit("solver names must be unique")

    coverage = portfolio_coverage(submissions, solutions)
    pairwise = pairwise_complementarity(submissions, solutions)

    print("=== Individual exact pass@2 ===")
    for name in sorted(submissions):
        score = score_submission(submissions[name], solutions)
        print(
            f"{name:>18}: {score.solved_rows}/{score.rows} "
            f"({100 * score.row_accuracy:.2f}%)"
        )

    print("\n=== Greedy marginal coverage ===")
    for index, step in enumerate(coverage.greedy_order, 1):
        print(
            f"{index:>2}. {step.solver:<18} "
            f"+{step.added:<4} cumulative={step.cumulative}/{coverage.total_rows}"
        )

    print("\n=== Pairwise oracle unions ===")
    for (a, b), report in sorted(pairwise.items()):
        print(
            f"{a} + {b}: union={report.oracle_union}/{coverage.total_rows} "
            f"unique=({report.unique_a},{report.unique_b}) shared={report.shared}"
        )

    print(
        "\n=== Full portfolio oracle ===\n"
        f"{coverage.oracle_union}/{coverage.total_rows} "
        f"({100 * coverage.oracle_union_accuracy:.2f}%)"
    )

    if args.json_out:
        payload = {
            "total_rows": coverage.total_rows,
            "per_solver": coverage.per_solver,
            "oracle_union": coverage.oracle_union,
            "oracle_union_accuracy": coverage.oracle_union_accuracy,
            "greedy_order": [
                {
                    "solver": step.solver,
                    "added": step.added,
                    "cumulative": step.cumulative,
                }
                for step in coverage.greedy_order
            ],
            "pairwise": {
                f"{a}|{b}": {
                    "solved_a": report.solved_a,
                    "solved_b": report.solved_b,
                    "shared": report.shared,
                    "unique_a": report.unique_a,
                    "unique_b": report.unique_b,
                    "oracle_union": report.oracle_union,
                }
                for (a, b), report in sorted(pairwise.items())
            },
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
