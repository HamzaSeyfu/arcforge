#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcforge.evaluation import score_submission
from arcforge.portfolio import complementarity


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare two ARC submissions on a labeled public-eval split."
    )
    p.add_argument("--solutions", required=True)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    args = p.parse_args()

    solutions = json.loads(Path(args.solutions).read_text())
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())

    sa = score_submission(a, solutions)
    sb = score_submission(b, solutions)
    comp = complementarity(a, b, solutions)

    total = sa.rows
    print("=== Solver A ===")
    print(f"Exact pass@2: {comp.solved_a}/{total} ({100*sa.row_accuracy:.2f}%)")
    print("=== Solver B ===")
    print(f"Exact pass@2: {comp.solved_b}/{total} ({100*sb.row_accuracy:.2f}%)")
    print("=== Complementarity ===")
    print(f"Shared solves: {comp.shared}")
    print(f"Unique to A:  {comp.unique_a}")
    print(f"Unique to B:  {comp.unique_b}")
    print(f"Oracle union: {comp.oracle_union}/{total} ({100*comp.oracle_union/total:.2f}%)")
    gap = comp.oracle_union - max(comp.solved_a, comp.solved_b)
    print(f"Raw oracle headroom over best individual solver: {gap} rows ({100*gap/total:.2f} pp)")


if __name__ == "__main__":
    main()
