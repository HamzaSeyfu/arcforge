# Experiment 001 — Qwen/NVARC public anchor

## Status

**Accepted as the current historical ARCForge public anchor.**

## Result

- Kaggle public leaderboard: **28.47**
- Version: ARCForge v7.1.1
- Solver family: Qwen/NVARC-derived ARC-specialized neural baseline
- Metric: competition public leaderboard score
- Date range: September 2026

## Evidence boundary

This score was observed on Kaggle during the v7.1.1 run.

The exact notebook artifact and full runtime provenance are not yet committed into this repository, so ARCForge treats **28.47 as a verified historical run, not a result reproduced from the current GitHub checkout**.

That distinction is intentional.

## Role in the portfolio

This branch is the anchor against which complementary solvers should be judged.

A new solver does not need to beat 28.47 by itself to be useful. It needs to add exact solves that the anchor misses.

## Promotion rule

Keep this branch as a trusted comparison point until a newer baseline is:

1. reproducible,
2. measured with exact public evaluation,
3. documented with model/runtime provenance,
4. demonstrably stronger or more useful in portfolio union.
