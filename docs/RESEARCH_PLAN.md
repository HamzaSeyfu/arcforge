# ARCForge research plan

## Current position — 2026-10-05

ARCForge has a working evaluation/validation core and a verified historical public Kaggle result of **28.47** from a Qwen/NVARC-derived branch.

Recent local program-synthesis experiments with Qwen3-Coder were stopped after an exact-solve gate produced no exact programs on the first two tasks. That branch is documented as a negative result instead of being presented as progress.

The immediate goal is therefore to strengthen the **portfolio engineering layer** while solver research continues.

## Phase 1 — Reproducible anchors

Maintain at least one trusted, reproducible neural anchor.

For every anchor record:

- model/checkpoint provenance;
- Kaggle inputs;
- exact public-evaluation score;
- runtime;
- accelerator;
- notebook/code version.

A historical leaderboard score and a locally reproducible result must be labeled separately.

## Phase 2 — Portfolio diagnostics

Status: **implemented in the repository.**

Measure:

- exact `pass@2` per solver;
- shared vs unique solves;
- pairwise oracle union;
- full n-way oracle union;
- greedy marginal solver coverage.

Primary question:

```text
How many new exact rows does this solver add?
```

If the answer is zero, the branch does not earn promotion merely because its internal confidence or cell-level fitness improved.

## Phase 3 — Solver adapters

Add thin adapters that normalize outputs from experimental solver families into the same submission schema.

Target families:

### A. Neural ARC anchor
Strong specialized model used as the portfolio base.

### B. Symbolic / object-centric solver
Only execution-verified candidates that fit every training pair.

### C. Program synthesis
Generate executable transforms, verify exact fit on all train examples, then predict test.

### D. Test-time adaptation
Evaluate only when it adds exact solves that justify its cost.

## Phase 4 — Selection

Once oracle union materially exceeds the best individual solver, work on a label-free selector.

Useful signals may include:

- agreement between augmentations;
- candidate frequency;
- exact train-fit evidence;
- solver confidence;
- structural task descriptors;
- solver-specific historical reliability.

Selection work is justified only when measurable oracle headroom exists.

## Phase 5 — Compute-aware routing

Run expensive experts only when expected marginal coverage justifies them.

Record:

- wall time;
- GPU type/count;
- model load time;
- per-task timeout;
- OOM / crash rate;
- exact gain per compute unit.

## Kill gates

Pause or reject a branch when:

- it adds no unique exact solves;
- its oracle union headroom is negligible;
- repeated repair loops do not converge to exact train fit;
- runtime cost is disproportionate;
- results depend on harness artifacts or invalid evaluation leakage.

## Score target

The project may research paths toward **51%+**, but this number is a target, not a README claim.

A target is promoted to a result only after exact evaluation evidence exists.
