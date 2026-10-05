# Recent ARCForge work — September / October 2026

This note records the recent experiment lineage without presenting unvalidated branches as score improvements.

## v7.1.1 — known-good neural anchor

- Solver family: Qwen/NVARC-derived ARC baseline
- Kaggle public leaderboard: **28.47**
- Status: current verified historical ARCForge public result
- Role: anchor for later complementary branches

The corresponding run was treated as the baseline to beat rather than being continuously modified in-place.

## v8.2.2 — CPU symbolic / DSL branch

- Lightweight symbolic branch designed to be cheap and complementary
- Local self-tests passed
- Earlier symbolic exact-fit checks did not produce a useful public-eval gain
- Status: retained as a research branch, not promoted as a scoring improvement

## v10 — portfolio calibration notebook

A calibration notebook was prepared to compare:

- neural baseline coverage,
- symbolic candidates,
- safe vs aggressive fusion,
- oracle union.

The first run was blocked by a missing attached Qwen model input, so it was not used as evidence for a score claim.

## v11 — local exact program synthesis

ARCForge moved from approximate candidate fitness toward executable program synthesis.

Core loop:

1. a local model writes `transform(grid)`;
2. the candidate is AST-checked and executed in a subprocess;
3. every visible train pair must match exactly;
4. failed candidates receive verifier feedback;
5. only exact-train-fit programs may predict test outputs;
6. two distinct exact outputs are selected for `pass@2`.

The initial local Qwen 27B run produced valid code but no exact programs on the first observed tasks, so the branch was stopped instead of consuming the full budget.

## v12 — Qwen3-Coder program-synthesis branch

The program-synthesis harness was moved to **Qwen3-Coder-30B-A3B-Instruct** on 4×L4.

Several harness issues were fixed during this iteration:

- preserving imports and helper functions generated before `transform()`;
- increasing output budget to avoid truncating valid programs;
- running four independent agents;
- richer exact verifier diagnostics;
- exposing common Python/ARC helper functions;
- enforcing exact train fit as the only acceptance criterion.

Observed gate:

- task `3e6067c3`: **0/4 exact agents**
- task `271d71e2`: **0/4 exact agents**

The run was stopped early. This branch did not demonstrate a score improvement.

## v13 — training-only retrieval / solution-atlas experiment

The next direction was to give the code model ARC-specific solved examples instead of relying on raw prompting.

Two notebooks were prepared:

- **v13.0**: build a training-only ARC solution-atlas dataset with explicit evaluation-split exclusion;
- **v13.1**: retrieve structurally similar solved training tasks and provide their verified programs/notes to Qwen3-Coder before synthesis.

The first builder run hit Kaggle DNS/network restrictions because the session had no outbound Internet access. The retrieval solver therefore has **not yet been validated as a scoring result**.

## Current conclusion

Recent work improved the experimental harness and clarified several failure modes, but it has **not yet improved the verified 28.47 public result**.

That is the current state of the project.
