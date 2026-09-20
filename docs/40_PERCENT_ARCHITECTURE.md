# ARCForge 40% architecture

The current trusted Qwen-family candidate pool has an oracle ceiling around 33% on
the public evaluation. Therefore ARCForge 40% cannot be a selector-only project.

## System shape

```text
                 ┌──────────────────────┐
                 │ Qwen / NVARC anchor │
                 └──────────┬───────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
symbolic expert      program synthesis     candidate repair
        │                   │                    │
        └───────────────────┼────────────────────┘
                            ▼
                    candidate union
                            │
                 train-fit verification
                            │
                            ▼
                     pair selector
                            │
                            ▼
                   attempt_1 / attempt_2
```

## Gate 1 — Candidate expansion

Do not optimize the selector first.

Goal: raise public-evaluation oracle candidate coverage from ~33% to at least 45%.
The extra margin is intentional because a label-free selector will not recover every
oracle solve.

## Gate 2 — Label-free recovery

Once oracle coverage is comfortably above 40%, optimize pair selection using features
that exist on hidden tasks:

- augmentation agreement;
- train-pair execution fit;
- candidate support count;
- source diversity;
- output-shape plausibility;
- color-set consistency;
- description/program complexity;
- neural likelihood / verifier scores;
- solver-family confidence.

## Gate 3 — Runtime

Every expert must report unique exact solves per added wall-clock minute.
Expensive experts should be routed only to uncertain tasks.

## What success looks like

A credible 40% candidate is not "one notebook scored 40 once." It is:

- reproducible;
- hidden-compatible;
- candidate oracle > 40%;
- selector gain survives held-out validation;
- full run fits Kaggle's submission limits.
