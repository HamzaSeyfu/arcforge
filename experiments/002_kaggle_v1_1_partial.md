# Experiment 002 — Kaggle offline portfolio v1.1 (partial)

## Environment

- Kaggle ARC Prize 2026 / ARC-AGI-2 competition data
- 120 evaluation tasks
- 172 evaluation outputs
- offline notebook, Internet disabled

## Result

One major expert failed in this run:

```
synthesis error: NameError: name '_copy' is not defined
```

Therefore this is a **partial benchmark**, not a final portfolio score.

Working experts produced:

- relational: 3 / 172 exact rows
- projection: 1 / 172 exact row
- symbolic: 0
- panels: 0
- objects: 0
- framewalk: 0
- ARCForge local union: 4 / 172 = 2.33%

Compared with the previously audited Qwen candidate-pool oracle:

- Qwen oracle: 57 / 172
- unique ARCForge rows beyond Qwen: 2
- Qwen + ARCForge oracle: **59 / 172 = 34.30%**
- unique rows: `8e5c0c38:0`, `b99e7126:0`

## Interpretation

The candidate-expansion hypothesis is validated at small scale: even the partial
non-neural portfolio generated two exact outputs absent from the Qwen pool.

The 2.33% standalone selector score is **not** an estimate of the combined Kaggle
system because Qwen candidates were not present in the notebook selector.

## Action

- fixed missing `_copy` import in synthesis;
- added a regression test;
- rerun the full 172-output evaluation before adding more task families.
