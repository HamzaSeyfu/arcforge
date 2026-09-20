# ARCForge

Research workspace for **ARC Prize 2026 — ARC-AGI-2**.

## Goal

Build a reproducible multi-solver system that improves ARC-AGI-2 exact-match `pass@2`.

Milestones:

- `v0`: Kaggle submission pipeline validated — **0.00 public score**
- `v1`: reproduce a strong public Qwen/NVARC baseline — target **~30–34%**
- `v2`: exceed the public baseline with complementary solvers / selection — target **40%+**
- long-term research target: **95%**

## Research principles

1. Reproduce before modifying.
2. Keep public evaluation and hidden-set claims separate.
3. Measure **candidate coverage** and **selection quality** independently.
4. A new expert must add genuinely complementary exact solves.
5. Do not promote an experiment from a small subset unless it survives the full public evaluation.
6. Track runtime because the Kaggle GPU submission budget is finite.

## Planned solver families

- Neural baseline: ARC-specialized Qwen/NVARC-style test-time adaptation.
- Symbolic / object-centric execution-verified solver.
- Program synthesis.
- Candidate refinement / repair.
- Portfolio routing and `pass@2` pair selection.
- Optional compute-aware routing.

## First useful command

Compare two public-evaluation submission files and quantify complementarity:

```bash
python scripts/analyze_complementarity.py \
  --solutions /path/to/arc-agi_evaluation_solutions.json \
  --a submission_a.json \
  --b submission_b.json
```

This reports each system's exact `pass@2`, oracle union, and rows solved uniquely by each system.
