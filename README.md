# ARCForge

**Research and engineering workspace for ARC Prize 2026 / ARC-AGI-2.**

ARCForge explores how complementary reasoning systems can be combined into a reproducible multi-solver pipeline for abstract reasoning tasks.

The project focuses not only on raw score, but also on **candidate coverage, solver complementarity, selection quality, reproducibility, and compute constraints**.

## Objective

Build a modular system that can:

- reproduce strong public ARC-AGI-2 baselines;
- compare solver families on exact-match performance;
- measure complementary solves between systems;
- combine neural, symbolic, and programmatic approaches;
- improve `pass@2` through better candidate generation and selection;
- track runtime and evaluation methodology rigorously.

## Current milestones

- **v0**: Kaggle submission pipeline validated.
- **v1**: reproduce a strong public Qwen/NVARC-style baseline.
- **v2**: add complementary solver families and selection logic.
- **Long-term research direction**: investigate high-coverage portfolio reasoning.

Public evaluation results and hidden-set claims are kept strictly separate.

## Research principles

1. Reproduce before modifying.
2. Separate benchmark evidence from assumptions.
3. Measure candidate coverage and selection quality independently.
4. Add a new solver only when it contributes complementary exact solves.
5. Do not promote small-subset results until they survive full public evaluation.
6. Track compute cost because Kaggle GPU budgets are finite.

## Solver families under study

- ARC-specialized neural baselines
- Object-centric / symbolic execution
- Program synthesis
- Candidate repair and refinement
- Portfolio routing
- `pass@2` pair selection
- Compute-aware routing

## Example analysis

Compare two public-evaluation submission files and quantify complementarity:

```bash
python scripts/analyze_complementarity.py \
  --solutions /path/to/arc-agi_evaluation_solutions.json \
  --a submission_a.json \
  --b submission_b.json
```

The analysis reports:

- exact `pass@2` for each system;
- oracle union coverage;
- tasks solved uniquely by each solver;
- evidence of whether combining two systems is actually useful.

## Repository structure

- `src/` — reusable ARCForge components
- `scripts/` — analysis and experiment utilities
- `experiments/` — reproducible experiment definitions/results
- `kaggle/` — Kaggle-specific integration work
- `docs/` — technical notes and research documentation
- `tests/` — validation of core utilities
- `.github/` — repository automation

## Stack

Python · ARC-AGI-2 · experiment automation · evaluation tooling · Kaggle · solver portfolio analysis

## What this project demonstrates

- Research-oriented software engineering
- Reproducible experimentation
- Benchmark evaluation discipline
- Python tooling for model/solver analysis
- Engineering trade-offs under compute constraints
- Iterative development from baseline reproduction toward novel combinations

---

**Status:** active research project for ARC Prize 2026.
