# ARCForge

**Engineering workspace for ARC Prize 2026 / ARC-AGI-2.**

ARCForge is a research-oriented Python project for building, measuring, and combining complementary ARC solvers under exact-match and compute constraints.

The project is intentionally evidence-driven: a solver is useful only when it adds **new exact solves**, improves a measurable selection bottleneck, or reduces compute for the same coverage.

![tests](https://github.com/HamzaSeyfu/arcforge/actions/workflows/tests.yml/badge.svg)

## Current status

| Experiment | Result | Decision |
| --- | ---: | --- |
| v0 transform-library smoke test | 0.00% local | Pipeline only, rejected as solver |
| v7.1.1 Qwen/NVARC-derived anchor | **28.47 public LB** | Strongest verified ARCForge public result so far |
| v12.2 local Qwen3-Coder program synthesis gate | 0 exact agents on first 2 gate tasks | Stopped early, raw branch rejected |

The 28.47 result is a historical Kaggle run and is documented in the experiment log; it is **not** presented as a result reproduced by the current GitHub checkout.

**51% remains a research target, not an achieved score.**

## Why a solver portfolio?

ARC systems often fail on different tasks. A second solver that scores lower overall can still be valuable if it solves tasks the anchor misses.

ARCForge therefore treats solver development as a coverage problem:

```text
solver outputs
     │
     ├── exact scoring
     ├── validation
     ├── pairwise complementarity
     ├── n-way oracle coverage
     └── greedy marginal coverage
              │
              ▼
        promote / reject branch
```

The important question is not only:

> “Did solver B score higher than solver A?”

It is also:

> “How many exact outputs did B add that A could not solve?”

## Core tooling

ARCForge currently includes:

- strict ARC grid and submission validation;
- exact `pass@2` scoring;
- row-weighted and task-weighted views;
- pairwise unique/shared solve analysis;
- **n-way solver portfolio coverage**;
- greedy solver ordering by marginal exact gain;
- reproducible CLI reports;
- pytest-based CI.

### Score one submission

```bash
python scripts/score_submission.py \
  --solutions /path/to/arc-agi_evaluation_solutions.json \
  --submission /path/to/submission.json
```

### Compare two solvers

```bash
python scripts/analyze_complementarity.py \
  --solutions /path/to/arc-agi_evaluation_solutions.json \
  --a submission_a.json \
  --b submission_b.json
```

### Analyze a full solver portfolio

```bash
python scripts/analyze_portfolio.py \
  --solutions /path/to/arc-agi_evaluation_solutions.json \
  --solver nvarc=/path/to/nvarc.json \
  --solver symbolic=/path/to/symbolic.json \
  --solver program=/path/to/program.json \
  --json-out portfolio_report.json
```

The report shows:

- exact `pass@2` per solver;
- greedy marginal exact coverage;
- pairwise oracle unions;
- full portfolio oracle coverage.

## Research principles

1. **Reproduce before modifying.**
2. **Exact match is the primary result.**
3. Measure candidate coverage and selection quality separately.
4. Promote a solver only when it contributes complementary exact solves.
5. Stop expensive branches early when they fail an exact-solve gate.
6. Record negative results instead of burying them.
7. Keep public-evaluation evidence separate from hidden-set claims.
8. Track compute because Kaggle GPU budgets are part of the engineering problem.

## Repository structure

```text
src/arcforge/
  evaluation.py      exact pass@2 scoring
  validation.py      ARC grid/submission validation
  portfolio.py       pairwise + n-way coverage analysis

scripts/
  score_submission.py
  analyze_complementarity.py
  analyze_portfolio.py

experiments/
  reproducible experiment log, including rejected branches

docs/
  architecture and research plan

kaggle/
  Kaggle execution notes

tests/
  unit tests run in GitHub Actions
```

## Experiment discipline

Every promoted experiment should record:

- date and commit;
- solver/model provenance;
- evaluation split;
- exact `pass@2`;
- unique exact solves vs the trusted anchor;
- oracle union;
- runtime and accelerator;
- timeouts / OOMs / harness failures;
- whether labels influenced development.

A notebook title is not evidence. A near-miss cell score is not an ARC solve. A branch that adds no exact solves is not promoted.

## What this project demonstrates

- research-oriented Python engineering;
- benchmark and evaluation discipline;
- experiment automation;
- solver portfolio analysis;
- failure analysis under GPU constraints;
- reproducible CLI tooling and CI;
- iterative system design from public baseline reproduction toward complementary reasoning systems.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [experiments/](experiments/) for the engineering record.

---

**Status:** active research project. Current priority is building a credible, measurable ARC solver portfolio rather than hiding failed branches behind optimistic notebook names.
