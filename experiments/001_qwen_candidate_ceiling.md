# Experiment 001 — Qwen candidate-pool ceiling

## Public evidence

Trusted Qwen/NVARC-family public-evaluation result:

- selected exact pass@2: **49 / 172 rows**
- row-weighted score: **28.49%**
- task-weighted score: **29.31%**
- decoded rows: **160 / 172**

The same run's full candidate pool contains the correct answer for:

- **57 / 172 rows**
- oracle candidate coverage: **33.14%**

## Consequence

A perfect selector over the existing candidate pool can recover at most 8 additional
rows. That is not enough for a 40% target.

Approximate row-count targets:

- 33.14% oracle ceiling: 57 / 172
- 40% row accuracy: about 69 / 172

Therefore the 40% program needs at least **12 additional exact rows beyond the current
oracle pool**, and about **20 additional exact rows over the current selected baseline**.

## Research decision

The primary 40% bottleneck is now **candidate generation**, not selection.

Priority order:

1. add genuinely complementary candidate families;
2. measure oracle-union growth;
3. only then train/engineer a label-free selector to recover that growth;
4. preserve runtime budget.

## Candidate-expansion branches

- execution-verified symbolic/object-centric solver;
- compact program synthesis;
- neural candidate repair;
- independent model/expert family;
- metamorphic candidate generation where demonstration consistency is exact.

No branch is promoted unless it adds unique exact solves on the public evaluation and
has a plausible hidden-set mechanism.
