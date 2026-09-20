# Experiment 003 — First genuine candidate-expansion gain

## Evaluation

Exact Kaggle public-evaluation snapshot:

- 120 tasks
- 172 output rows
- trusted Qwen candidate-pool oracle: **57 / 172 = 33.14%**

The projection families were evaluated by execution-fitting each rule against every
training pair, then applying only surviving rules to test inputs.

## Results

| Family | Exact rows | Unique rows beyond Qwen oracle |
|---|---:|---|
| Divider-period projection | 0 | 0 |
| Encoded block tiling | 0 | 0 |
| Cell-grid template completion | 1 | **1** |
| Frame/key relational walk | 2 | **2** |

Unique rows:

- `3e6067c3:0`
- `3e6067c3:1`
- `b99e7126:0`

Qwen + new execution-verified projection experts:

- **60 / 172 oracle rows**
- **34.88% row-level oracle coverage**
- gain over Qwen-only oracle: **+3 rows / +1.74 percentage points**

## Interpretation

This is the first ARCForge candidate-generation improvement that adds exact answers
absent from the Qwen candidate pool.

The useful signal is structural:

- frame/key graph walking is genuinely complementary;
- cell-grid template completion is genuinely complementary;
- generic divider-period and encoded-block-tiling rules are not yet general enough
  to survive the full demonstration-fit gate.

## Decision

Keep and expand:

1. frame/key graph reasoning;
2. cell-grid / repeated-cell reasoning.

Pause the current divider-period and block-tiling implementations until a broader
induction mechanism can infer their structure without task-specific assumptions.

Next target: raise oracle union from 60 to at least 69 rows (40.12%) and preferably
above 77 rows (~45%) before spending serious effort on the final label-free selector.
