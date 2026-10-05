# ARCForge architecture

ARCForge separates **solver generation** from **evaluation and portfolio analysis**.

That boundary is deliberate: solver experiments change quickly, while exact scoring, validation, and complementarity metrics should remain stable and testable.

## 1. Stable evaluation layer

### `arcforge.validation`

Validates:

- task IDs;
- number of test outputs;
- `attempt_1` / `attempt_2` schema;
- rectangular ARC grids;
- 1–30 row / column bounds;
- integer color values in `[0, 9]`.

### `arcforge.evaluation`

Provides exact-match scoring.

For each test row, the prediction is solved when either attempt exactly equals the reference grid.

Two views are reported:

- **row exact pass@2**: every test output row has equal weight;
- **task-weight exact pass@2**: each ARC task has equal total weight even when tasks contain different numbers of test rows.

The module also exposes solved-row keys such as:

```text
task_id:0
task_id:1
```

These keys are the common currency for portfolio analysis.

## 2. Portfolio layer

### Pairwise complementarity

For two solvers A and B, ARCForge measures:

- solves by A;
- solves by B;
- shared solves;
- solves unique to A;
- solves unique to B;
- oracle union.

The oracle union is diagnostic. It answers:

> If a perfect selector could choose between these solvers, how much exact coverage is available?

It is **not** a deployable score by itself.

### N-way coverage

`portfolio_coverage()` extends the same idea to any number of solvers.

It returns:

- exact solves per solver;
- full oracle union;
- greedy marginal coverage order.

### Greedy marginal order

The greedy order repeatedly chooses the remaining solver that adds the most previously unsolved rows.

Example:

```text
1. anchor       +42   cumulative=42
2. symbolic      +7   cumulative=49
3. program       +3   cumulative=52
4. repair        +0   cumulative=52
```

This is useful for engineering decisions:

- a branch with +0 unique solves is easy to reject;
- a lower-scoring branch can still be valuable if its marginal gain is high;
- expensive solvers can be evaluated against the exact coverage they add.

## 3. Solver layer

The repository intentionally does not hard-wire a single ARC solver.

Candidate solver families include:

- ARC-specialized neural models;
- symbolic/object-centric programs;
- program synthesis;
- candidate repair;
- test-time adaptation;
- retrieval-assisted program generation.

Every solver should ultimately emit the same submission schema so the stable evaluation layer can compare them.

## 4. Experiment lifecycle

```text
hypothesis
   ↓
small exact gate
   ↓
exact solves?
 ┌───────┴────────┐
 no              yes
 ↓                ↓
reject        full public eval
                  ↓
             unique solves?
             ┌────┴────┐
             no       yes
             ↓         ↓
          archive   portfolio candidate
```

The small gate is a stop-loss mechanism, not a substitute for full evaluation.

## 5. Design rule

ARCForge does not optimize for activity.

It optimizes for **auditable evidence**:

- exact outputs;
- reproducible metrics;
- marginal solver value;
- compute-aware decisions;
- honest experiment logs, including failures.
