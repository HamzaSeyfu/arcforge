# Experiment 002 — local Qwen3-Coder program-synthesis gate

## Status

**Rejected as a raw solver branch after an early exact-solve stop-loss.**

## Hypothesis

A local code-specialized LLM might solve ARC tasks by:

1. inferring a transformation from visible train pairs;
2. writing `transform(grid)`;
3. executing the program on every train example;
4. repairing verifier failures;
5. allowing only 100% exact-train-fit programs to predict test.

This mirrors a program-synthesis workflow rather than direct grid decoding.

## Setup

- Model: Qwen3-Coder-30B-A3B-Instruct
- Hardware: 4 × NVIDIA L4
- Serving: local vLLM
- Agents: 4 independent generations per task
- Repair budget: up to 2 attempts per agent
- Primary gate: exact train verification
- Intended panel: 20 ARC-AGI-2 public evaluation tasks
- Stop-loss: terminate early if the branch produces no exact evidence

## Observed gate

### Task `3e6067c3`

- exact agents: **0 / 4**
- distinct exact test candidates: **0**
- task pass@2: **0%**

Observed failure modes included:

- semantically wrong transformations affecting large portions of the grid;
- repeated nearly identical failure signatures across agents;
- one AST policy rejection before the harness was relaxed.

### Task `271d71e2`

- exact agents: **0 / 4**
- distinct exact test candidates: **0**
- task pass@2: **0%**

Observed failure modes included:

- large exact-grid mismatches;
- `KeyError`;
- invalid return types;
- arithmetic type errors during repair.

## Decision

The run was stopped before consuming the full 20-task / 90-minute budget.

The evidence did **not** justify continuing raw Qwen3-Coder prompting as an ARC solver.

This is not a claim that Qwen3-Coder cannot solve ARC. It is a narrower engineering conclusion:

> Under this local program-synthesis harness and budget, the branch failed the exact-solve promotion gate.

## Lessons carried forward

- Code-generation quality is not the same as ARC abstraction quality.
- Multiple random seeds do not guarantee genuinely different hypotheses.
- Exact verifier feedback is necessary but not sufficient for convergence.
- Harness restrictions must be tested so they do not reject harmless Python operations.
- GPU-heavy repair loops require early stop-loss criteria.
- The next useful program-synthesis iteration should add ARC-specific knowledge or retrieval, not merely more prompting.
