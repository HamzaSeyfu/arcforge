# Experiment 002 — Candidate expansion sprint

Current branch: `research/40pct-candidate-expansion`.

Candidate experts under benchmark:

- execution-verified symbolic primitives;
- depth-2 program synthesis;
- relational rules;
- multi-panel composition;
- object-centric selection/rendering;
- indicator-to-projection rules.

The benchmark uses the exact 120-task / 172-output Kaggle public evaluation snapshot
and compares oracle-union coverage against the trusted Qwen candidate pool.

Promotion rule: no task IDs or public answers are used by solver logic. A rule must be
inferred from, or exactly validated against, the task's own training demonstrations.
