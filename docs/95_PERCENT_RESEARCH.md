# Research note — What the 76% Kaggle lead and 95% public-eval systems imply

Date: 2026-09-20

## Rabbithole (Kaggle public leaderboard)

Current public leaderboard evidence shows Rabbithole at ~76.9%. No public method
write-up, notebook, or repository for the team's actual competition solution was
found during this research pass, so ARCForge should not pretend to know their
exact recipe.

The useful signal is therefore architectural, not a copied implementation:
the gap between the public ~30-35% Qwen/NVARC family and the 70%+ leaderboard
shows that a much richer per-task search/verification stack is feasible inside
the competition setting.

## Public 95%+ systems

### Confluence Labs — 97.92% public eval

Public configuration:
- Gemini 3.1 Pro
- 12 agents per test input
- up to 10 refinement iterations per agent
- parallel isolated sandboxes
- each agent writes executable transforms
- all candidate grids are pooled
- final pass@2 uses frequency / majority voting

Core lesson: generate many independently reasoned executable programs, verify
them, then aggregate the resulting grids.

### Athanor — 95.7% public eval

Core mechanisms:
- code as verification during reasoning;
- explicit transform hypothesis separated from final code;
- execute candidate code on all training pairs;
- independent reviewer can APPROVE, REJECT, or request a second candidate;
- state is compressed into portable artifacts between contexts.

Core lesson: train-perfect is necessary but not sufficient. A separate
generalization gate is valuable because multiple programs can interpolate the
few demonstrations.

Important caveat: the published 95.7% is the best-run upper envelope across
development traces, not a single deterministic uniform run.

### Squeeze-Evolve — 97.5% public eval

Core lesson:
- preserve candidate diversity;
- mutate/recombine hypotheses across generations;
- allocate stronger reasoning only where marginal value is high.

### Darwinian Evolver — 95.1% public eval

Core lesson:
- maintain a population of candidate programs;
- score fitness on demonstrations;
- sample parents using fitness + novelty;
- mutate programs using an LLM;
- verify after mutation.

## What ARCForge must become

The current hand-authored experts are useful as cheap deterministic specialists,
but they will not plausibly scale from ~34% candidate coverage to 95%.

The next architecture should add a task-specific program search engine:

1. Generate N candidate Python transform programs for the current task.
2. Execute every candidate on every training pair.
3. Retain exact-fit programs only.
4. Produce failure diagnostics for near-miss programs.
5. Mutate/repair promising programs for several generations.
6. Add an independent generalization score/reviewer.
7. Pool outputs from Qwen/NVARC, hand-authored experts, generated exact-fit programs, and mutated/repaired programs.
8. Select two outputs using support, diversity, verification evidence and source independence.

## Kaggle constraint

Public 95% systems above often use hosted frontier-model APIs and external
sandbox infrastructure. Those implementations cannot simply be dropped into
the final offline Kaggle notebook.

ARCForge therefore needs an offline distillation of the architecture:
- attached open-weight local model(s);
- local Python execution;
- bounded candidate budget;
- aggressive caching;
- task routing so hard search is spent only where cheap experts/Qwen are uncertain.

## Next engineering milestone

Build program_search/ as an offline Confluence/Athanor-inspired layer:

- structured transform-program format;
- exact train verifier;
- mutation operators;
- program deduplication by behavior;
- candidate provenance;
- majority / support aggregation;
- oracle-union benchmark against Qwen.

Target before selector work: push combined candidate oracle comfortably above 45%
for the near-term Kaggle goal, while designing the search engine so it can scale
far beyond that threshold.
