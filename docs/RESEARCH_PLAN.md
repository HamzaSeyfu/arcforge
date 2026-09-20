# ARCForge research plan

## Phase 1 — Anchor

Reproduce a strong, current, public ARC-AGI-2 baseline under Kaggle's submission
constraints. Target band: ~30–34% public score.

Do not modify the baseline until public-evaluation scoring is reproducible,
candidate artifacts can be inspected, runtime is known, and model/data provenance
is recorded.

## Phase 2 — Measure the actual bottleneck

For each labeled public-evaluation output collect candidate pool size, whether the
ground truth exists anywhere in the pool, rank under each selector, selector
agreement, shape/color statistics, augmentation support, and model/verifier scores.

Key measurement:

```text
oracle candidate coverage - selected pass@2
```

A large gap means selection/routing is worth attacking. A small gap means candidate
generation needs improvement first.

## Phase 3 — Parallel expert branches

### A. Portfolio selection
Optimize the *pair* of attempts, not two independent top-1 scores.

### B. Symbolic / object-centric expert
Only promote execution-verified candidates that fit every training pair.

### C. Program synthesis
Search a compact DSL derived from recurring solution primitives.

### D. Candidate refinement
Repair near-miss neural candidates using demonstration failures.

## Phase 4 — Cost-aware routing

Run expensive experts only on tasks whose descriptors / uncertainty justify the
compute.

## Kill gates

Pause a branch if it adds no unique exact solves, its oracle union headroom is
negligible, a label-free selector cannot recover the gain, or runtime cost is
disproportionate.
