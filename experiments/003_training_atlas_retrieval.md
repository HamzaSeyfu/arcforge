# Experiment 003 — training-only retrieval / solution-atlas branch

## Status

**Prepared, but not validated as a scoring improvement.**

## Motivation

The raw local Qwen3-Coder branch generated executable Python but failed the first exact-solve gate.

The next hypothesis was that code generation was not the main bottleneck: the model lacked enough ARC-specific transformation knowledge.

## v13.0 — training-only dataset builder

The builder was designed to collect only public ARC training examples with:

- task JSON;
- verified Python solution;
- rule notes / concepts;
- provenance metadata;
- an explicit guard excluding evaluation IDs.

The intended output was a Kaggle-attached `arc_atlas_training_only.jsonl` dataset.

### First run

The notebook failed before data collection because the Kaggle session could not resolve the GitHub host:

```text
URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```

This was an environment/network issue, not a solver result.

## v13.1 — retrieval-augmented program synthesis

The solver notebook was prepared to:

1. compute structural features from the target train pairs;
2. retrieve similar solved tasks from the training-only atlas;
3. try retrieved verified programs directly against the target train pairs;
4. provide different retrieved examples to multiple Qwen3-Coder agents;
5. accept only programs with 100% exact train fit.

Because the v13.0 dataset build did not complete in the first Kaggle session, v13.1 has not yet produced a valid evaluation result.

## Decision

Keep the branch as work-in-progress research code.

Do not claim a score improvement until the training-only dataset is built and the exact evaluation gate is run successfully.
