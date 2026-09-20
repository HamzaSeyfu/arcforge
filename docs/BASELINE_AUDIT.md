# Baseline audit — Qwen3-4B / NVARC family

## Purpose

ARCForge needs a competitive anchor before custom solver work begins.

## Verified implementation pattern

Public ARC-AGI-2 code in the NVARC family uses:

- ARC-specialized Qwen3-4B checkpoint `qwen3_4b_grids15_sft139`;
- per-task test-time adaptation with LoRA;
- dihedral/color augmentations of the training examples;
- beam/candidate generation restricted to ARC tokens;
- augmentation-aware candidate rescoring;
- a candidate selector such as KGMoN;
- two final attempts per test output.

## Engineering observations

The public code path is GPU-heavy and depends on Kaggle-attached model/runtime assets.
ARCForge should therefore keep its reusable evaluation, selection and symbolic logic
separate from the Kaggle model runner.

## Reproduction gate

Before modifying the neural anchor, record:

1. full 120-task public-evaluation exact pass@2;
2. task-weighted score;
3. candidate-pool oracle coverage;
4. unique candidate count per row;
5. total wall time and per-task timeout rate;
6. exact model/checkpoint identity;
7. software/runtime versions.

## Source handling

Do not vendor third-party source files into ARCForge unless their license is explicit.
Until then, reproduce behavior through clean-room integration and public notebook inputs,
and keep attribution/provenance in experiment records.

## Next experiment

Run the Qwen3-4B anchor on the public evaluation set, export its candidate pools and
selection outputs, then use ARCForge's evaluator to quantify the selector gap.
