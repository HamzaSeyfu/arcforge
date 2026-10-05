# Experiments

Every promoted experiment should record:

- date / commit
- solver family
- model/checkpoint provenance
- public-evaluation split
- exact `pass@2`
- candidate-pool oracle coverage
- unique solves vs the current trusted baseline
- wall time
- accelerator
- timeouts / OOMs
- whether any labels were used for model or selector development

Do not promote a result just because a notebook title advertises a high score.

## Logged experiments

- [000 — v0 pipeline smoke test](000_v0_zero_baseline.md)
- [001 — Qwen/NVARC public anchor](001_nvarc_public_anchor.md)
- [002 — local Qwen3-Coder program-synthesis gate](002_qwen3_coder_program_synthesis_gate.md)
- [003 — training-only retrieval / solution-atlas branch](003_training_atlas_retrieval.md)
