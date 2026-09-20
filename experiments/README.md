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
