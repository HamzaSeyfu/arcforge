# Kaggle integration

ARCForge keeps research code in GitHub and uses Kaggle as the official execution
and scoring environment.

Final competition notebooks must:

- run within the competition runtime limit,
- have Internet disabled,
- use public attached models/data only,
- write `/kaggle/working/submission.json`.

Current v1 anchor candidate: reproduce a strong public Qwen3-4B ARC baseline
before adding ARCForge-specific research modules.
