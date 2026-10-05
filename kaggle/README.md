# Kaggle integration

ARCForge keeps research code in GitHub and uses Kaggle as the official execution
and scoring environment.

Final competition notebooks must:

- run within the competition runtime limit,
- have Internet disabled,
- use public attached models/data only,
- write `/kaggle/working/submission.json`.

## Recent notebook lineage

### v7.1.1 — known-good baseline

Qwen/NVARC-derived notebook used for the current historical ARCForge public result:

- public leaderboard: **28.47**
- status: trusted anchor

### v11 — exact local program synthesis

Local model writes Python `transform(grid)` functions.

Candidates must:

- compile,
- execute safely,
- reproduce every train pair exactly,
- produce valid ARC grids.

Only exact-train-fit programs may predict test outputs.

### v12.2 — Qwen3-Coder harness

Moved the exact program-synthesis loop to Qwen3-Coder-30B-A3B-Instruct on 4×L4 and fixed code-extraction / truncation issues.

The first two gate tasks produced no exact agents, so the run was stopped early.

### v13.0 / v13.1 — training-only retrieval experiment

- v13.0 builds a training-only solved-program atlas.
- v13.1 retrieves structurally similar solved training tasks and feeds them to Qwen3-Coder.

The first v13.0 attempt was blocked by Kaggle DNS/network access and therefore did not produce a scoring result.

See `docs/RECENT_WORK.md` for the full recent timeline.
