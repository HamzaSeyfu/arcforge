# Experiment 000 — v0 pipeline smoke test

## Result

- Local ARC-AGI-2 evaluation: **0.00%**
- Kaggle public leaderboard: **0.00**
- Status: successful infrastructure smoke test, rejected as a solver baseline.

## Solver

Small exact transform library: identity, rotations, flips, transpose / anti-transpose,
and global color remapping. Fallbacks were identity and 180-degree rotation.

## Conclusion

The submission pipeline is verified end-to-end. No further time should be spent
expanding this toy baseline. Move immediately to a competitive public neural
baseline and use public evaluation for iteration before consuming Kaggle submissions.
