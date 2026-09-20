from __future__ import annotations

import numpy as np

from .types import ProgramArtifact


def _grid_diff(pred, truth, max_cells: int = 225) -> str:
    if pred is None:
        return "<no valid grid>"
    pa = np.asarray(pred)
    ta = np.asarray(truth)
    if pa.shape != ta.shape:
        return f"shape mismatch predicted={pa.shape} expected={ta.shape}"
    if pa.size > max_cells:
        mismatches = np.argwhere(pa != ta)
        sample = mismatches[:20].tolist()
        return f"{len(mismatches)} mismatched cells; first positions={sample}"

    lines = []
    for prow, trow in zip(pa, ta):
        cells = []
        for p, t in zip(prow, trow):
            cells.append(str(int(p)) if p == t else f"{int(p)}/{int(t)}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def build_feedback(artifact: ProgramArtifact, task: dict) -> str:
    report = artifact.report
    if report is None:
        return "Program has not been evaluated."

    blocks = [
        f"Program score: {report.mean_soft_score:.4f}",
        f"Exact training examples: {report.exact_examples}/{report.total_examples}",
    ]

    for i, (pair, pred, soft, err) in enumerate(
        zip(
            task["train"],
            report.train_outputs,
            report.per_example_soft,
            report.errors,
            strict=True,
        ),
        start=1,
    ):
        if pred == pair["output"]:
            blocks.append(f"Example {i}: exact.")
            continue
        detail = _grid_diff(pred, pair["output"])
        if err:
            detail += f"\nExecution error: {err}"
        blocks.append(
            f"Example {i}: incorrect, cell score={soft:.4f}\n{detail}"
        )

    return "\n\n".join(blocks)
