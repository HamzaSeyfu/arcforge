from __future__ import annotations

import ast
import re

from .types import ProgramArtifact


def structural_generalization_score(task: dict, artifact: ProgramArtifact) -> float:
    """
    Cheap Athanor-inspired gate for train-perfect programs.

    It cannot prove generalization. It only penalizes common ARC overfit smells:
    hard-coded coordinates, large literal tables, and source complexity.
    """
    if not artifact.exact_train:
        return 0.0

    source = artifact.source
    score = 1.0

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0.0

    numeric_literals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
    ]
    suspicious_coords = [x for x in numeric_literals if isinstance(x, int) and x > 9]

    if suspicious_coords:
        score -= min(0.35, 0.03 * len(suspicious_coords))

    list_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.List)]
    large_lists = sum(1 for n in list_nodes if len(n.elts) >= 8)
    score -= min(0.30, 0.08 * large_lists)

    lines = [line for line in source.splitlines() if line.strip()]
    if len(lines) > 80:
        score -= min(0.25, (len(lines) - 80) / 400)

    coordinate_patterns = len(
        re.findall(r"grid\s*\[\s*\d+\s*[,\]]", source)
    )
    score -= min(0.30, 0.05 * coordinate_patterns)

    return max(0.0, min(1.0, score))
