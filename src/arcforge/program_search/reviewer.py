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


class TextModelGeneralizationReviewer:
    """
    Athanor-style artifact-only reviewer.

    The reviewer sees only the task, explicit hypothesis, code, train-perfect
    status, and proposed test outputs. It does not see the generator's hidden
    reasoning or mutation history.
    """

    def __init__(self, generate_text, *, seed: int = 99173):
        self.generate_text = generate_text
        self.seed = seed

    def __call__(self, task: dict, artifact: ProgramArtifact) -> float:
        if not artifact.exact_train or artifact.report is None:
            return 0.0

        hypothesis = artifact.metadata.get("hypothesis", "")
        prompt = f"""
You are an independent ARC-AGI-2 generalization reviewer.

A solver produced a program that matches every training example exactly.
Your job is NOT to reward train fit. Decide whether the rule is likely to
generalize to the unseen test input rather than exploiting accidental details.

Inspect:
- whether the hypothesis explains all examples with one coherent rule;
- whether the code hard-codes example-specific coordinates, sizes, or colors;
- whether the rule uses stable object/relationship/contextual properties;
- whether another plausible rule could fit the demonstrations equally well.

Return exactly one line:
APPROVE <confidence 0.0-1.0>
or
REJECT <confidence 0.0-1.0>

Training task:
{task["train"]}

Test inputs:
{[x["input"] for x in task["test"]]}

Hypothesis:
{hypothesis}

Program:
{artifact.source}

Proposed test outputs:
{list(artifact.report.test_outputs)}
""".strip()

        response = self.generate_text([prompt], self.seed)[0].strip()
        m = re.search(r"\b(APPROVE|REJECT)\s+([01](?:\.\d+)?)", response, flags=re.I)
        if not m:
            return structural_generalization_score(task, artifact)

        verdict = m.group(1).upper()
        confidence = max(0.0, min(1.0, float(m.group(2))))
        return confidence if verdict == "APPROVE" else -confidence
