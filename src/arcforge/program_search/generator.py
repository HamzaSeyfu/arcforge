from __future__ import annotations

import hashlib
import re
from typing import Callable, Protocol

from .feedback import build_feedback
from .types import ProgramArtifact


class ProgramGenerator(Protocol):
    def generate_initial(
        self,
        task: dict,
        *,
        count: int,
        seed: int,
    ) -> list[ProgramArtifact]:
        ...

    def mutate(
        self,
        task: dict,
        parents: list[ProgramArtifact],
        *,
        count: int,
        generation: int,
        seed: int,
    ) -> list[ProgramArtifact]:
        ...


def _task_text(task: dict) -> str:
    lines = []
    for i, pair in enumerate(task["train"], start=1):
        lines.append(f"Example {i} input: {pair['input']}")
        lines.append(f"Example {i} output: {pair['output']}")
    for i, test in enumerate(task["test"], start=1):
        lines.append(f"Test {i} input: {test['input']}")
    return "\n".join(lines)


def _extract_python(text: str) -> str | None:
    xml = re.findall(r"<python>\s*(.*?)\s*</python>", text, flags=re.S | re.I)
    if xml:
        return xml[-1].strip()

    fenced = re.findall(
        r"[`]{3}python\s*(.*?)[`]{3}",
        text,
        flags=re.S | re.I,
    )
    if fenced:
        return fenced[-1].strip()

    if "def transform" in text:
        start = text.find("def transform")
        return text[start:].strip()
    return None


def _program_id(source: str, family: str, generation: int, index: int) -> str:
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{family}-g{generation}-{index}-{digest}"


class TextModelProgramGenerator:
    """
    Poetiq/Confluence-style generator over any local text-generation callable.

    generate_text receives a list of prompts plus a seed and returns one response
    per prompt. The search engine remains independent from transformers/vLLM.
    """

    def __init__(
        self,
        generate_text: Callable[[list[str], int], list[str]],
        *,
        family: str = "local-code-model",
    ):
        self.generate_text = generate_text
        self.family = family

    def _initial_prompt(self, task: dict) -> str:
        return f"""
You are solving an ARC-AGI-2 task by writing executable Python.

Write exactly one candidate transform function:
    def transform(grid: np.ndarray) -> np.ndarray

Rules:
- numpy is already available as np; do not import anything.
- output must be a rectangular grid of integers 0..9, max 30x30.
- infer one general rule from ALL training examples.
- prefer object, relationship, and contextual rules over memorized coordinates.
- return only <python> ... </python> containing the function.

Task:
{_task_text(task)}
""".strip()

    def _mutation_prompt(
        self,
        task: dict,
        parents: list[ProgramArtifact],
    ) -> str:
        parent_blocks = []
        for i, parent in enumerate(parents, start=1):
            parent_blocks.append(
                f"""
Candidate {i}
code:
<python>
{parent.source}
</python>
verification:
{build_feedback(parent, task)}
""".strip()
            )

        return f"""
You are repairing ARC-AGI-2 Python transformation programs.

The candidates below are partial attempts. Study their exact verification
feedback, preserve useful ideas, and produce ONE materially improved program.
Do not patch coordinates from the examples. Generalize the transformation.

Requirements:
- define transform(grid: np.ndarray) -> np.ndarray
- numpy is already available as np; no imports
- output colors 0..9
- return only <python> ... </python>

Task:
{_task_text(task)}

Previous candidates:
{chr(10).join(parent_blocks)}
""".strip()

    def _responses_to_artifacts(
        self,
        responses: list[str],
        *,
        generation: int,
        parent_ids: tuple[str, ...],
    ) -> list[ProgramArtifact]:
        out = []
        for i, text in enumerate(responses):
            source = _extract_python(text)
            if not source:
                continue
            out.append(
                ProgramArtifact(
                    program_id=_program_id(source, self.family, generation, i),
                    source=source,
                    family=self.family,
                    generation=generation,
                    parent_ids=parent_ids,
                )
            )
        return out

    def generate_initial(self, task: dict, *, count: int, seed: int):
        prompts = [self._initial_prompt(task) for _ in range(count)]
        responses = self.generate_text(prompts, seed)
        return self._responses_to_artifacts(
            responses,
            generation=0,
            parent_ids=(),
        )

    def mutate(
        self,
        task: dict,
        parents: list[ProgramArtifact],
        *,
        count: int,
        generation: int,
        seed: int,
    ):
        if not parents:
            return self.generate_initial(task, count=count, seed=seed)

        prompts = []
        parent_sets = []
        for i in range(count):
            width = min(3, len(parents))
            selected = [parents[(i + j) % len(parents)] for j in range(width)]
            prompts.append(self._mutation_prompt(task, selected))
            parent_sets.append(selected)

        responses = self.generate_text(prompts, seed)
        out = []
        for i, response in enumerate(responses):
            source = _extract_python(response)
            if not source:
                continue
            selected = parent_sets[i]
            out.append(
                ProgramArtifact(
                    program_id=_program_id(source, self.family, generation, i),
                    source=source,
                    family=self.family,
                    generation=generation,
                    parent_ids=tuple(p.program_id for p in selected),
                )
            )
        return out
