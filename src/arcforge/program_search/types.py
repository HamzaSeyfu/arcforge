from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Grid = list[list[int]]


@dataclass
class VerificationReport:
    exact_train: bool
    exact_examples: int
    total_examples: int
    mean_soft_score: float
    per_example_soft: tuple[float, ...]
    train_outputs: tuple[Grid | None, ...]
    test_outputs: tuple[Grid | None, ...]
    errors: tuple[str | None, ...]
    runtime_seconds: float = 0.0


@dataclass
class ProgramArtifact:
    program_id: str
    source: str
    family: str
    generation: int
    parent_ids: tuple[str, ...] = ()
    report: VerificationReport | None = None
    reviewer_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exact_train(self) -> bool:
        return bool(self.report and self.report.exact_train)

    @property
    def soft_score(self) -> float:
        return self.report.mean_soft_score if self.report else 0.0


@dataclass(frozen=True)
class SearchConfig:
    initial_candidates: int = 8
    mutations_per_generation: int = 8
    generations: int = 4
    parent_pool_size: int = 4
    max_exact_programs: int = 12
    max_programs_total: int = 64
    timeout_seconds: float = 1.5
    source_char_limit: int = 12000


@dataclass
class SearchResult:
    programs: list[ProgramArtifact]
    exact_programs: list[ProgramArtifact]
    candidate_pools: list[list[Grid]]
    generations_run: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
