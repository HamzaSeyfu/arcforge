from __future__ import annotations

from .aggregate import candidate_pools_from_exact
from .archive import BehaviorArchive
from .generator import ProgramGenerator
from .types import ProgramArtifact, SearchConfig, SearchResult
from .verifier import verify_program


class ProgramSearchEngine:
    """
    Execution-verified refinement loop inspired by Poetiq, Confluence, Athanor,
    and evolutionary program search.

    The engine itself is model-agnostic. A local Kaggle model only needs to
    implement ProgramGenerator.
    """

    def __init__(
        self,
        generator: ProgramGenerator,
        *,
        config: SearchConfig | None = None,
        reviewer=None,
    ):
        self.generator = generator
        self.config = config or SearchConfig()
        self.reviewer = reviewer

    def _evaluate(self, artifact: ProgramArtifact, task: dict) -> ProgramArtifact:
        report = verify_program(
            artifact.source,
            task,
            timeout_seconds=self.config.timeout_seconds,
            source_char_limit=self.config.source_char_limit,
        )
        artifact.report = report

        if self.reviewer is not None and report.exact_train:
            try:
                artifact.reviewer_score = float(self.reviewer(task, artifact))
            except Exception:
                artifact.reviewer_score = 0.0
        return artifact

    @staticmethod
    def _parent_rank(artifact: ProgramArtifact):
        report = artifact.report
        if report is None:
            return (0, 0.0, 0.0, 0)
        return (
            report.exact_examples,
            report.mean_soft_score,
            artifact.reviewer_score,
            -len(artifact.source),
        )

    def _choose_parents(self, artifacts: list[ProgramArtifact]) -> list[ProgramArtifact]:
        ranked = sorted(artifacts, key=self._parent_rank, reverse=True)
        chosen = []
        signatures = set()

        for artifact in ranked:
            if artifact.report is None:
                continue
            signature = tuple(round(x, 3) for x in artifact.report.per_example_soft)
            if signature in signatures and len(chosen) >= 2:
                continue
            signatures.add(signature)
            chosen.append(artifact)
            if len(chosen) >= self.config.parent_pool_size:
                break
        return chosen

    def search(self, task: dict, *, seed: int = 0) -> SearchResult:
        cfg = self.config
        archive = BehaviorArchive()
        all_artifacts = []

        pending = self.generator.generate_initial(
            task,
            count=cfg.initial_candidates,
            seed=seed,
        )

        generations_run = 0
        for generation in range(cfg.generations + 1):
            generations_run = generation
            for artifact in pending:
                if len(all_artifacts) >= cfg.max_programs_total:
                    break
                evaluated = self._evaluate(artifact, task)
                if archive.add(evaluated):
                    all_artifacts.append(evaluated)

            exact = [p for p in all_artifacts if p.exact_train]
            if (
                len(exact) >= cfg.max_exact_programs
                or len(all_artifacts) >= cfg.max_programs_total
                or generation >= cfg.generations
            ):
                break

            parents = self._choose_parents(all_artifacts)
            pending = self.generator.mutate(
                task,
                parents,
                count=cfg.mutations_per_generation,
                generation=generation + 1,
                seed=seed + 1009 * (generation + 1),
            )

        exact = [p for p in all_artifacts if p.exact_train]
        pools = candidate_pools_from_exact(exact)

        return SearchResult(
            programs=all_artifacts,
            exact_programs=exact,
            candidate_pools=pools,
            generations_run=generations_run,
            diagnostics={
                "programs_evaluated": len(all_artifacts),
                "exact_programs": len(exact),
                "best_soft_score": max(
                    (p.soft_score for p in all_artifacts),
                    default=0.0,
                ),
                "distinct_test_candidates": [len(row) for row in pools],
            },
        )
