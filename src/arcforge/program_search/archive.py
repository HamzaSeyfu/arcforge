from __future__ import annotations

from .types import ProgramArtifact


def _grid_key(grid):
    return None if grid is None else tuple(tuple(row) for row in grid)


def behavior_key(artifact: ProgramArtifact):
    if artifact.report is None:
        return ("unevaluated", artifact.program_id)
    return (
        tuple(_grid_key(g) for g in artifact.report.train_outputs),
        tuple(_grid_key(g) for g in artifact.report.test_outputs),
    )


class BehaviorArchive:
    def __init__(self):
        self._by_behavior = {}
        self._source_hashes = set()

    def add(self, artifact: ProgramArtifact) -> bool:
        source_hash = hash(artifact.source)
        if source_hash in self._source_hashes:
            return False

        key = behavior_key(artifact)
        previous = self._by_behavior.get(key)
        if previous is not None:
            if len(artifact.source) >= len(previous.source):
                self._source_hashes.add(source_hash)
                return False
            self._by_behavior[key] = artifact
            self._source_hashes.add(source_hash)
            return True

        self._by_behavior[key] = artifact
        self._source_hashes.add(source_hash)
        return True

    def artifacts(self) -> list[ProgramArtifact]:
        return list(self._by_behavior.values())
