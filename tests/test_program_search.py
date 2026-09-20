from arcforge.program_search.engine import ProgramSearchEngine
from arcforge.program_search.types import ProgramArtifact, SearchConfig
from arcforge.program_search.verifier import verify_program

IDENTITY = """
def transform(grid):
    return grid.copy()
"""

WRONG = """
def transform(grid):
    return np.zeros_like(grid)
"""


class FakeGenerator:
    def generate_initial(self, task, *, count, seed):
        return [ProgramArtifact("wrong-0", WRONG, "fake", 0)]

    def mutate(self, task, parents, *, count, generation, seed):
        return [
            ProgramArtifact(
                f"identity-{generation}",
                IDENTITY,
                "fake",
                generation,
                tuple(p.program_id for p in parents),
            )
        ]


def test_verifier_exact_and_soft_score():
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            {"input": [[5]], "output": [[5]]},
        ],
        "test": [{"input": [[7, 8]]}],
    }
    report = verify_program(IDENTITY, task)
    assert report.exact_train
    assert report.exact_examples == 2
    assert report.mean_soft_score == 1.0
    assert report.test_outputs[0] == [[7, 8]]


def test_refinement_loop_recovers_exact_program():
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        ],
        "test": [{"input": [[9, 8], [7, 6]]}],
    }
    engine = ProgramSearchEngine(
        FakeGenerator(),
        config=SearchConfig(
            initial_candidates=1,
            mutations_per_generation=1,
            generations=2,
            parent_pool_size=1,
            max_exact_programs=2,
            max_programs_total=8,
        ),
    )
    result = engine.search(task)
    assert result.exact_programs
    assert result.candidate_pools[0][0] == [[9, 8], [7, 6]]
    assert result.diagnostics["best_soft_score"] == 1.0


def test_imports_are_rejected():
    task = {
        "train": [{"input": [[1]], "output": [[1]]}],
        "test": [{"input": [[2]]}],
    }
    report = verify_program(
        "import os\ndef transform(grid):\n    return grid",
        task,
    )
    assert not report.exact_train
    assert "imports are forbidden" in report.errors[0]
