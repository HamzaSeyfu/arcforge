from arcforge.synthesis import synthesis_candidate_grids


def test_synthesis_identity_family_runs_without_nameerror():
    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        ],
        "test": [
            {"input": [[5, 6], [7, 8]]},
        ],
    }
    pools = synthesis_candidate_grids(task, max_depth=1, max_programs=8)
    assert pools
    assert [[5, 6], [7, 8]] in pools[0]
