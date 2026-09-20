from arcforge.symbolic import candidate_grids


def test_symbolic_rotation_is_execution_verified():
    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[3, 1], [4, 2]],
            }
        ],
        "test": [{"input": [[5, 6], [7, 8]]}],
    }
    pools = candidate_grids(task)
    assert [[7, 5], [8, 6]] in pools[0]


def test_symbolic_mask_tile():
    task = {
        "train": [
            {
                "input": [[1, 0], [0, 1]],
                "output": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        ],
        "test": [{"input": [[2, 0], [2, 2]]}],
    }
    pools = candidate_grids(task)
    assert pools[0]
