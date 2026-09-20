from arcforge.synthesis import synthesis_candidate_grids


def test_depth_two_crop_then_rotate():
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                    [0, 0, 0, 0],
                ],
                "output": [[3, 1], [4, 2]],
            }
        ],
        "test": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 5, 6, 0],
                    [0, 7, 8, 0],
                    [0, 0, 0, 0],
                ]
            }
        ],
    }
    pools = synthesis_candidate_grids(task, max_depth=2)
    assert [[7, 5], [8, 6]] in pools[0]


def test_final_color_map_after_geometry():
    task = {
        "train": [
            {
                "input": [[1, 0], [0, 1]],
                "output": [[2, 0], [0, 2]],
            }
        ],
        "test": [{"input": [[1, 1], [0, 1]]}],
    }
    pools = synthesis_candidate_grids(task, max_depth=1)
    assert [[2, 2], [0, 2]] in pools[0]
