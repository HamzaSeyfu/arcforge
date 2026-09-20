from arcforge.panels import panel_candidate_grids


def test_vertical_two_panel_xor_mask_with_learned_colors():
    task = {
        "train": [
            {
                "input": [
                    [1, 0],
                    [0, 1],
                    [0, 0],
                    [1, 1],
                ],
                "output": [
                    [2, 0],
                    [2, 0],
                ],
            }
        ],
        "test": [
            {
                "input": [
                    [1, 1],
                    [0, 0],
                    [1, 0],
                    [0, 0],
                ]
            }
        ],
    }
    pools = panel_candidate_grids(task)
    assert pools[0]


def test_select_panel():
    task = {
        "train": [
            {
                "input": [[1, 2, 9, 9], [3, 4, 9, 9]],
                "output": [[1, 2], [3, 4]],
            }
        ],
        "test": [{"input": [[5, 6, 0, 0], [7, 8, 0, 0]]}],
    }
    assert [[5, 6], [7, 8]] in panel_candidate_grids(task)[0]
