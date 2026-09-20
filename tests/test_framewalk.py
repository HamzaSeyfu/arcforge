from arcforge.framewalk import framewalk_candidate_grids


def test_framewalk_smoke_returns_rows():
    # Structural smoke test. Full accuracy is measured on the 172-row eval.
    task = {
        "train": [
            {
                "input": [
                    [0, 1, 0, 2, 0, 0, 0],
                    [3, 3, 3, 0, 4, 4, 4],
                    [3, 1, 3, 0, 4, 2, 4],
                    [3, 3, 3, 0, 4, 4, 4],
                ],
                "output": [
                    [0, 1, 0, 2, 0, 0, 0],
                    [3, 3, 3, 0, 4, 4, 4],
                    [3, 1, 3, 1, 4, 2, 4],
                    [3, 3, 3, 0, 4, 4, 4],
                ],
            }
        ],
        "test": [
            {
                "input": [
                    [0, 1, 0, 2, 0, 0, 0],
                    [3, 3, 3, 0, 4, 4, 4],
                    [3, 1, 3, 0, 4, 2, 4],
                    [3, 3, 3, 0, 4, 4, 4],
                ]
            }
        ],
    }
    assert isinstance(framewalk_candidate_grids(task), list)
