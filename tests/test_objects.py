from arcforge.objects import object_candidate_grids


def test_extract_unique_small_object_crop():
    task = {
        "train": [
            {
                "input": [
                    [0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 2],
                    [0, 0, 0, 0, 0],
                ],
                "output": [[2]],
            }
        ],
        "test": [
            {
                "input": [
                    [0, 3, 3, 0, 0],
                    [0, 3, 3, 0, 4],
                    [0, 0, 0, 0, 0],
                ]
            }
        ],
    }
    assert [[4]] in object_candidate_grids(task)[0]


def test_keep_largest_object():
    task = {
        "train": [
            {
                "input": [[1, 1, 0, 2], [1, 1, 0, 0]],
                "output": [[1, 1, 0, 0], [1, 1, 0, 0]],
            }
        ],
        "test": [{"input": [[3, 3, 0, 4], [3, 3, 0, 0]]}],
    }
    assert [[3, 3, 0, 0], [3, 3, 0, 0]] in object_candidate_grids(task)[0]
