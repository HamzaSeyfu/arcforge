from arcforge.relational import relational_candidate_grids


def test_plus_recolor_rule():
    inp = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]
    out = [
        [0, 2, 0],
        [2, 2, 2],
        [0, 2, 0],
    ]
    task = {
        "train": [{"input": inp, "output": out}],
        "test": [{"input": inp}],
    }
    assert out in relational_candidate_grids(task)[0]


def test_vertical_symmetry_prune_same_color_outlier():
    inp = [
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1],
    ]
    out = [
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
    ]
    task = {
        "train": [{"input": inp, "output": out}],
        "test": [{"input": inp}],
    }
    assert out in relational_candidate_grids(task)[0]
