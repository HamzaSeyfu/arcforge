from arcforge.evaluation import score_submission
from arcforge.portfolio import complementarity


def test_score_and_complementarity():
    solutions = {
        "a": [[[1]], [[2]]],
        "b": [[[3]]],
    }

    sub_a = {
        "a": [
            {"attempt_1": [[1]], "attempt_2": [[0]]},
            {"attempt_1": [[0]], "attempt_2": [[0]]},
        ],
        "b": [{"attempt_1": [[3]], "attempt_2": [[0]]}],
    }
    sub_b = {
        "a": [
            {"attempt_1": [[0]], "attempt_2": [[0]]},
            {"attempt_1": [[2]], "attempt_2": [[0]]},
        ],
        "b": [{"attempt_1": [[3]], "attempt_2": [[0]]}],
    }

    ra = score_submission(sub_a, solutions)
    rb = score_submission(sub_b, solutions)
    c = complementarity(sub_a, sub_b, solutions)

    assert ra.solved_rows == 2
    assert rb.solved_rows == 2
    assert c.shared == 1
    assert c.unique_a == 1
    assert c.unique_b == 1
    assert c.oracle_union == 3
