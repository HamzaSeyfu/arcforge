from arcforge.candidate_analysis import (
    candidate_oracle_coverage,
    oracle_union_coverage,
)


def test_candidate_oracle_coverage_and_union():
    solutions = {
        "a": [[[1]], [[2]]],
        "b": [[[3]]],
    }

    neural = {
        "a": [
            [[[1]], [[9]]],
            [[[8]]],
        ],
        "b": [
            [[[0]]],
        ],
    }

    symbolic = {
        "a": [
            [[[7]]],
            [[[2]]],
        ],
        "b": [
            [[[3]]],
        ],
    }

    r1 = candidate_oracle_coverage(neural, solutions)
    assert r1.rows == 3
    assert r1.oracle_solved_rows == 1

    union = oracle_union_coverage([neural, symbolic], solutions)
    assert union.oracle_solved_rows == 3
    assert union.oracle_row_accuracy == 1.0
