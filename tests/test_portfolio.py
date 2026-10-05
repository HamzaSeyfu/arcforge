from arcforge.evaluation import score_submission
from arcforge.portfolio import (
    complementarity,
    greedy_solver_order,
    pairwise_complementarity,
    portfolio_coverage,
)


def _fixtures():
    solutions = {
        "a": [[[1]], [[2]]],
        "b": [[[3]]],
        "c": [[[4]]],
    }

    sub_a = {
        "a": [
            {"attempt_1": [[1]], "attempt_2": [[0]]},
            {"attempt_1": [[0]], "attempt_2": [[0]]},
        ],
        "b": [{"attempt_1": [[3]], "attempt_2": [[0]]}],
        "c": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    sub_b = {
        "a": [
            {"attempt_1": [[0]], "attempt_2": [[0]]},
            {"attempt_1": [[2]], "attempt_2": [[0]]},
        ],
        "b": [{"attempt_1": [[3]], "attempt_2": [[0]]}],
        "c": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
    }
    sub_c = {
        "a": [
            {"attempt_1": [[0]], "attempt_2": [[0]]},
            {"attempt_1": [[0]], "attempt_2": [[0]]},
        ],
        "b": [{"attempt_1": [[0]], "attempt_2": [[0]]}],
        "c": [{"attempt_1": [[4]], "attempt_2": [[0]]}],
    }
    return solutions, sub_a, sub_b, sub_c


def test_score_and_complementarity():
    solutions, sub_a, sub_b, _ = _fixtures()

    ra = score_submission(sub_a, solutions)
    rb = score_submission(sub_b, solutions)
    c = complementarity(sub_a, sub_b, solutions)

    assert ra.solved_rows == 2
    assert rb.solved_rows == 2
    assert c.shared == 1
    assert c.unique_a == 1
    assert c.unique_b == 1
    assert c.oracle_union == 3


def test_n_way_portfolio_coverage_and_greedy_order():
    solutions, sub_a, sub_b, sub_c = _fixtures()
    submissions = {"alpha": sub_a, "beta": sub_b, "gamma": sub_c}

    report = portfolio_coverage(submissions, solutions)
    assert report.total_rows == 4
    assert report.per_solver == {"alpha": 2, "beta": 2, "gamma": 1}
    assert report.oracle_union == 4
    assert report.oracle_union_accuracy == 1.0

    steps = greedy_solver_order(submissions, solutions)
    assert [step.solver for step in steps] == ["alpha", "beta", "gamma"]
    assert [step.added for step in steps] == [2, 1, 1]
    assert [step.cumulative for step in steps] == [2, 3, 4]


def test_pairwise_complementarity_for_three_solvers():
    solutions, sub_a, sub_b, sub_c = _fixtures()
    pairs = pairwise_complementarity(
        {"alpha": sub_a, "beta": sub_b, "gamma": sub_c},
        solutions,
    )

    assert set(pairs) == {
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("beta", "gamma"),
    }
    assert pairs[("alpha", "beta")].oracle_union == 3
    assert pairs[("alpha", "gamma")].oracle_union == 3
    assert pairs[("beta", "gamma")].oracle_union == 3
