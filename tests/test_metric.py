from arc2.metric import score_submission
from arc2.submission import is_valid_grid, make_submission, validate

SOL = {"a": [[[1, 2]]], "b": [[[0]], [[3]]]}
CH = {
    "a": {"train": [], "test": [{"input": [[1, 2]]}]},
    "b": {"train": [], "test": [{"input": [[0]]}, {"input": [[5]]}]},
}


def test_score_counts_outputs_not_tasks():
    sub = {
        "a": [{"attempt_1": [[9, 9]], "attempt_2": [[1, 2]]}],
        "b": [{"attempt_1": [[0]], "attempt_2": [[0]]}, {"attempt_1": [[4]], "attempt_2": [[4]]}],
    }
    assert score_submission(sub, SOL) == 2 / 3


def test_shape_must_match():
    sub = {"a": [{"attempt_1": [[1], [2]], "attempt_2": [[1, 2, 0]]}], "b": []}
    assert score_submission(sub, SOL) == 0.0


def test_identity_submission_is_valid_and_scores():
    sub = make_submission(CH, lambda t, i: (t["test"][i]["input"], None))
    assert validate(sub, CH) == []
    assert score_submission(sub, SOL) == 2 / 3


def test_crashing_predictor_still_valid():
    def boom(task, i):
        raise RuntimeError

    sub = make_submission(CH, boom)
    assert validate(sub, CH) == []


def test_grid_validation():
    assert is_valid_grid([[0, 9]])
    assert not is_valid_grid([[10]])
    assert not is_valid_grid([[1, 2], [3]])
    assert not is_valid_grid([])
    assert not is_valid_grid([[True]])
    assert not is_valid_grid([[0] * 31])


def test_validate_reports_missing_task():
    assert validate({"a": [{"attempt_1": [[1]], "attempt_2": [[1]]}]}, CH)
