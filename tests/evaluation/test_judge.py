import pytest

from app.evaluation.judge import (
    JudgeResult,
    _parse_judge_result,
)


def test_parse_valid_judge_result():
    result = _parse_judge_result(
        '{"score": 1.0, "reason": "The answer is fully correct."}'
    )

    assert result == JudgeResult(
        score=1.0,
        reason="The answer is fully correct.",
    )


def test_parse_half_score():
    result = _parse_judge_result(
        '{"score": 0.5, "reason": "The answer is partially correct."}'
    )

    assert result.score == 0.5


def test_parse_zero_score():
    result = _parse_judge_result(
        '{"score": 0.0, "reason": "The answer is incorrect."}'
    )

    assert result.score == 0.0


def test_parse_json_inside_markdown_fence():
    result = _parse_judge_result(
        '```json\n'
        '{"score": 1.0, "reason": "Supported."}'
        '\n```'
    )

    assert result.score == 1.0
    assert result.reason == "Supported."


def test_parse_json_with_surrounding_text():
    result = _parse_judge_result(
        'Here is the evaluation:\n'
        '{"score": 0.5, "reason": "Partially supported."}'
    )

    assert result.score == 0.5


@pytest.mark.parametrize(
    "response",
    [
        '{"reason": "Missing score."}',
        '{"score": 1.0}',
        '{"score": 0.25, "reason": "Invalid score."}',
        '{"score": true, "reason": "Boolean is invalid."}',
        '{"score": 1.0, "reason": ""}',
        "not json",
    ],
)
def test_parse_invalid_judge_result(response):
    with pytest.raises(ValueError):
        _parse_judge_result(response)
