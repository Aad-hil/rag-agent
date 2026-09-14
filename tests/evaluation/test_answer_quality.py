import pytest

from app.evaluation.answer_quality import (
    AnswerQualityEvaluationResult,
    create_answer_quality_result,
)


def test_create_answer_quality_result():
    result = create_answer_quality_result(
        correctness_score=0.9,
        correctness_reason="The answer is mostly correct.",
        groundedness_score=0.8,
        groundedness_reason="Most claims are supported.",
        citation_correctness_score=1.0,
    )

    assert isinstance(
        result,
        AnswerQualityEvaluationResult,
    )

    assert result.correctness_score == 0.9
    assert result.correctness_reason == (
        "The answer is mostly correct."
    )

    assert result.groundedness_score == 0.8
    assert result.groundedness_reason == (
        "Most claims are supported."
    )

    assert result.citation_correctness_score == 1.0


@pytest.mark.parametrize(
    "field",
    [
        "correctness_score",
        "groundedness_score",
        "citation_correctness_score",
    ],
)
def test_score_cannot_be_below_zero(field):
    scores = {
        "correctness_score": 0.5,
        "correctness_reason": "Correctness reason.",
        "groundedness_score": 0.5,
        "groundedness_reason": "Groundedness reason.",
        "citation_correctness_score": 0.5,
    }

    scores[field] = -0.1

    with pytest.raises(ValueError):
        create_answer_quality_result(**scores)


@pytest.mark.parametrize(
    "field",
    [
        "correctness_score",
        "groundedness_score",
        "citation_correctness_score",
    ],
)
def test_score_cannot_be_above_one(field):
    scores = {
        "correctness_score": 0.5,
        "correctness_reason": "Correctness reason.",
        "groundedness_score": 0.5,
        "groundedness_reason": "Groundedness reason.",
        "citation_correctness_score": 0.5,
    }

    scores[field] = 1.1

    with pytest.raises(ValueError):
        create_answer_quality_result(**scores)


def test_correctness_reason_cannot_be_empty():
    with pytest.raises(ValueError):
        create_answer_quality_result(
            correctness_score=1.0,
            correctness_reason="",
            groundedness_score=1.0,
            groundedness_reason="Supported.",
            citation_correctness_score=1.0,
        )


def test_groundedness_reason_cannot_be_empty():
    with pytest.raises(ValueError):
        create_answer_quality_result(
            correctness_score=1.0,
            correctness_reason="Correct.",
            groundedness_score=1.0,
            groundedness_reason="",
            citation_correctness_score=1.0,
        )