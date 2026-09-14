from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerQualityEvaluationResult:
    """
    Semantic quality evaluation for a generated RAG answer.

    Scores are normalized to the range 0.0 to 1.0.
    """

    correctness_score: float
    correctness_reason: str
    groundedness_score: float
    groundedness_reason: str
    citation_correctness_score: float


def _validate_score(score: float, name: str) -> None:
    """
    Validate that a quality score is within the expected range.
    """

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0"
        )


def create_answer_quality_result(
    correctness_score: float,
    correctness_reason: str,
    groundedness_score: float,
    groundedness_reason: str,
    citation_correctness_score: float,
) -> AnswerQualityEvaluationResult:
    """
    Create a validated answer-quality evaluation result.
    """

    _validate_score(
        correctness_score,
        "correctness_score",
    )

    _validate_score(
        groundedness_score,
        "groundedness_score",
    )

    _validate_score(
        citation_correctness_score,
        "citation_correctness_score",
    )

    if not correctness_reason.strip():
        raise ValueError(
            "correctness_reason cannot be empty"
        )

    if not groundedness_reason.strip():
        raise ValueError(
            "groundedness_reason cannot be empty"
        )

    return AnswerQualityEvaluationResult(
        correctness_score=correctness_score,
        correctness_reason=correctness_reason.strip(),
        groundedness_score=groundedness_score,
        groundedness_reason=groundedness_reason.strip(),
        citation_correctness_score=citation_correctness_score,
    )