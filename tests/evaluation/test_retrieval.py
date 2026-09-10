from unittest.mock import patch

import pytest

from app.evaluation.dataset import EvaluationQuestion
from app.evaluation.retrieval import (
    evaluate_dataset,
    evaluate_question,
)
from app.retrieval.search import SearchResult


def make_result(
    page_number: int,
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        text=f"Page {page_number} content",
        score=score,
        source="test.pdf",
        page_number=page_number,
        chunk_index=0,
    )


def test_evaluate_question_calculates_metrics():
    item = EvaluationQuestion(
        question="Test question",
        relevant_pages=(20, 82),
    )

    results = [
        make_result(20),
        make_result(50),
        make_result(82),
        make_result(90),
    ]

    with patch(
        "app.evaluation.retrieval.search",
        return_value=results,
    ):
        evaluation = evaluate_question(
            item,
            limit=4,
        )

    assert evaluation.question == "Test question"
    assert evaluation.expected_pages == (20, 82)
    assert evaluation.retrieved_pages == (
        20,
        50,
        82,
        90,
    )

    assert evaluation.missing_pages == ()

    assert evaluation.irrelevant_pages == (
        50,
        90,
    )

    assert evaluation.duplicate_pages == ()

    assert evaluation.hit_rate == 1.0
    assert evaluation.recall == 1.0
    assert evaluation.precision == 0.5
    assert evaluation.reciprocal_rank == 1.0


def test_evaluate_question_respects_limit():
    item = EvaluationQuestion(
        question="Test question",
        relevant_pages=(82,),
    )

    results = [
        make_result(20),
        make_result(50),
        make_result(82),
    ]

    with patch(
        "app.evaluation.retrieval.search",
        return_value=results,
    ):
        evaluation = evaluate_question(
            item,
            limit=2,
        )

    assert evaluation.retrieved_pages == (
        20,
        50,
    )

    assert evaluation.missing_pages == (
        82,
    )

    assert evaluation.irrelevant_pages == (
        20,
        50,
    )

    assert evaluation.duplicate_pages == ()

    assert evaluation.hit_rate == 0.0
    assert evaluation.recall == 0.0
    assert evaluation.precision == 0.0
    assert evaluation.reciprocal_rank == 0.0


def test_evaluate_question_detects_duplicate_pages():
    item = EvaluationQuestion(
        question="Test question",
        relevant_pages=(20, 82),
    )

    results = [
        make_result(20),
        make_result(20),
        make_result(50),
        make_result(82),
    ]

    with patch(
        "app.evaluation.retrieval.search",
        return_value=results,
    ):
        evaluation = evaluate_question(
            item,
            limit=4,
        )

    assert evaluation.missing_pages == ()

    assert evaluation.irrelevant_pages == (
        50,
    )

    assert evaluation.duplicate_pages == (
        20,
    )


def test_evaluate_dataset_aggregates_metrics():
    dataset = [
        EvaluationQuestion(
            question="Question one",
            relevant_pages=(20,),
        ),
        EvaluationQuestion(
            question="Question two",
            relevant_pages=(82,),
        ),
    ]

    search_results = [
        [
            make_result(20),
            make_result(50),
        ],
        [
            make_result(50),
            make_result(82),
        ],
    ]

    with patch(
        "app.evaluation.retrieval.search",
        side_effect=search_results,
    ):
        summary = evaluate_dataset(
            dataset,
            limit=2,
        )

    assert len(summary.results) == 2

    assert summary.hit_rate == 1.0
    assert summary.recall == 1.0
    assert summary.precision == 0.5

    assert summary.mrr == pytest.approx(0.75)


def test_evaluate_empty_dataset():
    summary = evaluate_dataset(
        [],
        limit=5,
    )

    assert summary.results == ()
    assert summary.hit_rate == 0.0
    assert summary.recall == 0.0
    assert summary.precision == 0.0
    assert summary.mrr == 0.0
