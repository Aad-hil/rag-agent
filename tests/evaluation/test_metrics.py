import pytest

from app.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_rate_at_k_returns_one_when_relevant_page_is_retrieved():
    assert hit_rate_at_k(
        retrieved_pages=[10, 20, 30],
        relevant_pages=[20],
        k=3,
    ) == 1.0


def test_hit_rate_at_k_returns_zero_when_no_relevant_page_is_retrieved():
    assert hit_rate_at_k(
        retrieved_pages=[10, 20, 30],
        relevant_pages=[40],
        k=3,
    ) == 0.0


def test_hit_rate_at_k_respects_k():
    assert hit_rate_at_k(
        retrieved_pages=[10, 20, 30],
        relevant_pages=[30],
        k=2,
    ) == 0.0


def test_recall_at_k_returns_fraction_of_relevant_pages_retrieved():
    assert recall_at_k(
        retrieved_pages=[20, 50, 82, 99],
        relevant_pages=[20, 82, 150, 151],
        k=4,
    ) == 0.5


def test_recall_at_k_returns_one_when_all_relevant_pages_are_retrieved():
    assert recall_at_k(
        retrieved_pages=[20, 82, 150, 151],
        relevant_pages=[20, 82, 150, 151],
        k=4,
    ) == 1.0


def test_recall_at_k_handles_duplicate_retrieved_pages():
    assert recall_at_k(
        retrieved_pages=[20, 20, 82],
        relevant_pages=[20, 82],
        k=3,
    ) == 1.0


def test_precision_at_k_counts_relevant_results():
    assert precision_at_k(
        retrieved_pages=[20, 50, 82, 99],
        relevant_pages=[20, 82],
        k=4,
    ) == 0.5


def test_precision_at_k_handles_fewer_results_than_k():
    assert precision_at_k(
        retrieved_pages=[20, 50],
        relevant_pages=[20],
        k=5,
    ) == 0.5


def test_reciprocal_rank_returns_first_relevant_rank():
    assert reciprocal_rank(
        retrieved_pages=[50, 60, 20, 30],
        relevant_pages=[20],
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_one_for_rank_one():
    assert reciprocal_rank(
        retrieved_pages=[20, 50, 60],
        relevant_pages=[20],
    ) == 1.0


def test_reciprocal_rank_returns_zero_when_no_relevant_result_exists():
    assert reciprocal_rank(
        retrieved_pages=[50, 60, 70],
        relevant_pages=[20],
    ) == 0.0


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank(
        rankings=[
            [20, 50, 60],
            [50, 82, 90],
            [10, 11, 12],
        ],
        relevant_pages=[
            [20],
            [82],
            [30],
        ],
    ) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_metrics_handle_empty_retrieved_results():
    assert hit_rate_at_k([], [20], 5) == 0.0
    assert recall_at_k([], [20], 5) == 0.0
    assert precision_at_k([], [20], 5) == 0.0
    assert reciprocal_rank([], [20]) == 0.0


def test_metrics_handle_empty_relevant_pages():
    assert hit_rate_at_k([20], [], 5) == 0.0
    assert recall_at_k([20], [], 5) == 0.0
    assert precision_at_k([20], [], 5) == 0.0
    assert reciprocal_rank([20], []) == 0.0


def test_metrics_reject_non_positive_k():
    with pytest.raises(ValueError):
        hit_rate_at_k([20], [20], 0)

    with pytest.raises(ValueError):
        recall_at_k([20], [20], 0)

    with pytest.raises(ValueError):
        precision_at_k([20], [20], 0)
