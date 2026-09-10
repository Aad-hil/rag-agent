from dataclasses import dataclass

from app.evaluation.dataset import EvaluationQuestion
from app.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.search import search


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """
    Evaluation results for a single retrieval question.
    """

    question: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    missing_pages: tuple[int, ...]
    irrelevant_pages: tuple[int, ...]
    duplicate_pages: tuple[int, ...]
    hit_rate: float
    recall: float
    precision: float
    reciprocal_rank: float


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    """
    Aggregated retrieval evaluation results for a dataset.
    """

    results: tuple[RetrievalEvaluationResult, ...]
    hit_rate: float
    recall: float
    precision: float
    mrr: float


def evaluate_question(
    item: EvaluationQuestion,
    limit: int = 5,
) -> RetrievalEvaluationResult:
    """
    Evaluate retrieval for one evaluation question.

    The actual retrieval is performed by the existing search()
    function. Metric calculations are delegated to metrics.py.
    """

    results = search(
        item.question,
        limit=limit,
    )

    top_results = results[:limit]

    retrieved_pages = tuple(
        result.page_number
        for result in top_results
    )

    expected_pages = set(item.relevant_pages)
    retrieved_page_set = set(retrieved_pages)

    missing_pages = tuple(
        sorted(expected_pages - retrieved_page_set)
    )

    irrelevant_pages = tuple(
        page
        for page in retrieved_pages
        if page not in expected_pages
    )

    seen_pages: set[int] = set()
    duplicate_pages_list: list[int] = []

    for page in retrieved_pages:
        if page in seen_pages and page not in duplicate_pages_list:
            duplicate_pages_list.append(page)

        seen_pages.add(page)

    duplicate_pages = tuple(duplicate_pages_list)

    return RetrievalEvaluationResult(
        question=item.question,
        expected_pages=item.relevant_pages,
        retrieved_pages=retrieved_pages,
        missing_pages=missing_pages,
        irrelevant_pages=irrelevant_pages,
        duplicate_pages=duplicate_pages,
        hit_rate=hit_rate_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=item.relevant_pages,
            k=limit,
        ),
        recall=recall_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=item.relevant_pages,
            k=limit,
        ),
        precision=precision_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=item.relevant_pages,
            k=limit,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_pages=retrieved_pages,
            relevant_pages=item.relevant_pages,
        ),
    )


def evaluate_dataset(
    dataset: list[EvaluationQuestion],
    limit: int = 5,
) -> RetrievalEvaluationSummary:
    """
    Evaluate retrieval across an entire evaluation dataset.

    Returns both per-question results and aggregated metrics.
    """

    if not dataset:
        return RetrievalEvaluationSummary(
            results=(),
            hit_rate=0.0,
            recall=0.0,
            precision=0.0,
            mrr=0.0,
        )

    results = tuple(
        evaluate_question(
            item,
            limit=limit,
        )
        for item in dataset
    )

    rankings = [
        result.retrieved_pages
        for result in results
    ]

    relevant_pages = [
        result.expected_pages
        for result in results
    ]

    hit_rate = (
        sum(result.hit_rate for result in results)
        / len(results)
    )

    recall = (
        sum(result.recall for result in results)
        / len(results)
    )

    precision = (
        sum(result.precision for result in results)
        / len(results)
    )

    mrr = mean_reciprocal_rank(
        rankings=rankings,
        relevant_pages=relevant_pages,
    )

    return RetrievalEvaluationSummary(
        results=results,
        hit_rate=hit_rate,
        recall=recall,
        precision=precision,
        mrr=mrr,
    )
