from dataclasses import dataclass

from app.agent.rewrite import rewrite_query
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.search import SearchResult, search


QUERY_LIMIT = 10
FINAL_LIMIT = 5
RRF_K = 60


@dataclass(frozen=True)
class RankedCandidate:
    """
    A retrieved candidate with its RRF score.
    """

    result: SearchResult
    rrf_score: float


@dataclass(frozen=True)
class MultiQueryMetrics:
    """
    Retrieval metrics for one multi-query question.
    """

    retrieved_pages: tuple[int, ...]
    hit_rate: float
    recall: float
    precision: float
    reciprocal_rank: float


def reciprocal_rank_fusion(
    rankings: list[list[SearchResult]],
    k: int = RRF_K,
) -> list[RankedCandidate]:
    """
    Merge multiple ranked retrieval lists using Reciprocal Rank Fusion.

    Each unique chunk receives a score based on its rank in every
    retrieval list where it appears.
    """

    scores: dict[int, float] = {}
    results_by_chunk: dict[int, SearchResult] = {}

    for ranking in rankings:
        for rank, result in enumerate(
            ranking,
            start=1,
        ):
            chunk_index = result.chunk_index

            scores[chunk_index] = (
                scores.get(chunk_index, 0.0)
                + 1.0 / (k + rank)
            )

            results_by_chunk[chunk_index] = result

    candidates = [
        RankedCandidate(
            result=results_by_chunk[chunk_index],
            rrf_score=score,
        )
        for chunk_index, score in scores.items()
    ]

    candidates.sort(
        key=lambda candidate: candidate.rrf_score,
        reverse=True,
    )

    return candidates


def retrieve_multi_query(
    question: str,
) -> tuple[str, list[RankedCandidate]]:
    """
    Generate three rewritten queries, retrieve candidates for each
    query plus the original query, and merge them using RRF.
    """

    rewritten_queries = [
        rewrite_query(question)
        for _ in range(3)
    ]

    queries = [
        question,
        *rewritten_queries,
    ]

    rankings = [
        search(
            query,
            limit=QUERY_LIMIT,
        )
        for query in queries
    ]

    fused_candidates = reciprocal_rank_fusion(
        rankings=rankings,
    )

    return rewritten_queries[0], fused_candidates


def evaluate_question(
    question: str,
    relevant_pages: tuple[int, ...],
) -> MultiQueryMetrics:
    """
    Evaluate multi-query retrieval for one question.
    """

    _, candidates = retrieve_multi_query(
        question
    )

    top_results = [
        candidate.result
        for candidate in candidates[:FINAL_LIMIT]
    ]

    retrieved_pages = tuple(
        result.page_number
        for result in top_results
    )

    return MultiQueryMetrics(
        retrieved_pages=retrieved_pages,
        hit_rate=hit_rate_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=FINAL_LIMIT,
        ),
        recall=recall_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=FINAL_LIMIT,
        ),
        precision=precision_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=FINAL_LIMIT,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
        ),
    )


def _average(
    values: list[float],
) -> float:
    """
    Return the arithmetic mean of a non-empty list.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)


def _format_pages(
    pages: tuple[int, ...],
) -> str:
    """
    Format page numbers for readable output.
    """

    return str(list(pages))


def _format_change(
    baseline: float,
    experiment: float,
) -> str:
    """
    Format a metric difference in percentage points.
    """

    change = (experiment - baseline) * 100

    return f"{change:+.2f} pp"


def run_experiment() -> None:
    """
    Compare baseline single-query retrieval against multi-query
    retrieval across the evaluation dataset.
    """

    baseline_metrics = []
    multi_query_metrics = []

    print("=" * 80)
    print("MULTI-QUERY RETRIEVAL EXPERIMENT")
    print("=" * 80)
    print()
    print(
        f"Queries per question: 4 "
        f"(1 original + 3 rewrites)"
    )
    print(
        f"Candidates per query: {QUERY_LIMIT}"
    )
    print(
        f"Final results: Top-{FINAL_LIMIT}"
    )
    print(
        f"RRF constant: {RRF_K}"
    )
    print()

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):
        original_results = search(
            item.question,
            limit=FINAL_LIMIT,
        )

        baseline_pages = tuple(
            result.page_number
            for result in original_results
        )

        baseline = MultiQueryMetrics(
            retrieved_pages=baseline_pages,
            hit_rate=hit_rate_at_k(
                retrieved_pages=baseline_pages,
                relevant_pages=item.relevant_pages,
                k=FINAL_LIMIT,
            ),
            recall=recall_at_k(
                retrieved_pages=baseline_pages,
                relevant_pages=item.relevant_pages,
                k=FINAL_LIMIT,
            ),
            precision=precision_at_k(
                retrieved_pages=baseline_pages,
                relevant_pages=item.relevant_pages,
                k=FINAL_LIMIT,
            ),
            reciprocal_rank=reciprocal_rank(
                retrieved_pages=baseline_pages,
                relevant_pages=item.relevant_pages,
            ),
        )

        multi_query = evaluate_question(
            question=item.question,
            relevant_pages=item.relevant_pages,
        )

        baseline_metrics.append(baseline)
        multi_query_metrics.append(multi_query)

        print("=" * 80)
        print(f"Q{index:02d}")
        print("=" * 80)

        print(
            f"Question: {item.question}"
        )

        print(
            f"Expected: "
            f"{list(item.relevant_pages)}"
        )

        print()

        print(
            f"Baseline: "
            f"{_format_pages(baseline.retrieved_pages)}"
        )

        print(
            f"Multi-query: "
            f"{_format_pages(multi_query.retrieved_pages)}"
        )

        print()

        print(
            f"Recall@5: "
            f"{baseline.recall:.2%} → "
            f"{multi_query.recall:.2%} "
            f"({_format_change(baseline.recall, multi_query.recall)})"
        )

        print(
            f"Precision@5: "
            f"{baseline.precision:.2%} → "
            f"{multi_query.precision:.2%} "
            f"({_format_change(baseline.precision, multi_query.precision)})"
        )

        print(
            f"RR: "
            f"{baseline.reciprocal_rank:.2%} → "
            f"{multi_query.reciprocal_rank:.2%} "
            f"({_format_change(baseline.reciprocal_rank, multi_query.reciprocal_rank)})"
        )

    baseline_hit_rate = _average(
        [
            metric.hit_rate
            for metric in baseline_metrics
        ]
    )

    multi_query_hit_rate = _average(
        [
            metric.hit_rate
            for metric in multi_query_metrics
        ]
    )

    baseline_recall = _average(
        [
            metric.recall
            for metric in baseline_metrics
        ]
    )

    multi_query_recall = _average(
        [
            metric.recall
            for metric in multi_query_metrics
        ]
    )

    baseline_precision = _average(
        [
            metric.precision
            for metric in baseline_metrics
        ]
    )

    multi_query_precision = _average(
        [
            metric.precision
            for metric in multi_query_metrics
        ]
    )

    baseline_mrr = _average(
        [
            metric.reciprocal_rank
            for metric in baseline_metrics
        ]
    )

    multi_query_mrr = _average(
        [
            metric.reciprocal_rank
            for metric in multi_query_metrics
        ]
    )

    print()
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    print()
    print(
        f"{'Metric':<20}"
        f"{'Baseline':>12}"
        f"{'Multi-query':>15}"
        f"{'Change':>15}"
    )

    print("-" * 80)

    print(
        f"{'Hit Rate@5':<20}"
        f"{baseline_hit_rate:>11.2%}"
        f"{multi_query_hit_rate:>15.2%}"
        f"{_format_change(baseline_hit_rate, multi_query_hit_rate):>15}"
    )

    print(
        f"{'Recall@5':<20}"
        f"{baseline_recall:>11.2%}"
        f"{multi_query_recall:>15.2%}"
        f"{_format_change(baseline_recall, multi_query_recall):>15}"
    )

    print(
        f"{'Precision@5':<20}"
        f"{baseline_precision:>11.2%}"
        f"{multi_query_precision:>15.2%}"
        f"{_format_change(baseline_precision, multi_query_precision):>15}"
    )

    print(
        f"{'MRR@5':<20}"
        f"{baseline_mrr:>11.2%}"
        f"{multi_query_mrr:>15.2%}"
        f"{_format_change(baseline_mrr, multi_query_mrr):>15}"
    )


if __name__ == "__main__":
    run_experiment()