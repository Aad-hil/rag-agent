from dataclasses import dataclass

from app.evaluation.dataset import EvaluationQuestion, EVALUATION_DATASET
from app.evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.reranker import rerank
from app.retrieval.search import search


CANDIDATE_LIMIT = 20
FINAL_LIMIT = 5


@dataclass(frozen=True)
class RerankerQuestionResult:
    """
    Retrieval metrics for one question after cross-encoder reranking.
    """

    question: str
    expected_pages: tuple[int, ...]
    dense_pages: tuple[int, ...]
    reranked_pages: tuple[int, ...]

    dense_hit_rate: float
    dense_recall: float
    dense_precision: float
    dense_reciprocal_rank: float

    reranked_hit_rate: float
    reranked_recall: float
    reranked_precision: float
    reranked_reciprocal_rank: float


@dataclass(frozen=True)
class RerankerBenchmarkSummary:
    """
    Aggregate comparison between dense retrieval and reranked retrieval.
    """

    results: tuple[RerankerQuestionResult, ...]

    dense_hit_rate: float
    dense_recall: float
    dense_precision: float
    dense_mrr: float

    reranked_hit_rate: float
    reranked_recall: float
    reranked_precision: float
    reranked_mrr: float


def _pages_from_results(results) -> tuple[int, ...]:
    """
    Return page numbers in ranked result order.
    """

    return tuple(
        result.page_number
        for result in results
    )


def evaluate_question(
    item: EvaluationQuestion,
    candidate_limit: int = CANDIDATE_LIMIT,
    final_limit: int = FINAL_LIMIT,
) -> RerankerQuestionResult:
    """
    Compare dense retrieval Top-K against cross-encoder reranked Top-K.

    Dense retrieval produces the candidate pool. The reranker then
    selects the final results from that same pool.
    """

    candidates = search(
        item.question,
        limit=candidate_limit,
    )

    dense_top_k = candidates[:final_limit]

    reranked = rerank(
        query=item.question,
        results=candidates,
        limit=final_limit,
    )

    dense_pages = _pages_from_results(
        dense_top_k,
    )

    reranked_pages = tuple(
        reranked_result.result.page_number
        for reranked_result in reranked
    )

    return RerankerQuestionResult(
        question=item.question,
        expected_pages=item.relevant_pages,
        dense_pages=dense_pages,
        reranked_pages=reranked_pages,
        dense_hit_rate=hit_rate_at_k(
            retrieved_pages=dense_pages,
            relevant_pages=item.relevant_pages,
            k=final_limit,
        ),
        dense_recall=recall_at_k(
            retrieved_pages=dense_pages,
            relevant_pages=item.relevant_pages,
            k=final_limit,
        ),
        dense_precision=precision_at_k(
            retrieved_pages=dense_pages,
            relevant_pages=item.relevant_pages,
            k=final_limit,
        ),
        dense_reciprocal_rank=reciprocal_rank(
            retrieved_pages=dense_pages,
            relevant_pages=item.relevant_pages,
        ),
        reranked_hit_rate=hit_rate_at_k(
            retrieved_pages=reranked_pages,
            relevant_pages=item.relevant_pages,
            k=final_limit,
        ),
        reranked_recall=recall_at_k(
            retrieved_pages=reranked_pages,
            relevant_pages=item.relevant_pages,
            k=final_limit,
        ),
        reranked_precision=precision_at_k(
            retrieved_pages=reranked_pages,
            relevant_pages=item.relevant_pages,
            k=final_limit,
        ),
        reranked_reciprocal_rank=reciprocal_rank(
            retrieved_pages=reranked_pages,
            relevant_pages=item.relevant_pages,
        ),
    )


def evaluate_dataset(
    dataset: list[EvaluationQuestion],
    candidate_limit: int = CANDIDATE_LIMIT,
    final_limit: int = FINAL_LIMIT,
) -> RerankerBenchmarkSummary:
    """
    Run the reranker comparison across the complete dataset.
    """

    if not dataset:
        return RerankerBenchmarkSummary(
            results=(),
            dense_hit_rate=0.0,
            dense_recall=0.0,
            dense_precision=0.0,
            dense_mrr=0.0,
            reranked_hit_rate=0.0,
            reranked_recall=0.0,
            reranked_precision=0.0,
            reranked_mrr=0.0,
        )

    results = tuple(
        evaluate_question(
            item=item,
            candidate_limit=candidate_limit,
            final_limit=final_limit,
        )
        for item in dataset
    )

    total = len(results)

    return RerankerBenchmarkSummary(
        results=results,

        dense_hit_rate=(
            sum(result.dense_hit_rate for result in results)
            / total
        ),
        dense_recall=(
            sum(result.dense_recall for result in results)
            / total
        ),
        dense_precision=(
            sum(result.dense_precision for result in results)
            / total
        ),
        dense_mrr=(
            sum(result.dense_reciprocal_rank for result in results)
            / total
        ),

        reranked_hit_rate=(
            sum(result.reranked_hit_rate for result in results)
            / total
        ),
        reranked_recall=(
            sum(result.reranked_recall for result in results)
            / total
        ),
        reranked_precision=(
            sum(result.reranked_precision for result in results)
            / total
        ),
        reranked_mrr=(
            sum(result.reranked_reciprocal_rank for result in results)
            / total
        ),
    )


def _percentage_change(
    baseline: float,
    experiment: float,
) -> float:
    """
    Return the absolute percentage-point change.
    """

    return (experiment - baseline) * 100


def print_summary(
    summary: RerankerBenchmarkSummary,
) -> None:
    """
    Print the dense-vs-reranked benchmark comparison.
    """

    print()
    print("=" * 70)
    print("DENSE VS CROSS-ENCODER RERANKING")
    print("=" * 70)

    print()
    print(
        f"{'Metric':<20}"
        f"{'Dense':>12}"
        f"{'Reranked':>14}"
        f"{'Change':>14}"
    )
    print("-" * 70)

    metrics = [
        (
            "Hit Rate@5",
            summary.dense_hit_rate,
            summary.reranked_hit_rate,
        ),
        (
            "Recall@5",
            summary.dense_recall,
            summary.reranked_recall,
        ),
        (
            "Precision@5",
            summary.dense_precision,
            summary.reranked_precision,
        ),
        (
            "MRR@5",
            summary.dense_mrr,
            summary.reranked_mrr,
        ),
    ]

    for name, dense, reranked in metrics:
        change = _percentage_change(
            dense,
            reranked,
        )

        print(
            f"{name:<20}"
            f"{dense:>11.2%}"
            f"{reranked:>13.2%}"
            f"{change:>+13.2f} pp"
        )

    print()
    print("=" * 70)
    print("QUESTION-LEVEL COMPARISON")
    print("=" * 70)

    for index, result in enumerate(
        summary.results,
        start=1,
    ):
        print()
        print(f"Q{index:02d}")
        print(
            f"Expected:  {list(result.expected_pages)}"
        )
        print(
            f"Dense:     {list(result.dense_pages)}"
        )
        print(
            f"Reranked:  {list(result.reranked_pages)}"
        )

        print(
            f"Recall:    "
            f"{result.dense_recall:.2%}"
            f" → "
            f"{result.reranked_recall:.2%}"
        )

        print(
            f"Precision: "
            f"{result.dense_precision:.2%}"
            f" → "
            f"{result.reranked_precision:.2%}"
        )

        print(
            f"RR:        "
            f"{result.dense_reciprocal_rank:.2%}"
            f" → "
            f"{result.reranked_reciprocal_rank:.2%}"
        )


def run_benchmark() -> RerankerBenchmarkSummary:
    """
    Run the complete reranker benchmark and print the results.
    """

    print("=" * 70)
    print("RERANKER BENCHMARK")
    print("=" * 70)
    print(
        f"Candidate pool: Top-{CANDIDATE_LIMIT}"
    )
    print(
        f"Final results: Top-{FINAL_LIMIT}"
    )

    summary = evaluate_dataset(
        dataset=EVALUATION_DATASET,
        candidate_limit=CANDIDATE_LIMIT,
        final_limit=FINAL_LIMIT,
    )

    print_summary(summary)

    return summary


if __name__ == "__main__":
    run_benchmark()