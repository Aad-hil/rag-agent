from dataclasses import dataclass

from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.retrieval.search import search


BASELINE_COLLECTION = "documents"
EXPERIMENT_COLLECTION = "documents_chunking_800_100"

LIMIT = 5


@dataclass(frozen=True)
class RetrievalMetrics:
    """
    Retrieval metrics for one question.
    """

    retrieved_pages: tuple[int, ...]
    hit_rate: float
    recall: float
    precision: float
    reciprocal_rank: float


@dataclass(frozen=True)
class ComparisonResult:
    """
    Comparison between baseline and experimental chunking
    for one evaluation question.
    """

    question: str
    expected_pages: tuple[int, ...]

    baseline: RetrievalMetrics
    experiment: RetrievalMetrics


def evaluate_collection(
    question: str,
    relevant_pages: tuple[int, ...],
    collection_name: str,
) -> RetrievalMetrics:
    """
    Retrieve and evaluate one question against one Qdrant collection.
    """

    results = search(
        query=question,
        limit=LIMIT,
        collection_name=collection_name,
    )

    retrieved_pages = tuple(
        result.page_number
        for result in results
    )

    return RetrievalMetrics(
        retrieved_pages=retrieved_pages,
        hit_rate=hit_rate_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=LIMIT,
        ),
        recall=recall_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=LIMIT,
        ),
        precision=precision_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=LIMIT,
        ),
        reciprocal_rank=reciprocal_rank(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
        ),
    )


def compare_question(
    question: str,
    expected_pages: tuple[int, ...],
) -> ComparisonResult:
    """
    Compare one question across both chunking configurations.
    """

    baseline = evaluate_collection(
        question=question,
        relevant_pages=expected_pages,
        collection_name=BASELINE_COLLECTION,
    )

    experiment = evaluate_collection(
        question=question,
        relevant_pages=expected_pages,
        collection_name=EXPERIMENT_COLLECTION,
    )

    return ComparisonResult(
        question=question,
        expected_pages=expected_pages,
        baseline=baseline,
        experiment=experiment,
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
    Format the difference between experiment and baseline
    in percentage points.
    """

    change = (experiment - baseline) * 100

    return f"{change:+.2f} pp"


def run_experiment() -> None:
    """
    Run the complete chunking comparison benchmark.
    """

    comparisons = [
        compare_question(
            question=item.question,
            expected_pages=item.relevant_pages,
        )
        for item in EVALUATION_DATASET
    ]

    print("=" * 80)
    print("CHUNKING EXPERIMENT")
    print("=" * 80)
    print()
    print("Baseline:  1000 chars / 150 overlap")
    print("Experiment: 800 chars / 100 overlap")
    print()

    for index, comparison in enumerate(
        comparisons,
        start=1,
    ):
        baseline = comparison.baseline
        experiment = comparison.experiment

        print("=" * 80)
        print(f"Q{index:02d}")
        print("=" * 80)

        print(f"Question: {comparison.question}")
        print(
            f"Expected pages: "
            f"{_format_pages(comparison.expected_pages)}"
        )

        print()
        print(
            f"Baseline → "
            f"{_format_pages(baseline.retrieved_pages)}"
        )

        print(
            f"800/100   → "
            f"{_format_pages(experiment.retrieved_pages)}"
        )

        print()
        print(
            f"Recall@5: "
            f"{baseline.recall:.2%} → "
            f"{experiment.recall:.2%} "
            f"({_format_change(baseline.recall, experiment.recall)})"
        )

        print(
            f"Precision@5: "
            f"{baseline.precision:.2%} → "
            f"{experiment.precision:.2%} "
            f"({_format_change(baseline.precision, experiment.precision)})"
        )

        print(
            f"RR: "
            f"{baseline.reciprocal_rank:.2%} → "
            f"{experiment.reciprocal_rank:.2%} "
            f"({_format_change(baseline.reciprocal_rank, experiment.reciprocal_rank)})"
        )

    baseline_hit_rate = _average(
        [
            comparison.baseline.hit_rate
            for comparison in comparisons
        ]
    )

    experiment_hit_rate = _average(
        [
            comparison.experiment.hit_rate
            for comparison in comparisons
        ]
    )

    baseline_recall = _average(
        [
            comparison.baseline.recall
            for comparison in comparisons
        ]
    )

    experiment_recall = _average(
        [
            comparison.experiment.recall
            for comparison in comparisons
        ]
    )

    baseline_precision = _average(
        [
            comparison.baseline.precision
            for comparison in comparisons
        ]
    )

    experiment_precision = _average(
        [
            comparison.experiment.precision
            for comparison in comparisons
        ]
    )

    baseline_mrr = _average(
        [
            comparison.baseline.reciprocal_rank
            for comparison in comparisons
        ]
    )

    experiment_mrr = _average(
        [
            comparison.experiment.reciprocal_rank
            for comparison in comparisons
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
        f"{'800/100':>12}"
        f"{'Change':>15}"
    )

    print("-" * 80)

    print(
        f"{'Hit Rate@5':<20}"
        f"{baseline_hit_rate:>11.2%}"
        f"{experiment_hit_rate:>12.2%}"
        f"{_format_change(baseline_hit_rate, experiment_hit_rate):>15}"
    )

    print(
        f"{'Recall@5':<20}"
        f"{baseline_recall:>11.2%}"
        f"{experiment_recall:>12.2%}"
        f"{_format_change(baseline_recall, experiment_recall):>15}"
    )

    print(
        f"{'Precision@5':<20}"
        f"{baseline_precision:>11.2%}"
        f"{experiment_precision:>12.2%}"
        f"{_format_change(baseline_precision, experiment_precision):>15}"
    )

    print(
        f"{'MRR@5':<20}"
        f"{baseline_mrr:>11.2%}"
        f"{experiment_mrr:>12.2%}"
        f"{_format_change(baseline_mrr, experiment_mrr):>15}"
    )


if __name__ == "__main__":
    run_experiment()