from dataclasses import dataclass

from app.evaluation.dataset import EvaluationQuestion
from app.evaluation.generation import (
    GenerationEvaluationResult,
    evaluate_generation,
)
from app.evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.generation.answer import answer_from_results
from app.retrieval.search import SearchResult, search


@dataclass(frozen=True)
class BenchmarkQuestionResult:
    """
    Combined retrieval and generation evaluation for one question.
    """

    question: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    retrieval_hit_rate: float
    retrieval_recall: float
    retrieval_precision: float
    retrieval_reciprocal_rank: float
    generation: GenerationEvaluationResult | None
    error: str | None


@dataclass(frozen=True)
class BenchmarkSummary:
    """
    Aggregated evaluation results across the full dataset.
    """

    results: tuple[BenchmarkQuestionResult, ...]

    retrieval_hit_rate: float
    retrieval_recall: float
    retrieval_precision: float
    retrieval_mrr: float

    answer_rate: float
    citation_rate: float
    abstention_rate: float
    valid_citation_rate: float
    unsupported_citation_rate: float


def _retrieved_pages(
    results: list[SearchResult],
) -> tuple[int, ...]:
    """
    Return retrieved page numbers in result order.

    Duplicate pages are preserved because retrieval metrics operate
    on ranked retrieval positions.
    """
    return tuple(
        result.page_number
        for result in results
    )


def evaluate_question(
    item: EvaluationQuestion,
    limit: int = 5,
) -> BenchmarkQuestionResult:
    """
    Run retrieval and generation evaluation for one question.

    Retrieval is performed exactly once. The same retrieved results
    are then passed into the generation pipeline.
    """

    results = search(
        item.question,
        limit=limit,
    )

    retrieved_pages = _retrieved_pages(results)

    retrieval_hit_rate = hit_rate_at_k(
        retrieved_pages=retrieved_pages,
        relevant_pages=item.relevant_pages,
        k=limit,
    )

    retrieval_recall = recall_at_k(
        retrieved_pages=retrieved_pages,
        relevant_pages=item.relevant_pages,
        k=limit,
    )

    retrieval_precision = precision_at_k(
        retrieved_pages=retrieved_pages,
        relevant_pages=item.relevant_pages,
        k=limit,
    )

    retrieval_reciprocal_rank = reciprocal_rank(
        retrieved_pages=retrieved_pages,
        relevant_pages=item.relevant_pages,
    )

    try:
        answer = answer_from_results(
            question=item.question,
            results=results,
        )

        valid_citation_ids = set(
            range(1, len(results) + 1)
        )

        generation = evaluate_generation(
            answer=answer,
            relevant_pages=set(item.relevant_pages),
            valid_citation_ids=valid_citation_ids,
            retrieved_pages=retrieved_pages,
        )

        return BenchmarkQuestionResult(
            question=item.question,
            expected_pages=item.relevant_pages,
            retrieved_pages=retrieved_pages,
            retrieval_hit_rate=retrieval_hit_rate,
            retrieval_recall=retrieval_recall,
            retrieval_precision=retrieval_precision,
            retrieval_reciprocal_rank=retrieval_reciprocal_rank,
            generation=generation,
            error=None,
        )

    except Exception as exc:
        return BenchmarkQuestionResult(
            question=item.question,
            expected_pages=item.relevant_pages,
            retrieved_pages=retrieved_pages,
            retrieval_hit_rate=retrieval_hit_rate,
            retrieval_recall=retrieval_recall,
            retrieval_precision=retrieval_precision,
            retrieval_reciprocal_rank=retrieval_reciprocal_rank,
            generation=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_dataset(
    dataset: list[EvaluationQuestion],
    limit: int = 5,
) -> BenchmarkSummary:
    """
    Run the complete retrieval + generation benchmark.
    """

    if not dataset:
        return BenchmarkSummary(
            results=(),
            retrieval_hit_rate=0.0,
            retrieval_recall=0.0,
            retrieval_precision=0.0,
            retrieval_mrr=0.0,
            answer_rate=0.0,
            citation_rate=0.0,
            abstention_rate=0.0,
            valid_citation_rate=0.0,
            unsupported_citation_rate=0.0,
        )

    results = tuple(
        evaluate_question(
            item=item,
            limit=limit,
        )
        for item in dataset
    )

    total = len(results)

    retrieval_hit_rate = (
        sum(
            result.retrieval_hit_rate
            for result in results
        )
        / total
    )

    retrieval_recall = (
        sum(
            result.retrieval_recall
            for result in results
        )
        / total
    )

    retrieval_precision = (
        sum(
            result.retrieval_precision
            for result in results
        )
        / total
    )

    retrieval_mrr = (
        sum(
            result.retrieval_reciprocal_rank
            for result in results
        )
        / total
    )

    successful_generations = [
        result.generation
        for result in results
        if result.generation is not None
    ]

    answer_count = sum(
        generation.has_answer
        for generation in successful_generations
    )

    citation_count = sum(
        generation.has_citations
        for generation in successful_generations
    )

    abstention_count = sum(
        generation.is_abstention
        for generation in successful_generations
    )

    valid_citation_count = sum(
        generation.citations_valid
        for generation in successful_generations
    )

    unsupported_citation_count = sum(
        bool(generation.unsupported_citation_pages)
        for generation in successful_generations
    )

    return BenchmarkSummary(
        results=results,
        retrieval_hit_rate=retrieval_hit_rate,
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
        retrieval_mrr=retrieval_mrr,
        answer_rate=answer_count / total,
        citation_rate=citation_count / total,
        abstention_rate=abstention_count / total,
        valid_citation_rate=valid_citation_count / total,
        unsupported_citation_rate=unsupported_citation_count / total,
    )