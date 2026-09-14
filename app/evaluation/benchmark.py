from dataclasses import dataclass

from app.evaluation.answer_quality import (
    AnswerQualityEvaluationResult,
    create_answer_quality_result,
)
from app.evaluation.dataset import EvaluationQuestion
from app.evaluation.generation import (
    GenerationEvaluationResult,
    evaluate_generation,
)
from app.evaluation.judge import (
    judge_correctness,
    judge_groundedness,
)
from app.evaluation.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.generation.answer import answer_from_results
from app.generation.context import build_context
from app.retrieval.search import SearchResult, search


@dataclass(frozen=True)
class BenchmarkQuestionResult:
    """
    Combined retrieval, generation, and answer-quality evaluation
    for one question.
    """

    question: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    retrieval_hit_rate: float
    retrieval_recall: float
    retrieval_precision: float
    retrieval_reciprocal_rank: float
    generated_answer: str | None
    generation: GenerationEvaluationResult | None
    answer_quality: AnswerQualityEvaluationResult | None
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
    relevant_citation_rate: float
    abstention_rate: float
    valid_citation_rate: float
    unsupported_citation_rate: float

    correctness_score: float
    groundedness_score: float
    citation_correctness_score: float


def _retrieved_pages(
    results: list[SearchResult],
) -> tuple[int, ...]:
    return tuple(
        result.page_number
        for result in results
    )


def _citation_correctness_score(
    generation: GenerationEvaluationResult,
) -> float:
    """
    Deterministic citation-correctness score for now.

    A citation is considered correct at this stage when:
    - answer has at least one citation
    - citation IDs are valid
    - cited pages were actually retrieved

    Semantic claim-to-evidence matching is left for a later layer.
    """
    if not generation.has_citations:
        return 0.0

    if not generation.citations_valid:
        return 0.0

    if generation.unsupported_citation_pages:
        return 0.0

    return 1.0


def _evaluate_answer_quality(
    item: EvaluationQuestion,
    answer_text: str,
    results: list[SearchResult],
    generation: GenerationEvaluationResult,
) -> AnswerQualityEvaluationResult | None:
    if not item.reference_answer.strip():
        return None

    context = build_context(results)

    correctness = judge_correctness(
        question=item.question,
        reference_answer=item.reference_answer,
        generated_answer=answer_text,
    )

    groundedness = judge_groundedness(
        question=item.question,
        retrieved_context=context.text,
        generated_answer=answer_text,
    )

    return create_answer_quality_result(
        correctness_score=correctness.score,
        correctness_reason=correctness.reason,
        groundedness_score=groundedness.score,
        groundedness_reason=groundedness.reason,
        citation_correctness_score=_citation_correctness_score(
            generation
        ),
    )


def evaluate_question(
    item: EvaluationQuestion,
    limit: int = 5,
) -> BenchmarkQuestionResult:
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

        answer_quality = _evaluate_answer_quality(
            item=item,
            answer_text=answer.text,
            results=results,
            generation=generation,
        )

        return BenchmarkQuestionResult(
            question=item.question,
            expected_pages=item.relevant_pages,
            retrieved_pages=retrieved_pages,
            retrieval_hit_rate=retrieval_hit_rate,
            retrieval_recall=retrieval_recall,
            retrieval_precision=retrieval_precision,
            retrieval_reciprocal_rank=retrieval_reciprocal_rank,
            generated_answer=answer.text,
            generation=generation,
            answer_quality=answer_quality,
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
            generated_answer=None,
            generation=None,
            answer_quality=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_dataset(
    dataset: list[EvaluationQuestion],
    limit: int = 5,
) -> BenchmarkSummary:
    if not dataset:
        return BenchmarkSummary(
            results=(),
            retrieval_hit_rate=0.0,
            retrieval_recall=0.0,
            retrieval_precision=0.0,
            retrieval_mrr=0.0,
            answer_rate=0.0,
            citation_rate=0.0,
            relevant_citation_rate=0.0,
            abstention_rate=0.0,
            valid_citation_rate=0.0,
            unsupported_citation_rate=0.0,
            correctness_score=0.0,
            groundedness_score=0.0,
            citation_correctness_score=0.0,
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
        sum(result.retrieval_hit_rate for result in results) / total
    )

    retrieval_recall = (
        sum(result.retrieval_recall for result in results) / total
    )

    retrieval_precision = (
        sum(result.retrieval_precision for result in results) / total
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

    relevant_citation_count = sum(
        bool(generation.cited_relevant_pages)
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

    quality_results = [
        result.answer_quality
        for result in results
        if result.answer_quality is not None
    ]

    if quality_results:
        correctness_score = (
            sum(
                result.correctness_score
                for result in quality_results
            )
            / len(quality_results)
        )

        groundedness_score = (
            sum(
                result.groundedness_score
                for result in quality_results
            )
            / len(quality_results)
        )

        citation_correctness_score = (
            sum(
                result.citation_correctness_score
                for result in quality_results
            )
            / len(quality_results)
        )

    else:
        correctness_score = 0.0
        groundedness_score = 0.0
        citation_correctness_score = 0.0

    return BenchmarkSummary(
        results=results,
        retrieval_hit_rate=retrieval_hit_rate,
        retrieval_recall=retrieval_recall,
        retrieval_precision=retrieval_precision,
        retrieval_mrr=retrieval_mrr,
        answer_rate=answer_count / total,
        citation_rate=citation_count / total,
        relevant_citation_rate=relevant_citation_count / total,
        abstention_rate=abstention_count / total,
        valid_citation_rate=valid_citation_count / total,
        unsupported_citation_rate=unsupported_citation_count / total,
        correctness_score=correctness_score,
        groundedness_score=groundedness_score,
        citation_correctness_score=citation_correctness_score,
    )