from app.evaluation.benchmark import (
    BenchmarkQuestionResult,
    BenchmarkSummary,
    evaluate_dataset,
)
from app.evaluation.dataset import EVALUATION_DATASET


def _format_pages(pages: tuple[int, ...]) -> str:
    """
    Format page numbers for human-readable output.
    """
    return str(list(pages))


def _question_status(
    result: BenchmarkQuestionResult,
) -> str:
    """
    Determine the human-readable status for one benchmark question.

    ERROR means the generation pipeline raised an exception.

    PASS means the system produced an answer with at least one
    citation to an expected relevant page.

    FAIL means the system produced an answer but did not satisfy
    the citation-quality requirement.
    """

    if result.error is not None:
        return "ERROR"

    if result.generation is None:
        return "ERROR"

    if (
        result.generation.has_answer
        and result.generation.has_citations
        and result.generation.cited_relevant_pages
    ):
        return "PASS"

    return "FAIL"


def _print_question_result(
    index: int,
    result: BenchmarkQuestionResult,
    limit: int,
) -> None:
    """
    Print the evaluation result for one question.
    """

    status = _question_status(result)

    print(f"Q{index:02d} [{status}]")
    print(f"Question: {result.question}")
    print(
        f"Expected pages: "
        f"{_format_pages(result.expected_pages)}"
    )
    print(
        f"Retrieved pages: "
        f"{_format_pages(result.retrieved_pages)}"
    )

    print(
        f"Retrieval → "
        f"Hit@{limit}: {result.retrieval_hit_rate:.2f} | "
        f"Recall@{limit}: {result.retrieval_recall:.2f} | "
        f"P@{limit}: {result.retrieval_precision:.2f} | "
        f"RR: {result.retrieval_reciprocal_rank:.2f}"
    )

    if result.generation is None:
        print("Generation → ERROR")

        if result.error is not None:
            print(f"    {result.error}")

        print()
        return

    generation = result.generation

    print(
        f"Generation → "
        f"Answer: {'YES' if generation.has_answer else 'NO'} | "
        f"Abstention: {'YES' if generation.is_abstention else 'NO'} | "
        f"Citations: {'YES' if generation.has_citations else 'NO'}"
    )

    print(
        f"Cited pages: "
        f"{_format_pages(generation.citation_pages)}"
    )

    print(
        f"Relevant cited pages: "
        f"{_format_pages(generation.cited_relevant_pages)}"
    )

    if generation.unsupported_citation_pages:
        print(
            f"Unsupported cited pages: "
            f"{_format_pages(generation.unsupported_citation_pages)}"
        )

    print()


def _print_summary(
    summary: BenchmarkSummary,
    limit: int,
) -> None:
    """
    Print aggregate benchmark metrics.
    """

    total = len(summary.results)

    errors = sum(
        result.error is not None
        for result in summary.results
    )

    passed = sum(
        _question_status(result) == "PASS"
        for result in summary.results
    )

    failed = sum(
        _question_status(result) == "FAIL"
        for result in summary.results
    )

    print("=" * 60)
    print("RETRIEVAL")
    print("=" * 60)
    print(
        f"Hit Rate@{limit}: "
        f"{summary.retrieval_hit_rate:.2%}"
    )
    print(
        f"Recall@{limit}:   "
        f"{summary.retrieval_recall:.2%}"
    )
    print(
        f"Precision@{limit}: "
        f"{summary.retrieval_precision:.2%}"
    )
    print(
        f"MRR@{limit}:       "
        f"{summary.retrieval_mrr:.2%}"
    )

    print()
    print("=" * 60)
    print("GENERATION")
    print("=" * 60)
    print(
        f"Answer Rate:              "
        f"{summary.answer_rate:.2%}"
    )
    print(
        f"Citation Rate:            "
        f"{summary.citation_rate:.2%}"
    )
    print(
        f"Relevant Citation Rate:   "
        f"{summary.relevant_citation_rate:.2%}"
    )
    print(
        f"Abstention Rate:          "
        f"{summary.abstention_rate:.2%}"
    )
    print(
        f"Valid Citation Rate:      "
        f"{summary.valid_citation_rate:.2%}"
    )
    print(
        f"Unsupported Citation Rate:"
        f" {summary.unsupported_citation_rate:.2%}"
    )

    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Questions: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Generation errors: {errors}")
    print("=" * 60)


def evaluate(
    limit: int = 5,
) -> BenchmarkSummary:
    """
    Run and print the complete RAG evaluation benchmark.

    Returns the BenchmarkSummary so callers can also use the
    structured results programmatically.
    """

    summary = evaluate_dataset(
        dataset=EVALUATION_DATASET,
        limit=limit,
    )

    print()
    print("=" * 60)
    print("RAG AGENT EVALUATION")
    print("=" * 60)
    print()

    for index, result in enumerate(
        summary.results,
        start=1,
    ):
        _print_question_result(
            index=index,
            result=result,
            limit=limit,
        )

    _print_summary(
        summary=summary,
        limit=limit,
    )

    return summary


if __name__ == "__main__":
    evaluate()