from collections import Counter
from dataclasses import dataclass

from app.evaluation.dataset import EvaluationQuestion
from app.retrieval.search import search


CANDIDATE_LIMIT = 20


@dataclass(frozen=True)
class CandidateRecallDiagnostic:
    """
    Candidate-retrieval diagnostics for one evaluation question.
    """

    question: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    relevant_page_ranks: dict[int, tuple[int, ...]]
    missing_pages: tuple[int, ...]
    unique_page_count: int
    unique_chunk_count: int


def diagnose_question(
    item: EvaluationQuestion,
    limit: int = CANDIDATE_LIMIT,
) -> CandidateRecallDiagnostic:
    """
    Retrieve a larger candidate set and determine where expected
    relevant pages appear in the ranking.

    This function does not modify production retrieval behavior.
    """

    results = search(
        item.question,
        limit=limit,
    )

    retrieved_pages = tuple(
        result.page_number
        for result in results
    )

    retrieved_chunks = tuple(
        result.chunk_index
        for result in results
    )

    relevant_page_ranks: dict[int, list[int]] = {
        page: []
        for page in item.relevant_pages
    }

    for rank, result in enumerate(results, start=1):
        if result.page_number in relevant_page_ranks:
            relevant_page_ranks[result.page_number].append(rank)

    missing_pages = tuple(
        page
        for page in item.relevant_pages
        if not relevant_page_ranks[page]
    )

    return CandidateRecallDiagnostic(
        question=item.question,
        expected_pages=item.relevant_pages,
        retrieved_pages=retrieved_pages,
        relevant_page_ranks={
            page: tuple(ranks)
            for page, ranks in relevant_page_ranks.items()
        },
        missing_pages=missing_pages,
        unique_page_count=len(set(retrieved_pages)),
        unique_chunk_count=len(set(retrieved_chunks)),
    )


def _format_ranks(
    ranks: tuple[int, ...],
) -> str:
    """
    Format ranking positions for display.
    """

    if not ranks:
        return "NOT FOUND"

    return ", ".join(
        str(rank)
        for rank in ranks
    )


def _recall_at_k(
    diagnostic: CandidateRecallDiagnostic,
    k: int,
) -> float:
    """
    Calculate page-level recall at K for the diagnostic.
    """

    if not diagnostic.expected_pages:
        return 0.0

    found = sum(
        any(rank <= k for rank in ranks)
        for ranks in diagnostic.relevant_page_ranks.values()
    )

    return found / len(diagnostic.expected_pages)


def print_diagnostic(
    diagnostic: CandidateRecallDiagnostic,
) -> None:
    """
    Print candidate recall information for one question.
    """

    print("=" * 70)
    print("CANDIDATE RECALL DIAGNOSTIC")
    print("=" * 70)

    print(
        f"Question: {diagnostic.question}"
    )

    print(
        f"Expected pages: "
        f"{list(diagnostic.expected_pages)}"
    )

    print()
    print("Relevant page positions:")
    print("-" * 70)

    for page in diagnostic.expected_pages:
        ranks = diagnostic.relevant_page_ranks[page]

        print(
            f"Page {page}: "
            f"{_format_ranks(ranks)}"
        )

    print()
    print("Recall by candidate size:")
    print("-" * 70)

    for k in (5, 10, 20):
        print(
            f"Recall@{k}: "
            f"{_recall_at_k(diagnostic, k):.2%}"
        )

    print()
    print(
        f"Unique chunks: "
        f"{diagnostic.unique_chunk_count}/"
        f"{len(diagnostic.retrieved_pages)}"
    )

    print(
        f"Unique pages: "
        f"{diagnostic.unique_page_count}/"
        f"{len(diagnostic.retrieved_pages)}"
    )

    print(
        f"Missing from Top-{len(diagnostic.retrieved_pages)}: "
        f"{list(diagnostic.missing_pages)}"
    )

    print()


if __name__ == "__main__":
    from app.evaluation.dataset import EVALUATION_DATASET

    print(
        f"Running candidate recall experiment "
        f"with Top-{CANDIDATE_LIMIT} retrieval..."
    )
    print()

    diagnostics = []

    for item in EVALUATION_DATASET:
        diagnostic = diagnose_question(
            item=item,
            limit=CANDIDATE_LIMIT,
        )

        diagnostics.append(diagnostic)

        print_diagnostic(diagnostic)

    print("=" * 70)
    print("OVERALL CANDIDATE RECALL")
    print("=" * 70)

    for k in (5, 10, 20):
        recalls = [
            _recall_at_k(
                diagnostic,
                k,
            )
            for diagnostic in diagnostics
        ]

        average_recall = (
            sum(recalls) / len(recalls)
            if recalls
            else 0.0
        )

        print(
            f"Average Recall@{k}: "
            f"{average_recall:.2%}"
        )

    print("=" * 70)