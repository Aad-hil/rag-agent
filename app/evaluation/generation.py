from dataclasses import dataclass

from app.generation.answer import Answer


@dataclass(frozen=True)
class GenerationEvaluationResult:
    """
    Deterministic evaluation results for a generated answer.
    """

    has_answer: bool
    is_abstention: bool
    has_citations: bool
    citations_valid: bool
    citation_pages: tuple[int, ...]
    cited_relevant_pages: tuple[int, ...]
    unsupported_citation_pages: tuple[int, ...]


ABSTENTION_TEXT = (
    "The provided documents do not contain enough "
    "information to answer this question."
)


def has_answer(answer: Answer) -> bool:
    """
    Return True when the answer contains non-whitespace text.
    """
    return bool(answer.text.strip())


def is_abstention(answer: Answer) -> bool:
    """
    Return True when the answer uses the standard abstention response.
    """
    return answer.text.strip() == ABSTENTION_TEXT


def has_citations(answer: Answer) -> bool:
    """
    Return True when the answer contains at least one citation.
    """
    return bool(answer.citations)


def citation_pages(answer: Answer) -> tuple[int, ...]:
    """
    Return the unique page numbers cited by the answer.
    """
    return tuple(
        dict.fromkeys(
            citation.page_number
            for citation in answer.citations
        )
    )


def validate_citations(
    answer: Answer,
    valid_citation_ids: set[int],
) -> bool:
    """
    Return True when every citation ID in the answer refers
    to a valid retrieved source.
    """
    return all(
        citation.citation_id in valid_citation_ids
        for citation in answer.citations
    )


def cited_relevant_pages(
    answer: Answer,
    relevant_pages: set[int],
) -> tuple[int, ...]:
    """
    Return unique cited pages that are part of the expected
    relevant pages.
    """
    return tuple(
        page
        for page in citation_pages(answer)
        if page in relevant_pages
    )


def unsupported_citation_pages(
    answer: Answer,
    retrieved_pages: set[int],
) -> tuple[int, ...]:
    """
    Return unique cited pages that were not present in the
    retrieved context.

    This checks the final Answer citation metadata rather than
    parsing rendered page references from answer text.
    """
    return tuple(
        page
        for page in citation_pages(answer)
        if page not in retrieved_pages
    )


def evaluate_generation(
    answer: Answer,
    relevant_pages: set[int],
    valid_citation_ids: set[int],
    retrieved_pages: set[int],
) -> GenerationEvaluationResult:
    """
    Evaluate a generated answer using deterministic checks.

    This function does not call the LLM and does not perform retrieval.
    """

    return GenerationEvaluationResult(
        has_answer=has_answer(answer),
        is_abstention=is_abstention(answer),
        has_citations=has_citations(answer),
        citations_valid=validate_citations(
            answer,
            valid_citation_ids,
        ),
        citation_pages=citation_pages(answer),
        cited_relevant_pages=cited_relevant_pages(
            answer,
            relevant_pages,
        ),
        unsupported_citation_pages=unsupported_citation_pages(
            answer,
            retrieved_pages,
        ),
    )
