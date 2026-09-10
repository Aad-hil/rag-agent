from unittest.mock import patch

from app.evaluation.generation import (
    ABSTENTION_TEXT,
    citation_pages,
    cited_relevant_pages,
    evaluate_generation,
    generate_and_evaluate,
    has_answer,
    has_citations,
    is_abstention,
    retrieved_page_numbers,
    unsupported_citation_pages,
    validate_citations,
)
from app.generation.answer import Answer, AnswerCitation
from app.retrieval.search import SearchResult


def make_citation(
    citation_id: int,
    page_number: int,
) -> AnswerCitation:
    return AnswerCitation(
        citation_id=citation_id,
        source="test.pdf",
        page_number=page_number,
        chunk_index=0,
    )


def make_answer(
    text: str = "Test answer.",
    citations: tuple[AnswerCitation, ...] = (),
) -> Answer:
    return Answer(
        text=text,
        citations=citations,
    )


def make_search_result(
    page_number: int,
    chunk_index: int = 0,
) -> SearchResult:
    return SearchResult(
        text=f"Content from page {page_number}.",
        score=0.9,
        source="test.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
    )


def test_has_answer_returns_true_for_non_empty_answer():
    answer = make_answer(
        text="This is a grounded answer."
    )

    assert has_answer(answer) is True


def test_has_answer_returns_false_for_empty_answer():
    answer = make_answer(
        text="   "
    )

    assert has_answer(answer) is False


def test_is_abstention_detects_standard_abstention():
    answer = make_answer(
        text=ABSTENTION_TEXT
    )

    assert is_abstention(answer) is True


def test_is_abstention_returns_false_for_normal_answer():
    answer = make_answer(
        text="The solution uses Amazon DynamoDB."
    )

    assert is_abstention(answer) is False


def test_has_citations_returns_true_when_citations_exist():
    answer = make_answer(
        citations=(
            make_citation(1, 20),
        )
    )

    assert has_citations(answer) is True


def test_has_citations_returns_false_when_no_citations_exist():
    answer = make_answer()

    assert has_citations(answer) is False


def test_citation_pages_returns_unique_pages_in_order():
    answer = make_answer(
        citations=(
            make_citation(1, 20),
            make_citation(2, 21),
            make_citation(3, 20),
        )
    )

    assert citation_pages(answer) == (
        20,
        21,
    )


def test_retrieved_page_numbers_returns_unique_pages_in_order():
    results = [
        make_search_result(20),
        make_search_result(21),
        make_search_result(20, chunk_index=1),
    ]

    assert retrieved_page_numbers(results) == (
        20,
        21,
    )


def test_validate_citations_returns_true_for_valid_ids():
    answer = make_answer(
        citations=(
            make_citation(1, 20),
            make_citation(2, 21),
        )
    )

    assert validate_citations(
        answer,
        {1, 2, 3},
    ) is True


def test_validate_citations_returns_false_for_invalid_id():
    answer = make_answer(
        citations=(
            make_citation(1, 20),
            make_citation(9, 21),
        )
    )

    assert validate_citations(
        answer,
        {1, 2, 3},
    ) is False


def test_cited_relevant_pages_returns_only_relevant_pages():
    answer = make_answer(
        citations=(
            make_citation(1, 20),
            make_citation(2, 50),
            make_citation(3, 82),
        )
    )

    assert cited_relevant_pages(
        answer,
        {20, 82},
    ) == (
        20,
        82,
    )


def test_cited_relevant_pages_returns_empty_when_no_relevant_page_is_cited():
    answer = make_answer(
        citations=(
            make_citation(1, 183),
            make_citation(2, 188),
        )
    )

    assert cited_relevant_pages(
        answer,
        {157, 169},
    ) == ()


def test_unsupported_citation_pages_detects_pages_not_retrieved():
    answer = make_answer(
        citations=(
            make_citation(1, 20),
            make_citation(2, 50),
        )
    )

    assert unsupported_citation_pages(
        answer,
        {20, 82},
    ) == (
        50,
    )


def test_evaluate_generation_returns_expected_result():
    answer = make_answer(
        text="The solution uses DynamoDB.",
        citations=(
            make_citation(1, 21),
            make_citation(2, 50),
        ),
    )

    result = evaluate_generation(
        answer=answer,
        relevant_pages={21},
        valid_citation_ids={1, 2, 3},
        retrieved_pages=(21, 50),
    )

    assert result.has_answer is True
    assert result.is_abstention is False
    assert result.has_citations is True
    assert result.citations_valid is True

    assert result.citation_pages == (
        21,
        50,
    )

    assert result.retrieved_pages == (
        21,
        50,
    )

    assert result.cited_relevant_pages == (
        21,
    )

    assert result.unsupported_citation_pages == ()


def test_evaluate_generation_detects_invalid_and_unsupported_citations():
    answer = make_answer(
        text="Test answer.",
        citations=(
            make_citation(1, 20),
            make_citation(9, 99),
        ),
    )

    result = evaluate_generation(
        answer=answer,
        relevant_pages={20},
        valid_citation_ids={1, 2, 3},
        retrieved_pages=(20, 50),
    )

    assert result.has_answer is True
    assert result.has_citations is True
    assert result.citations_valid is False

    assert result.citation_pages == (
        20,
        99,
    )

    assert result.retrieved_pages == (
        20,
        50,
    )

    assert result.cited_relevant_pages == (
        20,
    )

    assert result.unsupported_citation_pages == (
        99,
    )


def test_evaluate_generation_handles_abstention():
    answer = make_answer(
        text=ABSTENTION_TEXT,
    )

    result = evaluate_generation(
        answer=answer,
        relevant_pages={20},
        valid_citation_ids={1, 2, 3},
        retrieved_pages=(20, 50),
    )

    assert result.has_answer is True
    assert result.is_abstention is True
    assert result.has_citations is False
    assert result.citations_valid is True
    assert result.citation_pages == ()
    assert result.retrieved_pages == (
        20,
        50,
    )
    assert result.cited_relevant_pages == ()
    assert result.unsupported_citation_pages == ()


def test_generate_and_evaluate_uses_real_pipeline():
    results = [
        make_search_result(20),
        make_search_result(21),
        make_search_result(50),
    ]

    answer = make_answer(
        text="The orchestrator processes the request.",
        citations=(
            make_citation(1, 20),
            make_citation(2, 21),
        ),
    )

    with patch(
        "app.evaluation.generation.search",
        return_value=results,
    ) as mocked_search, patch(
        "app.evaluation.generation.answer_from_results",
        return_value=answer,
    ) as mocked_answer_from_results:
        result = generate_and_evaluate(
            question="How does the orchestrator process requests?",
            relevant_pages={20, 21},
            limit=5,
        )

    mocked_search.assert_called_once_with(
        "How does the orchestrator process requests?",
        limit=5,
    )

    mocked_answer_from_results.assert_called_once_with(
        question="How does the orchestrator process requests?",
        results=results,
    )

    assert result.has_answer is True
    assert result.is_abstention is False
    assert result.has_citations is True
    assert result.citations_valid is True

    assert result.citation_pages == (
        20,
        21,
    )

    assert result.retrieved_pages == (
        20,
        21,
        50,
    )

    assert result.cited_relevant_pages == (
        20,
        21,
    )

    assert result.unsupported_citation_pages == ()


def test_generate_and_evaluate_preserves_retrieved_pages_when_no_citations():
    results = [
        make_search_result(157),
        make_search_result(169),
        make_search_result(183),
    ]

    answer = make_answer(
        text="Conversation history is maintained by the solution."
    )

    with patch(
        "app.evaluation.generation.search",
        return_value=results,
    ), patch(
        "app.evaluation.generation.answer_from_results",
        return_value=answer,
    ):
        result = generate_and_evaluate(
            question="How is conversation history maintained?",
            relevant_pages={157, 169},
            limit=5,
        )

    assert result.has_answer is True
    assert result.has_citations is False
    assert result.citation_pages == ()
    assert result.retrieved_pages == (
        157,
        169,
        183,
    )
    assert result.cited_relevant_pages == ()
    assert result.unsupported_citation_pages == ()