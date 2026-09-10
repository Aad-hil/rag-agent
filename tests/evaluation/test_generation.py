from app.evaluation.generation import (
    ABSTENTION_TEXT,
    citation_pages,
    cited_relevant_pages,
    evaluate_generation,
    has_answer,
    has_citations,
    is_abstention,
    unsupported_citation_pages,
    validate_citations,
)
from app.generation.answer import Answer, AnswerCitation


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
        retrieved_pages={21, 50},
    )

    assert result.has_answer is True
    assert result.is_abstention is False
    assert result.has_citations is True
    assert result.citations_valid is True

    assert result.citation_pages == (
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
        retrieved_pages={20, 50},
    )

    assert result.has_answer is True
    assert result.has_citations is True
    assert result.citations_valid is False

    assert result.citation_pages == (
        20,
        99,
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
        retrieved_pages={20, 50},
    )

    assert result.has_answer is True
    assert result.is_abstention is True
    assert result.has_citations is False
    assert result.citations_valid is True
    assert result.citation_pages == ()
    assert result.cited_relevant_pages == ()
    assert result.unsupported_citation_pages == ()
