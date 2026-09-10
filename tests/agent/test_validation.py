from app.agent.validation import is_valid_answer
from app.generation.answer import Answer, AnswerCitation


def test_answer_with_text_and_citation_is_valid():
    answer = Answer(
        text="Grounded answer",
        citations=(
            AnswerCitation(
                citation_id=1,
                source="document.pdf",
                page_number=1,
                chunk_index=0,
            ),
        ),
    )

    assert is_valid_answer(answer) is True


def test_answer_with_whitespace_only_text_is_invalid():
    answer = Answer(
        text="   ",
        citations=(
            AnswerCitation(
                citation_id=1,
                source="document.pdf",
                page_number=1,
                chunk_index=0,
            ),
        ),
    )

    assert is_valid_answer(answer) is False


def test_answer_without_citations_is_invalid():
    answer = Answer(
        text="Ungrounded answer",
        citations=(),
    )

    assert is_valid_answer(answer) is False
