from app.generation.answer import answer_question


def test_answerable_question_returns_grounded_answer():
    result = answer_question(
        "How does the LangChain Orchestrator process incoming requests?",
        limit=3,
    )

    assert result.text
    assert "Page 20" in result.text or "Page 21" in result.text

    assert result.citations

    citation_pages = {
        citation.page_number
        for citation in result.citations
    }

    assert 20 in citation_pages or 21 in citation_pages


def test_unanswerable_question_abstains_without_citations():
    result = answer_question(
        "What is the population of India according to this document?",
        limit=5,
    )

    assert (
        result.text
        == "The information about the population of India "
        "is not available in the provided documents."
    )

    assert result.citations == ()
