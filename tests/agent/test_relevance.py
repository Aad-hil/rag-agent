from app.agent.relevance import is_relevant
from app.retrieval.search import SearchResult


def _result(score: float) -> SearchResult:
    return SearchResult(
        text="Test content",
        score=score,
        source="document.pdf",
        page_number=1,
        chunk_index=0,
    )


def test_relevant_retrieval_passes_the_gate():
    assert is_relevant([_result(0.45)])


def test_irrelevant_retrieval_fails_the_gate():
    assert not is_relevant([_result(0.30)])


def test_empty_retrieval_fails_the_gate():
    assert not is_relevant([])
