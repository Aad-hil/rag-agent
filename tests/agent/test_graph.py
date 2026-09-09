import app.agent.graph as agent_graph

from app.agent.graph import graph
from app.generation.answer import Answer
from app.retrieval.search import SearchResult


def test_answerable_question_returns_answer_with_citations():
    result = graph.invoke(
        {
            "question": (
                "How does the LangChain Orchestrator process "
                "incoming requests?"
            ),
        }
    )

    answer = result["answer"]

    assert isinstance(answer, Answer)
    assert answer.text
    assert answer.citations


def test_unsupported_question_abstains_without_citations():
    result = graph.invoke(
        {
            "question": (
                "What is the population of India according to "
                "this document?"
            ),
        }
    )

    answer = result["answer"]

    assert isinstance(answer, Answer)
    assert (
        answer.text
        == "The provided documents do not contain enough "
        "information to answer this question."
    )
    assert answer.citations == ()


def test_irrelevant_question_does_not_invoke_generation(monkeypatch):
    results = [
        SearchResult(
            text="Unrelated content",
            score=0.25,
            source="document.pdf",
            page_number=1,
            chunk_index=0,
        ),
    ]

    generation_called = False

    def fail_if_generation_is_called(*_args, **_kwargs):
        nonlocal generation_called

        generation_called = True
        raise AssertionError(
            "Generation must not run for irrelevant results"
        )

    monkeypatch.setattr(
        agent_graph,
        "search",
        lambda _question: results,
    )
    monkeypatch.setattr(
        agent_graph,
        "answer_from_results",
        fail_if_generation_is_called,
    )

    result = graph.invoke(
        {
            "question": "An unrelated question",
        }
    )

    assert not generation_called
    assert result["answer"].citations == ()


def test_irrelevant_question_rewrites_and_retrieves_again(monkeypatch):
    retrieval_queries = []
    rewrite_called = False

    first_results = [
        SearchResult(
            text="Unrelated content",
            score=0.25,
            source="document.pdf",
            page_number=1,
            chunk_index=0,
        ),
    ]

    second_results = [
        SearchResult(
            text="Relevant content",
            score=0.60,
            source="document.pdf",
            page_number=157,
            chunk_index=10,
        ),
    ]

    def fake_search(query):
        retrieval_queries.append(query)

        if len(retrieval_queries) == 1:
            return first_results

        return second_results

    def fake_rewrite(question):
        nonlocal rewrite_called

        rewrite_called = True
        return "rewritten search query"

    monkeypatch.setattr(
        agent_graph,
        "search",
        fake_search,
    )
    monkeypatch.setattr(
        agent_graph,
        "rewrite_query_text",
        fake_rewrite,
    )

    result = graph.invoke(
        {
            "question": "Original question",
        }
    )

    assert rewrite_called

    assert retrieval_queries == [
        "Original question",
        "rewritten search query",
    ]

    assert result["retry_count"] == 1
    assert result["rewritten_query"] == "rewritten search query"
    assert result["is_relevant"] is True


def test_relevant_retrieval_generates_answer_without_rewrite(monkeypatch):
    results = [
        SearchResult(
            text="Relevant content",
            score=0.60,
            source="document.pdf",
            page_number=20,
            chunk_index=10,
        ),
    ]

    rewrite_called = False
    generation_called = False

    def fail_if_rewrite_is_called(_question):
        nonlocal rewrite_called

        rewrite_called = True
        raise AssertionError(
            "Rewrite must not run when retrieval is already relevant"
        )

    def fake_generate_answer(question, results):
        nonlocal generation_called

        generation_called = True

        return Answer(
            text="Generated answer",
            citations=(),
        )

    monkeypatch.setattr(
        agent_graph,
        "search",
        lambda _question: results,
    )
    monkeypatch.setattr(
        agent_graph,
        "rewrite_query_text",
        fail_if_rewrite_is_called,
    )
    monkeypatch.setattr(
        agent_graph,
        "answer_from_results",
        fake_generate_answer,
    )

    result = graph.invoke(
        {
            "question": "A relevant question",
        }
    )

    assert not rewrite_called
    assert generation_called
    assert result["is_relevant"] is True
    assert result["retry_count"] == 0