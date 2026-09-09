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
        raise AssertionError("Generation must not run for irrelevant results")

    monkeypatch.setattr(agent_graph, "search", lambda _question: results)
    monkeypatch.setattr(
        agent_graph,
        "answer_from_results",
        fail_if_generation_is_called,
    )

    result = graph.invoke({"question": "An unrelated question"})

    assert not generation_called
    assert result["answer"].citations == ()
