from app.agent.graph import graph
from app.generation.answer import Answer


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
        == "The information about the population of India "
        "is not available in the provided documents."
    )
    assert answer.citations == ()
