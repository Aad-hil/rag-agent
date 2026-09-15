from unittest.mock import patch

from fastapi.testclient import TestClient

from app.generation.answer import Answer, AnswerCitation
from app.main import app


client = TestClient(
    app,
    raise_server_exceptions=False,
)


def _build_test_answer() -> Answer:
    return Answer(
        text=(
            "The LangChain Orchestrator processes incoming requests "
            "through API Gateway and Amazon SQS. [Page 20]"
        ),
        citations=(
            AnswerCitation(
                citation_id=1,
                source="generative-ai-application-builder-on-aws.pdf",
                page_number=20,
                chunk_index=59,
            ),
        ),
    )


def test_health_returns_healthy_status():
    with (
        patch(
            "app.main.check_database_connection",
            return_value=True,
        ),
        patch(
            "app.main.check_qdrant_connection",
            return_value=True,
        ),
    ):
        response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "rag-agent",
        "dependencies": {
            "postgres": True,
            "qdrant": True,
        },
    }


def test_query_returns_answer_and_citations():
    answer = _build_test_answer()

    with patch(
        "app.main.graph.invoke",
        return_value={
            "answer": answer,
        },
    ) as mock_invoke:
        response = client.post(
            "/query",
            json={
                "question": (
                    "How does the LangChain Orchestrator "
                    "process incoming requests?"
                ),
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == answer.text

    assert data["citations"] == [
        {
            "citation_id": 1,
            "source": "generative-ai-application-builder-on-aws.pdf",
            "page_number": 20,
            "chunk_index": 59,
        }
    ]

    mock_invoke.assert_called_once_with(
        {
            "question": (
                "How does the LangChain Orchestrator "
                "process incoming requests?"
            ),
            "retry_count": 0,
        }
    )


def test_query_rejects_empty_question():
    response = client.post(
        "/query",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422


def test_query_rejects_missing_question():
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422


def test_query_returns_empty_citations_when_answer_has_none():
    answer = Answer(
        text=(
            "The provided documents do not contain enough "
            "information to answer this question."
        ),
        citations=(),
    )

    with patch(
        "app.main.graph.invoke",
        return_value={
            "answer": answer,
        },
    ):
        response = client.post(
            "/query",
            json={
                "question": "What is the capital of Mars?",
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == answer.text
    assert data["citations"] == []


def test_query_returns_safe_error_when_unexpected_exception_occurs():
    with patch(
        "app.main.graph.invoke",
        side_effect=RuntimeError("database connection details"),
    ):
        response = client.post(
            "/query",
            json={
                "question": "Test unexpected failure",
            },
        )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "An internal error occurred while processing "
            "the request."
        )
    }

    assert "database connection details" not in response.text