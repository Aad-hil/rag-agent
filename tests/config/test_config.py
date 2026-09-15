import pytest
from pydantic import ValidationError

from app.config import Settings


def test_valid_configuration_is_accepted():
    settings = Settings(
        database_url=(
            "postgresql+psycopg://"
            "rag_user:rag_password@localhost:5432/rag_agent"
        ),
        qdrant_url="http://localhost:6333",
        qdrant_collection="documents",
        ollama_url="http://localhost:11434/api/chat",
        ollama_model="gemma3",
        langsmith_tracing=False,
        langsmith_api_key=None,
        langsmith_project="rag-agent",
    )

    assert str(settings.database_url).startswith("postgresql")
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "documents"
    assert settings.ollama_url == "http://localhost:11434/api/chat"
    assert settings.ollama_model == "gemma3"


def test_invalid_database_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings(
            database_url="mysql://user:password@localhost/database",
            qdrant_url="http://localhost:6333",
        )


def test_invalid_http_url_is_rejected():
    with pytest.raises(ValidationError):
        Settings(
            database_url=(
                "postgresql+psycopg://"
                "rag_user:rag_password@localhost:5432/rag_agent"
            ),
            qdrant_url="localhost:6333",
        )


def test_empty_qdrant_collection_is_rejected():
    with pytest.raises(ValidationError):
        Settings(
            database_url=(
                "postgresql+psycopg://"
                "rag_user:rag_password@localhost:5432/rag_agent"
            ),
            qdrant_url="http://localhost:6333",
            qdrant_collection="   ",
        )
