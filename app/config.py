from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    database_url: str
    qdrant_url: str
    qdrant_collection: str = "documents"

    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "gemma3"

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "rag-agent"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith("postgresql"):
            raise ValueError(
                "DATABASE_URL must use a PostgreSQL connection URL."
            )

        return value

    @field_validator("qdrant_url", "ollama_url")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError(
                "URL must start with http:// or https://."
            )

        return value

    @field_validator("qdrant_collection")
    @classmethod
    def validate_qdrant_collection(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "QDRANT_COLLECTION cannot be empty."
            )

        return value


settings = Settings()
