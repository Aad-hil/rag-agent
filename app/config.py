from dotenv import load_dotenv
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


settings = Settings()
