from fastapi import FastAPI

from app.database import check_database_connection
from app.vector_store import check_qdrant_connection


app = FastAPI(
    title="RAG Agent",
    description="Production-oriented RAG agent built with LangGraph",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    database_ok = check_database_connection()
    qdrant_ok = check_qdrant_connection()

    return {
        "status": "ok",
        "service": "rag-agent",
        "dependencies": {
            "postgres": database_ok,
            "qdrant": qdrant_ok,
        },
    }
