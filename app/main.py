import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent.graph import graph
from app.database import check_database_connection
from app.logging_config import configure_logging
from app.vector_store import check_qdrant_connection


configure_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="RAG Agent",
    description="Production-oriented RAG agent built with LangGraph",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask the RAG agent.",
    )


class CitationResponse(BaseModel):
    citation_id: int
    source: str
    page_number: int
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


@app.exception_handler(Exception)
async def handle_unexpected_exception(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.exception(
        "Unhandled exception while processing %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred while processing the request."
        },
    )


@app.get("/health")
def health_check():
    database_ok = check_database_connection()
    qdrant_ok = check_qdrant_connection()

    logger.info(
        "Health check completed: postgres=%s qdrant=%s",
        database_ok,
        qdrant_ok,
    )

    return {
        "status": "ok",
        "service": "rag-agent",
        "dependencies": {
            "postgres": database_ok,
            "qdrant": qdrant_ok,
        },
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_agent(request: QueryRequest) -> QueryResponse:
    logger.info("Processing query request")

    result = graph.invoke(
        {
            "question": request.question,
            "retry_count": 0,
        }
    )

    answer = result["answer"]

    citations = [
        CitationResponse(
            citation_id=citation.citation_id,
            source=citation.source,
            page_number=citation.page_number,
            chunk_index=citation.chunk_index,
        )
        for citation in answer.citations
    ]

    logger.info(
        "Query completed successfully with %d citations",
        len(citations),
    )

    return QueryResponse(
        answer=answer.text,
        citations=citations,
    )