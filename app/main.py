from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.agent.graph import graph
from app.database import check_database_connection
from app.vector_store import check_qdrant_connection


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


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_agent(request: QueryRequest) -> QueryResponse:
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

    return QueryResponse(
        answer=answer.text,
        citations=citations,
    )