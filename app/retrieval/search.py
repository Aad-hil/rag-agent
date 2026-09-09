from dataclasses import dataclass

from app.config import settings
from app.ingestion.embeddings import embed_text
from app.vector_store import qdrant_client


@dataclass
class SearchResult:
    text: str
    score: float
    source: str
    page_number: int
    chunk_index: int


def search(
    query: str,
    limit: int = 5,
) -> list[SearchResult]:
    """
    Perform semantic search against Qdrant.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    query_vector = embed_text(query)

    response = qdrant_client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return [
        SearchResult(
            text=point.payload["text"],
            score=point.score,
            source=point.payload["source"],
            page_number=point.payload["page_number"],
            chunk_index=point.payload["chunk_index"],
        )
        for point in response.points
    ]
