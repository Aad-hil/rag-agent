from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.retrieval.search import SearchResult


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = CrossEncoder(MODEL_NAME)


@dataclass(frozen=True)
class RerankedResult:
    """
    A retrieval result with its cross-encoder reranking score.
    """

    result: SearchResult
    score: float


def rerank(
    query: str,
    results: list[SearchResult],
    limit: int = 5,
) -> list[RerankedResult]:
    """
    Rerank retrieved candidates using a cross-encoder.

    The input results are treated as the candidate pool. The
    cross-encoder scores each query/document pair independently.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    if not results:
        return []

    pairs = [
        (query, result.text)
        for result in results
    ]

    scores = _model.predict(pairs)

    reranked = [
        RerankedResult(
            result=result,
            score=float(score),
        )
        for result, score in zip(
            results,
            scores,
            strict=True,
        )
    ]

    reranked.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return reranked[:limit]
