from dataclasses import dataclass

from app.retrieval.search import SearchResult


@dataclass(frozen=True)
class ContextSource:
    citation_id: int
    source: str
    page_number: int
    chunk_index: int


@dataclass(frozen=True)
class BuiltContext:
    text: str
    sources: tuple[ContextSource, ...]


def build_context(
    results: list[SearchResult],
) -> BuiltContext:
    """
    Build citation-aware context from retrieved chunks.
    """

    if not results:
        return BuiltContext(
            text="",
            sources=(),
        )

    context_parts = []
    sources = []

    for citation_id, result in enumerate(results, start=1):
        sources.append(
            ContextSource(
                citation_id=citation_id,
                source=result.source,
                page_number=result.page_number,
                chunk_index=result.chunk_index,
            )
        )

        context_parts.append(
            (
                f"[{citation_id}]\n"
                f"Source: {result.source}\n"
                f"Page: {result.page_number}\n"
                f"Chunk: {result.chunk_index}\n"
                f"Content:\n{result.text}"
            )
        )

    return BuiltContext(
        text="\n\n---\n\n".join(context_parts),
        sources=tuple(sources),
    )
