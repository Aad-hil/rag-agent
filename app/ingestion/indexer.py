from qdrant_client.models import PointStruct

from app.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.embeddings import embed_texts
from app.ingestion.loader import load_pdf
from app.vector_store import create_collection, qdrant_client


def index_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    collection_name: str | None = None,
) -> int:
    """
    Load, chunk, embed, and index a PDF into Qdrant.

    Returns the number of indexed chunks.

    By default, the configured Qdrant collection is used.
    A different collection can be supplied for experiments.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    target_collection = (
        collection_name
        if collection_name is not None
        else settings.qdrant_collection
    )

    # 1. Load PDF
    pages = load_pdf(pdf_path)

    # 2. Create chunks
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError("No chunks were generated from the PDF")

    # 3. Create Qdrant collection
    vectors = embed_texts(
        [chunk.text for chunk in chunks]
    )

    vector_size = len(vectors[0])

    create_collection(
        vector_size=vector_size,
        collection_name=target_collection,
    )

    # 4. Build Qdrant points
    points = [
        PointStruct(
            id=chunk.chunk_index,
            vector=vector,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        )
    ]

    # 5. Insert into Qdrant
    qdrant_client.upsert(
        collection_name=target_collection,
        points=points,
    )

    return len(points)