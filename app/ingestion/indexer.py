from qdrant_client.models import PointStruct

from app.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.embeddings import embed_texts
from app.ingestion.loader import load_pdf
from app.vector_store import create_collection, qdrant_client


def index_pdf(pdf_path: str) -> int:
    """
    Load, chunk, embed, and index a PDF into Qdrant.

    Returns the number of indexed chunks.
    """

    # 1. Load PDF
    pages = load_pdf(pdf_path)

    # 2. Create chunks
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No chunks were generated from the PDF")

    # 3. Create Qdrant collection
    vectors = embed_texts([chunk.text for chunk in chunks])

    vector_size = len(vectors[0])

    create_collection(vector_size)

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
        for chunk, vector in zip(chunks, vectors)
    ]

    # 5. Insert into Qdrant
    qdrant_client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    return len(points)
