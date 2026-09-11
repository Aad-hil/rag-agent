from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings


qdrant_client = QdrantClient(
    url=settings.qdrant_url,
)


def check_qdrant_connection() -> bool:
    qdrant_client.get_collections()
    return True


def create_collection(
    vector_size: int,
    recreate: bool = False,
    collection_name: str | None = None,
) -> None:
    """
    Create a Qdrant collection if it does not exist.

    By default, the configured collection is used.
    """

    target_collection = (
        collection_name
        if collection_name is not None
        else settings.qdrant_collection
    )

    collections = qdrant_client.get_collections()

    existing_names = {
        collection.name
        for collection in collections.collections
    }

    if target_collection in existing_names:
        if not recreate:
            return

        qdrant_client.delete_collection(
            collection_name=target_collection
        )

    qdrant_client.create_collection(
        collection_name=target_collection,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )