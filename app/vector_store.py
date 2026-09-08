from qdrant_client import QdrantClient

from app.config import settings


qdrant_client = QdrantClient(
    url=settings.qdrant_url,
)


def check_qdrant_connection() -> bool:
    qdrant_client.get_collections()
    return True
