from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = SentenceTransformer(MODEL_NAME)


def get_embedding_dimension() -> int:
    return _model.get_embedding_dimension()


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding for a single piece of text.
    """

    vector = _model.encode(
        text,
        normalize_embeddings=True,
    )

    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts.
    """

    if not texts:
        return []

    vectors = _model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return vectors.tolist()
