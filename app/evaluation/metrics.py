from collections.abc import Sequence


def hit_rate_at_k(
    retrieved_pages: Sequence[int],
    relevant_pages: Sequence[int],
    k: int,
) -> float:
    """
    Return 1.0 if at least one relevant page appears in the
    top-k retrieved results, otherwise 0.0.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_pages:
        return 0.0

    retrieved = retrieved_pages[:k]
    relevant = set(relevant_pages)

    return float(any(page in relevant for page in retrieved))


def recall_at_k(
    retrieved_pages: Sequence[int],
    relevant_pages: Sequence[int],
    k: int,
) -> float:
    """
    Standard Recall@K.

    Measures the fraction of unique relevant pages that appear
    in the top-k retrieved results.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_pages:
        return 0.0

    retrieved = set(retrieved_pages[:k])
    relevant = set(relevant_pages)

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    retrieved_pages: Sequence[int],
    relevant_pages: Sequence[int],
    k: int,
) -> float:
    """
    Precision@K.

    Measures the fraction of the top-k retrieved results that
    are relevant.

    Duplicate retrieved pages are counted as separate retrieved
    results because each result represents a retrieval position.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved = retrieved_pages[:k]

    if not retrieved:
        return 0.0

    relevant = set(relevant_pages)

    relevant_count = sum(
        page in relevant
        for page in retrieved
    )

    return relevant_count / len(retrieved)


def reciprocal_rank(
    retrieved_pages: Sequence[int],
    relevant_pages: Sequence[int],
) -> float:
    """
    Return the reciprocal rank of the first relevant result.

    Returns 0.0 when no relevant result is retrieved.
    """
    if not relevant_pages:
        return 0.0

    relevant = set(relevant_pages)

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[int]],
    relevant_pages: Sequence[Sequence[int]],
) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.
    """
    if len(rankings) != len(relevant_pages):
        raise ValueError(
            "rankings and relevant_pages must have the same length"
        )

    if not rankings:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(
            rankings,
            relevant_pages,
            strict=True,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)