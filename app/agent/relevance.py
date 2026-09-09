from app.retrieval.search import SearchResult


# Temporary baseline derived from the current retrieval controls: supported
# evaluation questions have best scores of 0.4286 or higher, while the existing
# unsupported India question has a best score of 0.2872. Calibrate this value
# against a labeled negative evaluation set in a later evaluation phase.
MIN_RELEVANCE_SCORE = 0.40


def is_relevant(results: list[SearchResult]) -> bool:
    """Return whether the retrieved results meet the baseline score gate."""

    return bool(results) and max(
        result.score
        for result in results
    ) >= MIN_RELEVANCE_SCORE
