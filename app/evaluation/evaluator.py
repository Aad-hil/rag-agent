from app.evaluation.dataset import EVALUATION_DATASET
from app.retrieval.search import search


def evaluate_retrieval(limit: int = 5) -> None:
    """
    Evaluate retrieval using:
    - Recall@k
    - Precision@k
    - Mean Reciprocal Rank (MRR)
    - Retrieval similarity scores
    """

    passed = 0
    reciprocal_ranks = []
    precision_scores = []

    for index, item in enumerate(EVALUATION_DATASET, start=1):
        results = search(item.question, limit=limit)

        retrieved_pages = [
            result.page_number
            for result in results
        ]

        expected_pages = set(item.relevant_pages)

        relevant_count = sum(
            page in expected_pages
            for page in retrieved_pages
        )

        # Recall@k
        hit = relevant_count > 0

        if hit:
            passed += 1

        # MRR
        reciprocal_rank = 0.0

        for rank, page in enumerate(retrieved_pages, start=1):
            if page in expected_pages:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

        # Precision@k
        precision = (
            relevant_count / len(retrieved_pages)
            if retrieved_pages
            else 0.0
        )

        precision_scores.append(precision)

        status = "PASS" if hit else "FAIL"

        print(
            f"Q{index:02d} [{status}] "
            f"Expected: {sorted(expected_pages)} "
            f"P@{limit}: {precision:.2f} "
            f"RR: {reciprocal_rank:.2f}"
        )

        for rank, result in enumerate(results, start=1):
            relevance = (
                "RELEVANT"
                if result.page_number in expected_pages
                else "other"
            )

            print(
                f"    #{rank} "
                f"score={result.score:.4f} "
                f"page={result.page_number} "
                f"chunk={result.chunk_index} "
                f"{relevance}"
            )

        print()

    total = len(EVALUATION_DATASET)

    recall = passed / total if total else 0

    mrr = (
        sum(reciprocal_ranks) / total
        if total
        else 0
    )

    mean_precision = (
        sum(precision_scores) / total
        if total
        else 0
    )

    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Recall@{limit}: {recall:.2%}")
    print(f"Precision@{limit}: {mean_precision:.2%}")
    print(f"MRR@{limit}: {mrr:.2%}")
    print("=" * 60)
