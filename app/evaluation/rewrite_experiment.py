from app.agent.rewrite import rewrite_query
from app.evaluation.dataset import EVALUATION_DATASET
from app.retrieval.search import search


CANDIDATE_LIMIT = 20


def run_experiment() -> None:
    """
    Compare original and rewritten queries using Top-20
    dense retrieval.
    """

    print("=" * 80)
    print("QUERY REWRITE RETRIEVAL EXPERIMENT")
    print("=" * 80)
    print()

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):
        rewritten_query = rewrite_query(
            item.question
        )

        original_results = search(
            item.question,
            limit=CANDIDATE_LIMIT,
        )

        rewritten_results = search(
            rewritten_query,
            limit=CANDIDATE_LIMIT,
        )

        original_pages = [
            result.page_number
            for result in original_results
        ]

        rewritten_pages = [
            result.page_number
            for result in rewritten_results
        ]

        print("=" * 80)
        print(f"Q{index:02d}")
        print("=" * 80)

        print(
            f"Original:  {item.question}"
        )

        print(
            f"Rewritten: {rewritten_query}"
        )

        print(
            f"Expected:  {list(item.relevant_pages)}"
        )

        print()

        print(
            f"Original Top-{CANDIDATE_LIMIT}:"
        )
        print(original_pages)

        print()

        print(
            f"Rewrite Top-{CANDIDATE_LIMIT}:"
        )
        print(rewritten_pages)

        print()


if __name__ == "__main__":
    run_experiment()