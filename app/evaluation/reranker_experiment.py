from app.evaluation.dataset import EVALUATION_DATASET
from app.retrieval.reranker import rerank
from app.retrieval.search import search


CANDIDATE_LIMIT = 20
FINAL_LIMIT = 5


def run_experiment() -> None:
    """
    Compare dense retrieval rankings with cross-encoder reranking.
    """

    print("=" * 70)
    print("CROSS-ENCODER RERANKING EXPERIMENT")
    print("=" * 70)
    print()

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1,
    ):
        candidates = search(
            item.question,
            limit=CANDIDATE_LIMIT,
        )

        reranked = rerank(
            query=item.question,
            results=candidates,
            limit=FINAL_LIMIT,
        )

        print("=" * 70)
        print(f"Q{index:02d}")
        print("=" * 70)

        print(
            f"Question: {item.question}"
        )

        print(
            f"Expected pages: "
            f"{list(item.relevant_pages)}"
        )

        print()
        print("Dense retrieval Top-5:")
        print("-" * 70)

        for rank, result in enumerate(
            candidates[:FINAL_LIMIT],
            start=1,
        ):
            relevance = (
                "RELEVANT"
                if result.page_number in item.relevant_pages
                else "IRRELEVANT"
            )

            print(
                f"{rank}. "
                f"Page {result.page_number} | "
                f"Embedding {result.score:.4f} | "
                f"{relevance}"
            )

        print()
        print("Cross-encoder reranked Top-5:")
        print("-" * 70)

        for rank, reranked_result in enumerate(
            reranked,
            start=1,
        ):
            result = reranked_result.result

            relevance = (
                "RELEVANT"
                if result.page_number in item.relevant_pages
                else "IRRELEVANT"
            )

            print(
                f"{rank}. "
                f"Page {result.page_number} | "
                f"Rerank {reranked_result.score:.4f} | "
                f"{relevance}"
            )

        print()


if __name__ == "__main__":
    run_experiment()