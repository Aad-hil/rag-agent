from app.evaluation.benchmark import evaluate_dataset
from app.evaluation.dataset import EVALUATION_DATASET


def main() -> None:
    summary = evaluate_dataset(
        dataset=EVALUATION_DATASET,
        limit=5,
    )

    print("\n=== RAG Agent Benchmark ===\n")

    print("Retrieval")
    print(f"  Hit Rate@5:   {summary.retrieval_hit_rate:.2%}")
    print(f"  Recall@5:     {summary.retrieval_recall:.2%}")
    print(f"  Precision@5:  {summary.retrieval_precision:.2%}")
    print(f"  MRR@5:        {summary.retrieval_mrr:.2%}")

    print("\nGeneration")
    print(f"  Answer Rate:            {summary.answer_rate:.2%}")
    print(f"  Citation Rate:          {summary.citation_rate:.2%}")
    print(
        "  Relevant Citation Rate: "
        f"{summary.relevant_citation_rate:.2%}"
    )
    print(
        f"  Abstention Rate:        {summary.abstention_rate:.2%}"
    )
    print(
        "  Valid Citation Rate:    "
        f"{summary.valid_citation_rate:.2%}"
    )
    print(
        "  Unsupported Citation:   "
        f"{summary.unsupported_citation_rate:.2%}"
    )

    print("\nAnswer Quality")
    print(
        f"  Correctness:            "
        f"{summary.correctness_score:.2%}"
    )
    print(
        f"  Groundedness:           "
        f"{summary.groundedness_score:.2%}"
    )
    print(
        f"  Citation Correctness:   "
        f"{summary.citation_correctness_score:.2%}"
    )

    print("\nPer-question results")

    for index, result in enumerate(summary.results, start=1):
        print("\n" + "=" * 80)
        print(f"Q{index}: {result.question}")
        print("=" * 80)

        print(
            f"\nExpected pages:  {result.expected_pages}"
        )

        print(
            f"Retrieved pages: {result.retrieved_pages}"
        )

        if result.error:
            print(f"\nERROR: {result.error}")
            continue

        print("\nGenerated answer")
        print("-" * 80)

        if result.generated_answer is not None:
            print(result.generated_answer)
        else:
            print("No generated answer.")

        print("-" * 80)

        if result.generation:
            print("\nGeneration evaluation")
            print(
                f"  Has answer:       "
                f"{result.generation.has_answer}"
            )
            print(
                f"  Is abstention:    "
                f"{result.generation.is_abstention}"
            )
            print(
                f"  Has citations:    "
                f"{result.generation.has_citations}"
            )
            print(
                f"  Citations valid:  "
                f"{result.generation.citations_valid}"
            )
            print(
                f"  Citation pages:   "
                f"{result.generation.citation_pages}"
            )

        if result.answer_quality:
            quality = result.answer_quality

            print("\nAnswer quality")

            print(
                f"  Correctness: "
                f"{quality.correctness_score:.1f}"
            )

            print(
                f"  Correctness reason: "
                f"{quality.correctness_reason}"
            )

            print(
                f"  Groundedness: "
                f"{quality.groundedness_score:.1f}"
            )

            print(
                f"  Groundedness reason: "
                f"{quality.groundedness_reason}"
            )

            print(
                f"  Citation correctness: "
                f"{quality.citation_correctness_score:.1f}"
            )


if __name__ == "__main__":
    main()