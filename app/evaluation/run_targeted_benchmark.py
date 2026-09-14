from app.evaluation.benchmark import evaluate_dataset
from app.evaluation.dataset import EVALUATION_DATASET


TARGET_QUESTIONS = {
    "What information is included in the prompt for a RAG deployment?",
    "How is conversation history maintained in the LangChain implementation?",
    "What AWS services are involved in monitoring the solution?",
}


def main() -> None:
    targeted_dataset = [
        item
        for item in EVALUATION_DATASET
        if item.question in TARGET_QUESTIONS
    ]

    if len(targeted_dataset) != len(TARGET_QUESTIONS):
        raise ValueError(
            "Targeted benchmark dataset does not contain "
            "all expected questions."
        )

    summary = evaluate_dataset(
        dataset=targeted_dataset,
        limit=5,
    )

    print("\n=== Targeted RAG Benchmark ===")
    print("Questions: Q6, Q8, Q9\n")

    print("Aggregate")
    print(
        f"  Retrieval Recall@5: "
        f"{summary.retrieval_recall:.2%}"
    )
    print(
        f"  Retrieval Precision@5: "
        f"{summary.retrieval_precision:.2%}"
    )
    print(
        f"  Retrieval MRR@5: "
        f"{summary.retrieval_mrr:.2%}"
    )
    print(
        f"  Answer Rate: "
        f"{summary.answer_rate:.2%}"
    )
    print(
        f"  Citation Rate: "
        f"{summary.citation_rate:.2%}"
    )
    print(
        f"  Correctness: "
        f"{summary.correctness_score:.2%}"
    )
    print(
        f"  Groundedness: "
        f"{summary.groundedness_score:.2%}"
    )
    print(
        f"  Citation Correctness: "
        f"{summary.citation_correctness_score:.2%}"
    )

    for result in summary.results:
        print("\n" + "=" * 80)
        print(f"Question: {result.question}")
        print("=" * 80)

        print(
            f"\nExpected pages:  {result.expected_pages}"
        )

        print(
            f"Retrieved pages: {result.retrieved_pages}"
        )

        if result.error:
            print(
                f"\nERROR: {result.error}"
            )
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
                f"  Has answer:      "
                f"{result.generation.has_answer}"
            )
            print(
                f"  Has citations:   "
                f"{result.generation.has_citations}"
            )
            print(
                f"  Citations valid: "
                f"{result.generation.citations_valid}"
            )
            print(
                f"  Citation pages:  "
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