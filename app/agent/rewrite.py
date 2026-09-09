from app.llm.client import generate


REWRITE_SYSTEM_PROMPT = """
You rewrite user questions into better search queries for a document
retrieval system.

Your job is to improve retrieval, not answer the question.

Rules:
1. Preserve the original meaning.
2. Keep important technical terms.
3. Make the query specific and focused.
4. Do not add facts that are not present in the user's question.
5. Return ONLY the rewritten search query.
6. Do not include explanations, quotes, or labels.
"""


def rewrite_query(question: str) -> str:
    """
    Rewrite a user question into a retrieval-friendly search query.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    return generate(
        prompt=question,
        system_prompt=REWRITE_SYSTEM_PROMPT,
    ).strip()