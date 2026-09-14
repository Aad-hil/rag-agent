import json
import re
from dataclasses import dataclass

from app.llm.client import generate


@dataclass(frozen=True)
class JudgeResult:
    """
    Result returned by the local LLM judge.
    """

    score: float
    reason: str


CORRECTNESS_SYSTEM_PROMPT = """
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate whether the generated answer correctly answers the user's question
by comparing it against the reference answer.

Scoring rubric:
- 1.0 = Fully correct. The answer addresses the question and agrees with
  the important information in the reference answer.
- 0.5 = Partially correct. The answer contains some correct information but
  is incomplete or misses an important part.
- 0.0 = Incorrect or irrelevant. The answer is wrong, does not answer the
  question, or materially contradicts the reference answer.

Do not reward unsupported extra information merely because it sounds plausible.

Return ONLY valid JSON in exactly this format:
{"score": 1.0, "reason": "brief explanation"}

The score MUST be exactly one of:
0.0, 0.5, 1.0
"""


GROUNDEDNESS_SYSTEM_PROMPT = """
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate whether the generated answer is supported by the retrieved context.

Scoring rubric:
- 1.0 = Every meaningful factual claim in the answer is supported by the
  retrieved context.
- 0.5 = The answer is partly supported, but at least one meaningful factual
  claim is unsupported or insufficiently supported.
- 0.0 = The answer is largely unsupported by the retrieved context or
  contradicts the context.

Do not use outside knowledge. Judge only what can be supported by the
provided retrieved context.

Return ONLY valid JSON in exactly this format:
{"score": 1.0, "reason": "brief explanation"}

The score MUST be exactly one of:
0.0, 0.5, 1.0
"""


def _build_correctness_prompt(
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> str:
    """
    Build the prompt used to evaluate answer correctness.
    """

    return f"""
Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Evaluate the generated answer using the correctness rubric.
"""


def _build_groundedness_prompt(
    question: str,
    retrieved_context: str,
    generated_answer: str,
) -> str:
    """
    Build the prompt used to evaluate answer groundedness.
    """

    return f"""
Question:
{question}

Retrieved context:
{retrieved_context}

Generated answer:
{generated_answer}

Evaluate the generated answer using the groundedness rubric.
"""


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from the judge response.

    The judge is instructed to return JSON only, but this fallback handles
    responses that wrap the JSON in markdown fences or surrounding text.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        ).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if not match:
            raise ValueError(
                "Judge response did not contain valid JSON"
            )

        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Judge response contained invalid JSON"
            ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            "Judge response must be a JSON object"
        )

    return result


def _parse_judge_result(text: str) -> JudgeResult:
    """
    Parse and validate a judge response.
    """

    data = _extract_json(text)

    if "score" not in data:
        raise ValueError(
            "Judge response is missing 'score'"
        )

    if "reason" not in data:
        raise ValueError(
            "Judge response is missing 'reason'"
        )

    score = data["score"]
    reason = data["reason"]

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(
            "Judge score must be numeric"
        )

    score = float(score)

    if score not in {0.0, 0.5, 1.0}:
        raise ValueError(
            "Judge score must be exactly 0.0, 0.5, or 1.0"
        )

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            "Judge reason must be a non-empty string"
        )

    return JudgeResult(
        score=score,
        reason=reason.strip(),
    )


def judge_correctness(
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> JudgeResult:
    """
    Evaluate the semantic correctness of a generated answer.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not reference_answer.strip():
        raise ValueError("Reference answer cannot be empty")

    if not generated_answer.strip():
        raise ValueError("Generated answer cannot be empty")

    response = generate(
        prompt=_build_correctness_prompt(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        ),
        system_prompt=CORRECTNESS_SYSTEM_PROMPT,
    )

    return _parse_judge_result(response)


def judge_groundedness(
    question: str,
    retrieved_context: str,
    generated_answer: str,
) -> JudgeResult:
    """
    Evaluate whether a generated answer is supported by retrieved context.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not retrieved_context.strip():
        raise ValueError("Retrieved context cannot be empty")

    if not generated_answer.strip():
        raise ValueError("Generated answer cannot be empty")

    response = generate(
        prompt=_build_groundedness_prompt(
            question=question,
            retrieved_context=retrieved_context,
            generated_answer=generated_answer,
        ),
        system_prompt=GROUNDEDNESS_SYSTEM_PROMPT,
    )

    return _parse_judge_result(response)