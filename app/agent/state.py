from typing import NotRequired, TypedDict

from app.generation.answer import Answer
from app.retrieval.search import SearchResult


class AgentState(TypedDict):
    question: str
    rewritten_query: NotRequired[str]
    retry_count: int
    results: list[SearchResult]
    is_relevant: bool
    answer: Answer
    is_answer_valid: bool
