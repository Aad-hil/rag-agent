from typing import TypedDict

from app.generation.answer import Answer
from app.retrieval.search import SearchResult


class AgentState(TypedDict):
    question: str
    results: list[SearchResult]
    is_relevant: bool
    answer: Answer
