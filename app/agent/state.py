from typing import TypedDict

from app.generation.answer import Answer


class AgentState(TypedDict):
    question: str
    answer: Answer
