from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent.relevance import is_relevant
from app.agent.state import AgentState
from app.generation.answer import Answer, abstain_answer, answer_from_results
from app.retrieval.search import SearchResult, search


def retrieve(state: AgentState) -> dict[str, list[SearchResult]]:
    return {
        "results": search(state["question"]),
    }


def check_relevance(state: AgentState) -> dict[str, bool]:
    return {
        "is_relevant": is_relevant(state["results"]),
    }


def route_after_relevance(
    state: AgentState,
) -> Literal["generate_answer", "abstain"]:
    if state["is_relevant"]:
        return "generate_answer"

    return "abstain"


def generate_answer(state: AgentState) -> dict[str, Answer]:
    return {
        "answer": answer_from_results(
            state["question"],
            state["results"],
        ),
    }


def abstain(_: AgentState) -> dict[str, Answer]:
    return {
        "answer": abstain_answer(),
    }


builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("check_relevance", check_relevance)
builder.add_node("generate_answer", generate_answer)
builder.add_node("abstain", abstain)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "check_relevance")
builder.add_conditional_edges(
    "check_relevance",
    route_after_relevance,
)
builder.add_edge("generate_answer", END)
builder.add_edge("abstain", END)

graph = builder.compile()
