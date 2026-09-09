from typing import Literal

from langgraph.graph import END, START, StateGraph

from app.agent.relevance import is_relevant
from app.agent.rewrite import rewrite_query as rewrite_query_text
from app.agent.state import AgentState
from app.generation.answer import Answer, abstain_answer, answer_from_results
from app.retrieval.search import SearchResult, search


def retrieve(state: AgentState) -> dict[str, list[SearchResult] | int]:
    query = state.get("rewritten_query") or state["question"]

    return {
        "results": search(query),
        "retry_count": state.get("retry_count", 0),
    }


def check_relevance(state: AgentState) -> dict[str, bool]:
    return {
        "is_relevant": is_relevant(state["results"]),
    }


def route_after_relevance(
    state: AgentState,
) -> Literal["generate_answer", "rewrite_query", "abstain"]:
    if state["is_relevant"]:
        return "generate_answer"

    if state.get("retry_count", 0) >= 1:
        return "abstain"

    return "rewrite_query"


def rewrite_query(state: AgentState) -> dict[str, str | int]:
    return {
        "rewritten_query": rewrite_query_text(
            state["question"]
        ),
        "retry_count": state.get("retry_count", 0) + 1,
    }


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
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate_answer", generate_answer)
builder.add_node("abstain", abstain)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "check_relevance")

builder.add_conditional_edges(
    "check_relevance",
    route_after_relevance,
    {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query",
        "abstain": "abstain",
    },
)

builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate_answer", END)
builder.add_edge("abstain", END)

graph = builder.compile()