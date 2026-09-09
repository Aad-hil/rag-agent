from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.generation.answer import Answer, answer_question


def generate_answer(state: AgentState) -> dict[str, Answer]:
    return {
        "answer": answer_question(state["question"]),
    }


builder = StateGraph(AgentState)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
