from fastapi import FastAPI

app = FastAPI(
    title="RAG Agent",
    description="Production-oriented RAG agent built with LangGraph",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "rag-agent",
    }
