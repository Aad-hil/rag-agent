# RAG Agent

A production-oriented Retrieval-Augmented Generation (RAG) agent built incrementally with FastAPI, LangGraph, Qdrant, PostgreSQL, LangSmith, Docker, and pytest.

## Phase 0

The initial foundation includes:

- FastAPI application
- Health endpoint
- Python dependency manifest
- Environment template
- Git hygiene

## Run locally

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API docs.

Health check:

```text
GET /health
```

Expected response:

```json
{
  "status": "ok",
  "service": "rag-agent"
}
```

## Roadmap

1. Foundation
2. Basic RAG pipeline
3. LangGraph agent
4. FastAPI + PostgreSQL integration
5. Evaluation and observability
6. Dockerized production setup
7. Portfolio/demo polish
