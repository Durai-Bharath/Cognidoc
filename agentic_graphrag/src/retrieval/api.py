"""FastAPI service: ``POST /query`` and ``GET /report/{id}``.

    graphrag serve [--offline] [--host 127.0.0.1 --port 8000]

Requires the optional ``api`` extra (``pip install -e ".[api]"``). The framework is not
re-entrant (each query owns the trace sink), so queries are serialised with a lock.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieval.graph import AgenticGraphRetrievalFramework, load_report
from retrieval.models import StatementCitation, Turn


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list)


class QueryResponse(BaseModel):
    report_id: str
    answer: str
    verified: bool
    low_confidence: bool
    cached: bool
    citations: list[StatementCitation]
    report_url: str


def create_app(framework: AgenticGraphRetrievalFramework) -> FastAPI:
    app = FastAPI(title="Agentic GraphRAG - Stage 3 retrieval", version="1.0.0")
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "graph": framework.kg.backend,
            "vectors": framework.vdb.backend,
            "cache_entries": len(framework.ctx.cache),
        }

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        with lock:
            result = framework.query(req.query, req.history)
        return QueryResponse(
            report_id=result.report_id,
            answer=result.answer,
            verified=result.verified,
            low_confidence=result.low_confidence,
            cached=result.cached,
            citations=result.citations,
            report_url=f"/report/{result.report_id}",
        )

    @app.get("/report/{report_id}")
    def report(report_id: str) -> dict[str, Any]:
        data = load_report(report_id, framework.config)
        if data is None:
            raise HTTPException(status_code=404, detail=f"report {report_id} not found")
        return data

    return app
