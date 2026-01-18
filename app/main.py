from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .indexer import ingest
from .rag import answer_question


app = FastAPI(title="Alrouf RAG KB MVP (Light)", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_docs(rebuild: bool = False):
    stats = ingest(rebuild=rebuild)
    return {"ok": True, "stats": stats}


@app.post("/query")
def query(req: QueryRequest):
    return answer_question(req.question)
