from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from .config import settings
from .indexer import search
from .llm import generate_answer


def build_context(results: List[Dict]) -> Tuple[str, List[Dict]]:
    context_parts: List[str] = []
    citations: List[Dict] = []

    for idx, r in enumerate(results, start=1):
        header = f"[{idx}] Source: {r.get('source')} (chunk {r.get('chunk_index')})"
        text = r.get('text') or ""
        context_parts.append(header + "\n" + text)
        citations.append({"n": idx, **r})

    context = "\n\n---\n\n".join(context_parts)
    # keep it small
    if len(context) > 8000:
        context = context[:8000] + "\n\n[context trimmed]"

    return context, citations


def answer_question(question: str) -> Dict[str, Any]:
    t0 = time.time()

    retrieved = search(question, top_k=settings.top_k)
    retrieval_ms = int((time.time() - t0) * 1000)

    best_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)

    if not retrieved or best_score < settings.min_score:
        return {
            "question": question,
            "answer": (
                "I don’t have enough information in the current knowledge base to answer that. "
                "Please share the relevant document or ask a more specific question."
            ),
            "citations": [],
            "latency": {"retrieval_ms": retrieval_ms, "generation_ms": 0, "total_ms": retrieval_ms},
            "cost": {"provider": settings.llm_provider, "usage": None},
            "debug": {"best_score": best_score, "top_k": settings.top_k},
        }

    context, citations = build_context(retrieved)
    llm_res = generate_answer(question=question, context=context, citations=citations)

    total_ms = int((time.time() - t0) * 1000)

    return {
        "question": question,
        "answer": llm_res.answer,
        "citations": [
            {
                "n": c["n"],
                "source": c.get("source"),
                "chunk_index": c.get("chunk_index"),
                "score": round(float(c.get("score") or 0.0), 3),
            }
            for c in citations
        ],
        "latency": {
            "retrieval_ms": retrieval_ms,
            "generation_ms": llm_res.latency_ms,
            "total_ms": total_ms,
        },
        "cost": {
            "provider": settings.llm_provider,
            "model": settings.openai_model if settings.llm_provider == "openai" else "mock",
            "usage": llm_res.usage,
        },
        "debug": {"best_score": best_score, "top_k": settings.top_k},
    }
