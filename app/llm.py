from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import settings


@dataclass
class LLMResult:
    answer: str
    usage: Dict[str, Any]
    latency_ms: int


def _is_arabic(text: str) -> bool:
    # Basic heuristic: count Arabic letters
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars / max(len(text), 1) > 0.10


def generate_answer(question: str, context: str, citations: List[Dict]) -> LLMResult:
    """Generates an answer using either OpenAI or a mock model.

    - Returns citations separately; the UI will render them.
    """
    t0 = time.time()

    want_ar = _is_arabic(question)

    if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)

            system = (
                "You are a helpful assistant for Alrouf Lighting. "
                "Answer strictly using the provided context. "
                "If the answer is not in the context, say you don't know and suggest what to ask for. "
                "Keep answers concise. "
                "Always include citation markers like [1], [2] referring to the provided sources. "
            )
            if want_ar:
                system += "Respond in Arabic."
            else:
                system += "Respond in English."

            user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nINSTRUCTIONS:\n- Use only the context\n- If missing, refuse politely\n- Add citations like [1] [2]"

            resp = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )

            answer = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            usage_dict = usage.model_dump() if usage else {}

            return LLMResult(
                answer=answer.strip(),
                usage=usage_dict,
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception as e:
            # Fall back to mock mode
            pass

    # Mock mode: extractive summary + citations
    # This keeps the project runnable without secrets.
    bullets = []
    for i, c in enumerate(citations, start=1):
        # Take first sentence-ish
        snippet = c.get("text", "").strip().replace("\n", " ")
        snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
        bullets.append(f"- {snippet} [{i}]")

    if want_ar:
        prefix = "لا أستطيع إعطاء إجابة نهائية بدون نموذج لغة متصل. ولكن هذه هي المقاطع الأكثر صلة من قاعدة المعرفة:"
    else:
        prefix = "I can’t generate a fully polished answer without an LLM key. Here are the most relevant snippets from the knowledge base:"

    answer = prefix + "\n" + "\n".join(bullets)

    return LLMResult(
        answer=answer,
        usage={"provider": "mock"},
        latency_ms=int((time.time() - t0) * 1000),
    )
