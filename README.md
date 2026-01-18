# Task 3 — RAG Knowledge Base (AR/EN) — **Light MVP**

This is a minimal RAG knowledge base that runs locally **without heavy GPU dependencies**.

It satisfies the demo requirements:
- 3–5 docs ingestion
- chunk → vectorize (TF‑IDF) → index
- retrieve Top‑K
- answer with citations
- Arabic + English queries
- refusal if out-of-scope
- latency + usage report

## Why TF‑IDF?
For the demo, TF‑IDF + cosine similarity works well enough and keeps install time small.
You can swap to embeddings (Chroma + sentence-transformers) later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Build the index

```bash
python cli.py --ingest --rebuild
```

## Query from CLI

English:
```bash
python cli.py --question "What is the standard warranty?"
```

Arabic:
```bash
python cli.py --question "كم مدة الضمان القياسي؟"
```

## Run an API

```bash
uvicorn app.main:app --reload --port 8000
```

Then:
- `POST /ingest?rebuild=true`
- `POST /query` with body:
```json
{ "question": "What is the typical lead time?" }
```

## Optional: OpenAI generation

Set:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_KEY
```

Without a key, the system still works in `mock` mode.
