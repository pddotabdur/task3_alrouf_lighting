FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DOCS_DIR=data/docs
ENV INDEX_PATH=index/tfidf_index.joblib
ENV LLM_PROVIDER=mock

CMD ["sh", "-c", "python cli.py --ingest --rebuild && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
