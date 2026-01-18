from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import settings
from .loader import DocumentChunk, build_chunks


def build_tfidf_vectorizer() -> TfidfVectorizer:
    # char_wb ngrams are robust across English + Arabic without extra tokenizers.
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
        lowercase=False,
    )


def ingest(rebuild: bool = False) -> Dict[str, int]:
    """Build TF-IDF index and persist it.

    Index file contains:
    - vectorizer
    - matrix (sparse)
    - chunks metadata
    """
    if rebuild and os.path.exists(settings.index_path):
        os.remove(settings.index_path)

    chunks: List[DocumentChunk] = build_chunks()
    if not chunks:
        return {"documents": 0, "chunks": 0, "added": 0}

    texts = [c.text for c in chunks]
    vectorizer = build_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(texts)

    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunks": [asdict(c) for c in chunks],
    }

    os.makedirs(os.path.dirname(settings.index_path), exist_ok=True)
    joblib.dump(payload, settings.index_path)

    return {
        "documents": len(set(c.source for c in chunks)),
        "chunks": len(chunks),
        "added": len(chunks),
    }


def load_index() -> Dict:
    if not os.path.exists(settings.index_path):
        raise FileNotFoundError(
            f"Index not found at {settings.index_path}. Run ingest first."
        )
    return joblib.load(settings.index_path)


def search(query: str, top_k: int | None = None) -> List[Dict]:
    top_k = top_k or settings.top_k

    idx = load_index()
    vectorizer: TfidfVectorizer = idx["vectorizer"]
    matrix = idx["matrix"]
    chunks = idx["chunks"]

    q_vec = vectorizer.transform([query])

    # cosine similarity for L2-normalized TF-IDF vectors = dot product
    scores = (matrix @ q_vec.T).toarray().ravel()

    if scores.size == 0:
        return []

    top_indices = np.argsort(-scores)[:top_k]

    results: List[Dict] = []
    for i in top_indices:
        c = chunks[int(i)]
        results.append(
            {
                "id": c["id"],
                "text": c["text"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": float(scores[int(i)]),
            }
        )

    return results
