from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from pypdf import PdfReader

from .config import settings


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class DocumentChunk:
    id: str
    text: str
    source: str
    chunk_index: int


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append(f"[page {i+1}]\n{txt}")
    return "\n\n".join(pages)


def load_documents(docs_dir: str) -> List[Tuple[str, str]]:
    """Returns list of (source_name, raw_text)."""
    root = Path(docs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Docs directory not found: {root.resolve()}")

    docs: List[Tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() in {".txt", ".md"}:
            text = _read_text_file(path)
        else:
            text = _read_pdf_file(path)

        # basic sanitation
        text = "\n".join(line.rstrip() for line in text.splitlines())
        if text.strip():
            docs.append((path.name, text))

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple character-based chunking with overlap."""
    if chunk_size <= 0:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks() -> List[DocumentChunk]:
    docs = load_documents(settings.docs_dir)
    all_chunks: List[DocumentChunk] = []

    for src, raw in docs:
        parts = chunk_text(raw, settings.chunk_size, settings.chunk_overlap)
        for idx, part in enumerate(parts):
            chunk_id = f"{src}::chunk_{idx}"
            all_chunks.append(
                DocumentChunk(id=chunk_id, text=part, source=src, chunk_index=idx)
            )

    return all_chunks
