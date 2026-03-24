from __future__ import annotations

import os
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings, get_index, get_pc
from .utils import make_vector_id, clean_text

# 4 chars ≈ 1 token; convert token counts to character counts
_CHUNK_SIZE_CHARS = settings.chunk_size * 4
_CHUNK_OVERLAP_CHARS = settings.chunk_overlap * 4

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE_CHARS,
    chunk_overlap=_CHUNK_OVERLAP_CHARS,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _extract_pages(file_path: str) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples. Page is 0 for plain text files."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        pages = []
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((i + 1, text))
        return pages
    else:
        with open(file_path, encoding="utf-8") as f:
            return [(0, clean_text(f.read()))]


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Pinecone Inference API, in batches of 96."""
    pc = get_pc()
    vectors: list[list[float]] = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = pc.inference.embed(
            model=settings.embedding_model,
            inputs=batch,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        vectors.extend([item["values"] for item in response.data])
    return vectors


def delete_document(filename: str) -> int:
    """
    Delete all Pinecone vectors that belong to *filename*.
    Returns the number of vectors deleted (approximate via stats diff).
    """
    index = get_index()

    # Count before
    before = index.describe_index_stats().total_vector_count

    # Delete by metadata filter — removes every chunk for this file
    index.delete(filter={"source_file": {"$eq": filename}})

    # Count after
    after = index.describe_index_stats().total_vector_count
    return max(0, before - after)


def ingest(file_path: str) -> list[dict]:
    """
    Parse, chunk, embed, and upsert a document into Pinecone.
    Returns a list of chunk metadata dicts.
    """
    filename = Path(file_path).name
    pages = _extract_pages(file_path)

    # Build flat list of chunk records preserving page provenance
    chunks: list[dict] = []
    chunk_index = 0
    for page_num, page_text in pages:
        splits = _splitter.split_text(page_text)
        for text in splits:
            chunks.append({
                "chunk_index": chunk_index,
                "source_file": filename,
                "page": page_num,
                "text": text,
            })
            chunk_index += 1

    if not chunks:
        return []

    # Embed all chunks
    texts = [c["text"] for c in chunks]
    vectors = _embed_batch(texts)

    # Build Pinecone upsert payload
    records = [
        {
            "id": make_vector_id(filename, c["chunk_index"]),
            "values": vec,
            "metadata": {
                "source_file": c["source_file"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
                "text": c["text"],
            },
        }
        for c, vec in zip(chunks, vectors)
    ]

    index = get_index()
    batch_size = 100
    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i : i + batch_size])

    return chunks
