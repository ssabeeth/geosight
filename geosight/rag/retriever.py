"""
RAG retriever — FAISS-backed document retrieval over pre-embedded UK policy corpus.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "geosight/rag/index/faiss.index")
FAISS_META_PATH = os.getenv("FAISS_META_PATH", "geosight/rag/index/metadata.json")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Load once, reuse across queries
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_metadata: list[dict] | None = None


def _load_resources():
    global _model, _index, _metadata
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _index is None:
        _index = faiss.read_index(FAISS_INDEX_PATH)
    if _metadata is None:
        with open(FAISS_META_PATH) as f:
            _metadata = json.load(f)
    return _model, _index, _metadata


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    score: float


class RAGResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    context: str


def retrieve(query: str, top_k: int = RAG_TOP_K) -> RAGResult:
    """
    Retrieve the most relevant policy chunks for a given query.
    """
    model, index, metadata = _load_resources()

    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx]
        chunks.append(RetrievedChunk(
            text=meta["text"],
            source=meta["source"],
            page=meta.get("page"),
            score=float(dist),  # cosine similarity from IndexFlatIP
        ))

    context_parts = []
    for i, c in enumerate(chunks, 1):
        page_str = f", p.{c.page}" if c.page else ""
        context_parts.append(f"[{i}] Source: {c.source}{page_str}\n{c.text}")
    context = "\n\n---\n\n".join(context_parts)

    return RAGResult(query=query, chunks=chunks, context=context)