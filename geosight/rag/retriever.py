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

_INDEX_DIR = Path(__file__).parent / "index"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(_INDEX_DIR / "faiss.index"))
FAISS_META_PATH = os.getenv("FAISS_META_PATH", str(_INDEX_DIR / "metadata.json"))
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
    section: str | None = None
    url: str | None = None
    score: float
    query: str = ""

    @property
    def location(self) -> str:
        """Where in the document the chunk came from, for citations."""
        if self.page:
            return f"p.{self.page}"
        if self.section:
            return f'section "{self.section}"'
        return ""

    @property
    def label(self) -> str:
        return f"{self.source}, {self.location}" if self.location else self.source


class RAGResult(BaseModel):
    queries: list[str]
    chunks: list[RetrievedChunk]
    context: str


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(f"[{i}] Source: {c.label}\n{c.text}" for i, c in enumerate(chunks, 1))


def retrieve_many(queries: list[str], per_query: int = 2, max_chunks: int = 6) -> RAGResult:
    """
    Run each query and keep its best `per_query` chunks, skipping duplicates,
    so every topic in the report gets its own policy evidence instead of one
    topic crowding out the rest.
    """
    model, index, metadata = _load_resources()
    query_vecs = model.encode(queries, normalize_embeddings=True).astype(np.float32)
    # Over-fetch so duplicates across queries can be skipped.
    scores, ids = index.search(query_vecs, per_query + max_chunks)

    chunks: list[RetrievedChunk] = []
    seen: set[int] = set()
    for query, row_scores, row_ids in zip(queries, scores, ids, strict=True):
        taken = 0
        for score, idx in zip(row_scores, row_ids, strict=True):
            if idx == -1 or idx in seen or taken == per_query:
                continue
            seen.add(int(idx))
            taken += 1
            meta = metadata[idx]
            chunks.append(RetrievedChunk(
                text=meta["text"],
                source=meta["source"],
                page=meta.get("page"),
                section=meta.get("section"),
                url=meta.get("url"),
                score=float(score),  # cosine similarity from IndexFlatIP
                query=query,
            ))
    chunks = chunks[:max_chunks]
    return RAGResult(queries=queries, chunks=chunks, context=_format_context(chunks))


def retrieve(query: str, top_k: int = RAG_TOP_K) -> RAGResult:
    """Retrieve the most relevant policy chunks for a single query."""
    return retrieve_many([query], per_query=top_k, max_chunks=top_k)
