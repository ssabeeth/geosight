"""
Build FAISS index from public UK policy/planning PDFs.
Run once: python scripts/build_index.py
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("geosight/rag/index")
DOCS_DIR = Path("geosight/rag/docs")

DOCUMENTS = [
    {
        "name": "National Planning Policy Framework (NPPF) 2023",
        "url": "https://assets.publishing.service.gov.uk/media/65a11af7e8f5ec000f1f8c46/NPPF_December_2023.pdf",
        "filename": "nppf_2023.pdf",
    },
    {
        "name": "Environment Agency — Flood Risk Standing Advice",
        "url": "https://assets.publishing.service.gov.uk/media/5f0d4c33d3bf7f7213d9fd39/LIT_8116.pdf",
        "filename": "ea_flood_standing_advice.pdf",
    },
    {
        "name": "Natural England — Green Infrastructure Framework",
        "url": "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1062319/Green_Infrastructure_Framework-Principles_and_Standards_for_England.pdf",
        "filename": "natural_england_gi_framework.pdf",
    },
    {
        "name": "DEFRA — England Biodiversity Strategy",
        "url": "https://assets.publishing.service.gov.uk/media/63c9a12dd3bf7f37b2839e16/england-biodiversity-strategy.pdf",
        "filename": "defra_biodiversity_strategy.pdf",
    },
]


def download_pdf(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return True
    print(f"  Downloading: {dest.name} ...")
    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "GeoSight/0.1"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  Saved {len(resp.content) / 1024:.0f}KB")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def extract_text_chunks(pdf_path: Path, source_name: str) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    chunks = []
    char_size = CHUNK_SIZE * 4
    char_overlap = CHUNK_OVERLAP * 4

    for page_num, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if len(text) < 100:
            continue
        start = 0
        while start < len(text):
            chunk_text = text[start:start + char_size].strip()
            if len(chunk_text) > 100:
                chunks.append({
                    "text": chunk_text,
                    "source": source_name,
                    "page": page_num,
                })
            start += char_size - char_overlap

    return chunks


def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n Downloading documents...")
    all_chunks: list[dict] = []

    for doc in DOCUMENTS:
        dest = DOCS_DIR / doc["filename"]
        if download_pdf(doc["url"], dest):
            print(f"  Extracting chunks from: {doc['name']}")
            chunks = extract_text_chunks(dest, doc["name"])
            all_chunks.extend(chunks)
            print(f"  {len(chunks)} chunks extracted")

    print(f"\n Total chunks: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks extracted. Check PDF downloads.")
        sys.exit(1)

    print(f"\n Embedding with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    print("\n Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "metadata.json"

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n Index saved:")
    print(f"   FAISS index : {faiss_path} ({faiss_path.stat().st_size / 1024:.0f}KB)")
    print(f"   Metadata    : {meta_path} ({meta_path.stat().st_size / 1024:.0f}KB)")
    print(f"   Vectors     : {index.ntotal} x {dim}d")


if __name__ == "__main__":
    build_index()