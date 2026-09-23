"""
Build the FAISS index from public UK planning documents.
Run once: python scripts/build_index.py

PDFs are chunked by page; gov.uk guidance pages are fetched through the GOV.UK
Content API and chunked by section heading, so every chunk can be cited as
"document, page" or "document, section".

The build fails if any document fails. An earlier version skipped failures
silently and shipped an index holding one document out of four.
"""

import json
import re
import sys
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_CHARS = 2000
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("geosight/rag/index")
DOCS_DIR = Path("geosight/rag/docs")
HEADERS = {"User-Agent": "GeoSight/0.2 (+https://github.com/ssabeeth/geosight)"}

DOCUMENTS = [
    {
        "name": "National Planning Policy Framework (August 2026)",
        "kind": "pdf",
        "url": "https://assets.publishing.service.gov.uk/media/6aabe97844ec1aa417346c0b/National_Planning_Policy_Framework.pdf",
        "filename": "nppf_2026_08.pdf",
    },
    {
        "name": "Planning Practice Guidance: Flood risk and coastal change",
        "kind": "govuk",
        "path": "/guidance/flood-risk-and-coastal-change",
    },
    {
        "name": "Preparing a flood risk assessment: standing advice",
        "kind": "govuk",
        "path": "/guidance/preparing-a-flood-risk-assessment-standing-advice",
    },
    {
        "name": "Planning Practice Guidance: Natural environment",
        "kind": "govuk",
        "path": "/guidance/natural-environment",
    },
]


def split_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + CHUNK_CHARS].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        start += CHUNK_CHARS - OVERLAP_CHARS
    return chunks


# --- PDFs ------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading: {dest.name} ...")
    resp = requests.get(url, timeout=60, headers=HEADERS)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    dest.write_bytes(resp.content)
    print(f"  Saved {len(resp.content) / 1024:.0f} KB")


def pdf_chunks(pdf_path: Path, doc: dict) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page_num, page in enumerate(reader.pages, 1):
        for text in split_text(page.extract_text() or ""):
            chunks.append({
                "text": text,
                "source": doc["name"],
                "page": page_num,
                "section": None,
                "url": doc["url"],
            })
    return chunks


# --- GOV.UK guidance pages -----------------------------------------------------

class _SectionParser(HTMLParser):
    """Split a GOV.UK body into (heading, text) sections at each h2/h3."""

    def __init__(self):
        super().__init__()
        self.sections: list[tuple[str, list[str]]] = [("Introduction", [])]
        self._in_heading = False
        self._heading: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("h2", "h3"):
            self._in_heading, self._heading = True, []
        elif tag in ("p", "li", "br", "tr"):
            self.sections[-1][1].append(" ")

    def handle_endtag(self, tag):
        if tag in ("h2", "h3") and self._in_heading:
            self._in_heading = False
            self.sections.append(("".join(self._heading).strip(), []))

    def handle_data(self, data):
        (self._heading if self._in_heading else self.sections[-1][1]).append(data)


def govuk_chunks(doc: dict) -> list[dict]:
    resp = requests.get(f"https://www.gov.uk/api/content{doc['path']}", timeout=60, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    if data.get("schema_name") == "redirect":
        raise ValueError(f"{doc['path']} now redirects to {data.get('redirects')}")
    body = data["details"].get("body") or "".join(p["body"] for p in data["details"].get("parts", []))
    if not body:
        raise ValueError(f"{doc['path']} has no body text")

    parser = _SectionParser()
    parser.feed(body)
    chunks = []
    for heading, parts in parser.sections:
        for text in split_text(unescape("".join(parts))):
            chunks.append({
                "text": text,
                "source": doc["name"],
                "page": None,
                "section": heading or None,
                "url": f"https://www.gov.uk{doc['path']}",
            })
    return chunks


# --- Build -----------------------------------------------------------------------

def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n Collecting documents...")
    all_chunks: list[dict] = []
    failures = []

    for doc in DOCUMENTS:
        print(f"  {doc['name']}")
        try:
            if doc["kind"] == "pdf":
                dest = DOCS_DIR / doc["filename"]
                download_pdf(doc["url"], dest)
                chunks = pdf_chunks(dest, doc)
            else:
                chunks = govuk_chunks(doc)
        except Exception as e:
            print(f"  FAILED: {e}")
            failures.append(doc["name"])
            continue
        if not chunks:
            print("  FAILED: no text extracted")
            failures.append(doc["name"])
            continue
        all_chunks.extend(chunks)
        print(f"  {len(chunks)} chunks")

    if failures:
        print(f"\n{len(failures)} document(s) failed: {', '.join(failures)}. Index not written.")
        sys.exit(1)

    print(f"\n Total chunks: {len(all_chunks)}")
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
        json.dump(all_chunks, f, indent=1)

    print("\n Index saved:")
    print(f"   FAISS index : {faiss_path} ({faiss_path.stat().st_size / 1024:.0f} KB)")
    print(f"   Metadata    : {meta_path} ({meta_path.stat().st_size / 1024:.0f} KB)")
    print(f"   Vectors     : {index.ntotal} x {dim}d")


if __name__ == "__main__":
    build_index()
