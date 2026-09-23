import importlib.util
import json
from pathlib import Path

import faiss

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location("build_index", ROOT / "scripts" / "build_index.py")
build_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_index)


def test_committed_index_holds_every_document():
    """An earlier index silently held one document out of four."""
    metadata = json.loads((ROOT / "geosight/rag/index/metadata.json").read_text())
    index = faiss.read_index(str(ROOT / "geosight/rag/index/faiss.index"))
    assert index.ntotal == len(metadata)
    assert {m["source"] for m in metadata} == {d["name"] for d in build_index.DOCUMENTS}
    assert all(m.get("page") or m.get("section") for m in metadata)


def test_govuk_body_is_split_by_heading():
    parser = build_index._SectionParser()
    parser.feed("<p>Intro text.</p><h2>The sequential test</h2><p>Steer new development</p>"
                "<ul><li>to Flood Zone 1.</li></ul><h3>Exceptions</h3><p>Some.</p>")
    sections = [(h, "".join(t).split()) for h, t in parser.sections]
    assert sections == [
        ("Introduction", ["Intro", "text."]),
        ("The sequential test", ["Steer", "new", "development", "to", "Flood", "Zone", "1."]),
        ("Exceptions", ["Some."]),
    ]


def test_split_text_overlaps_and_drops_scraps():
    text = "x" * (build_index.CHUNK_CHARS * 2)
    chunks = build_index.split_text(text)
    assert all(len(c) >= build_index.MIN_CHUNK_CHARS for c in chunks)
    assert sum(len(c) for c in chunks) > len(text)  # overlap
    assert build_index.split_text("too short") == []
