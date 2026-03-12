# 🗺️ GeoSight — UK Land Intelligence Agent

> Ask anything about any piece of UK land.  
> Flood risk · protected designations · planning policy · land use · site photography —  
> synthesised by a multi-tool agentic AI pipeline. **Runs entirely free on your laptop.**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Cost](https://img.shields.io/badge/LLM%20cost-%C2%A30%2Fquery-brightgreen)

---

## What It Does

GeoSight is a **LangGraph multi-tool agent** that takes a UK postcode and an optional
site photograph, then autonomously:

1. **Geocodes** the location (OpenStreetMap / Nominatim)
2. **Fetches flood risk** — active warnings and monitoring stations (Environment Agency API)
3. **Identifies protected designations** within 2km — SSSIs, AONBs, NNRs, National Parks (Natural England ArcGIS API)
4. **Retrieves land use context** — agricultural, natural, and water features (OpenStreetMap / Overpass)
5. **Analyses a site photograph** using a local vision model (LLaVA via Ollama)
6. **Retrieves relevant planning and conservation policy** via RAG over pre-embedded UK government documents
7. **Synthesises a structured report** with source citations and RED/AMBER/GREEN risk flags

All displayed in an interactive Streamlit app with a Folium map.

---

## Why It's Free to Run

No OpenAI API key. No cloud inference spend.

| Component | Technology | Cost |
|---|---|---|
| Agent orchestration | LangGraph | Free |
| Text LLM | Ollama + llama3.2:3b | Free (local) |
| Vision LLM | Ollama + LLaVA:7b | Free (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free (local) |
| Vector store | FAISS | Free |
| Geospatial data | EA, Natural England, OpenStreetMap | Free public APIs |
| UI | Streamlit | Free |

---

## Architecture
```
Input: UK Postcode + Optional Photo
         │
         ▼
┌─────────────────────────────────────────────┐
│           LangGraph Agent Graph              │
│                                             │
│  geocode → flood_risk → protected_areas     │
│                              │              │
│                          land_use           │
│                              │              │
│                   ┌──────────┴────────┐     │
│              vision (if photo)       rag    │
│                   └──────────┬────────┘     │
│                          synthesise         │
└──────────────────────────────┬─────────────┘
                               │
                               ▼
                    Structured Report (Streamlit)
```

---

## Quickstart

### Prerequisites
```bash
# Install Ollama from ollama.com then pull models
ollama pull llama3.2:3b
ollama pull llava:7b
```

### Install
```bash
git clone https://github.com/ssabeeth/geosight
cd geosight
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pyproj
```

### Build the document index (one-time, ~3 minutes)
```bash
python scripts/build_index.py
```

### Run
```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`. Enter a UK postcode. Click **Analyse Land**.

---

## Example Output

**Postcode: TQ13 8HH (Dartmoor, Devon)**
```
## 1. Location Overview
Gidleigh, West Devon, Devon, England

## 2. Flood Risk Assessment
GREEN — No active flood warnings within 5km.

## 3. Protected Designations & Ecological Constraints
AMBER — 2 designations found within 2km:
- North Dartmoor SSSI (Wildlife & Countryside Act 1981)
- DARTMOOR National Park

## 4. Land Use & Character
Woodland / forest. Natural features: Open water, Scrubland,
Woodland (Blackaton Wood). Waterways: North Teign, Forder Brook.

## 5. Planning Policy Context
Development within National Parks should be limited [NPPF p.55].
Sequential test applies for any flood risk zone development [NPPF p.50].
```

---

## Document Corpus

The RAG system is indexed over these public domain UK documents:

| Document | Source |
|---|---|
| National Planning Policy Framework (NPPF) 2023 | DLUHC |
| Environment Agency — Flood Risk Standing Advice | EA |
| Natural England — Green Infrastructure Framework | NE |
| DEFRA — England Biodiversity Strategy | DEFRA |

---

## Project Structure
```
geosight/
├── app.py                     Streamlit UI
├── geosight/
│   ├── agent.py               LangGraph agent graph
│   ├── tools/
│   │   ├── geocoder.py        Nominatim / OSM geocoding
│   │   ├── flood_risk.py      Environment Agency API
│   │   ├── protected_areas.py Natural England ArcGIS API
│   │   ├── land_use.py        Overpass / OpenStreetMap
│   │   └── vision.py          LLaVA via Ollama
│   └── rag/
│       ├── retriever.py       FAISS + sentence-transformers
│       └── index/             Pre-built index
└── scripts/
    └── build_index.py         One-off index builder
```

---

## Data Sources

- **Flood data**: © Environment Agency. Open Government Licence v3.0.
- **Protected areas**: © Natural England. Open Government Licence v3.0.
- **Land use**: © OpenStreetMap contributors. ODbL licence.
- **Policy documents**: © Crown Copyright. Open Government Licence v3.0.

---

## Author

**Syed Sabeeth Shoeb** — AI/ML Engineer

[LinkedIn](https://linkedin.com/in/syed-sabeeth) · [GitHub](https://github.com/ssabeeth)

---

## Licence

MIT