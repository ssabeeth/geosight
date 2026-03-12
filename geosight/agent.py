"""
GeoSight LangGraph Agent
Orchestrates: geocode → flood risk → protected areas → land use → synthesise
"""

import os
from typing import TypedDict, Annotated
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from geosight.tools.geocoder import geocode_postcode, GeoLocation
from geosight.tools.flood_risk import fetch_flood_risk, FloodRiskResult
from geosight.tools.protected_areas import fetch_protected_areas, ProtectedAreasResult
from geosight.tools.land_use import fetch_land_use, LandUseResult
from geosight.tools.vision import describe_land_image, VisionResult
from geosight.rag.retriever import retrieve, RAGResult


# ---------------------------------------------------------------------------
# State — this is the data that flows through the entire pipeline
# ---------------------------------------------------------------------------

class GeoSightState(TypedDict):
    # Inputs
    postcode: str
    rag: RAGResult | None
    # Intermediate results
    location: GeoLocation | None
    flood_risk: FloodRiskResult | None
    protected_areas: ProtectedAreasResult | None
    land_use: LandUseResult | None
    image_bytes: bytes | None
    vision: VisionResult | None
    # Output
    report: str
    errors: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=groq_key,
            model=os.getenv("GROQ_MODEL", "llama-3.2-3b-preview"),
            temperature=0.2,
        )
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_TEXT_MODEL", "llama3.2:3b"),
        temperature=0.2,
    )

# ---------------------------------------------------------------------------
# Nodes — one function per tool
# ---------------------------------------------------------------------------

def node_geocode(state: GeoSightState) -> dict:
    try:
        location = geocode_postcode(state["postcode"])
        return {"location": location, "errors": []}
    except Exception as e:
        return {"location": None, "errors": [f"Geocoding failed: {e}"]}


def node_flood_risk(state: GeoSightState) -> dict:
    if not state.get("location"):
        return {"flood_risk": None, "errors": []}
    try:
        result = fetch_flood_risk(state["location"].lat, state["location"].lon)
        return {"flood_risk": result, "errors": []}
    except Exception as e:
        return {"flood_risk": None, "errors": [f"Flood risk failed: {e}"]}


def node_protected_areas(state: GeoSightState) -> dict:
    if not state.get("location"):
        return {"protected_areas": None, "errors": []}
    try:
        result = fetch_protected_areas(state["location"].lat, state["location"].lon)
        return {"protected_areas": result, "errors": []}
    except Exception as e:
        return {"protected_areas": None, "errors": [f"Protected areas failed: {e}"]}


def node_land_use(state: GeoSightState) -> dict:
    if not state.get("location"):
        return {"land_use": None, "errors": []}
    try:
        result = fetch_land_use(state["location"].lat, state["location"].lon)
        return {"land_use": result, "errors": []}
    except Exception as e:
        return {"land_use": None, "errors": [f"Land use failed: {e}"]}
def node_rag(state: GeoSightState) -> dict:
    if not state.get("location"):
        return {"rag": None, "errors": []}
    try:
        loc = state["location"]
        pa = state.get("protected_areas")
        flood = state.get("flood_risk")

        query_parts = [f"land use planning policy for {loc.county or 'England'}"]
        if pa and pa.designations:
            types = list({d.type for d in pa.designations})
            query_parts.append(f"planning restrictions near {' and '.join(types)}")
        if flood and flood.severe_warnings > 0:
            query_parts.append("flood risk development planning")

        query = ". ".join(query_parts)
        result = retrieve(query)
        return {"rag": result, "errors": []}
    except Exception as e:
        return {"rag": None, "errors": [f"RAG retrieval failed: {e}"]}

def node_vision(state: GeoSightState) -> dict:
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {
            "vision": VisionResult(
                description="No image provided.",
                model_used="llava:7b",
                image_provided=False
            ),
            "errors": []
        }
    try:
        result = describe_land_image(image_path=None, image_bytes=image_bytes)
        return {"vision": result, "errors": []}
    except Exception as e:
        return {"vision": None, "errors": [f"Vision analysis failed: {e}"]}
def should_run_vision(state: GeoSightState) -> str:
    return "vision" if state.get("image_bytes") else "rag"
def node_synthesise(state: GeoSightState) -> dict:
    loc = state.get("location")
    flood = state.get("flood_risk")
    pa = state.get("protected_areas")
    lu = state.get("land_use")

    geo_section = f"Location: {loc.display_name}" if loc else "Location: unknown"
    flood_section = flood.summary if flood else "Flood risk data unavailable."
    pa_section = pa.summary if pa else "Protected areas data unavailable."
    if pa and pa.designations:
        desig_list = "\n".join(
            f"  - {d.name} ({d.label})"
            for d in pa.designations
        )
        pa_section += f"\n\nDesignations:\n{desig_list}"
    lu_section = lu.summary if lu else "Land use data unavailable."
    vis = state.get("vision")
    vis_section = vis.description if vis and vis.image_provided else "No image provided."

    rag = state.get("rag")
    rag_context = rag.context if rag else "Policy documents unavailable."
    rag_sources = (
        "\n".join(f"  [{i+1}] {c.source}, p.{c.page} (relevance: {c.score:.2f})"
                  for i, c in enumerate(rag.chunks))
        if rag else ""
    )

    prompt = f"""You are a senior land and planning analyst. Using the data below,
write a structured land intelligence report. Cite policy sources by number where relevant.

## DATA

{geo_section}

### Flood Risk
{flood_section}

### Protected Designations (within 2km)
{pa_section}

### Land Use (within 500m)
{lu_section}
### Site Photograph Analysis
{vis_section}

### Relevant Policy & Guidance
{rag_context}

## REQUIRED OUTPUT FORMAT

# Land Intelligence Report: {state['postcode'].upper()}

## 1. Location Overview
## 2. Flood Risk Assessment
## 3. Protected Designations & Ecological Constraints
## 4. Land Use & Character
## 5. Planning Policy Context
## 6. Key Considerations & Summary

## Policy Sources
{rag_sources}

Use RED/AMBER/GREEN ratings. Cite sources by number e.g. [1]. Be direct and specific.
"""

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"report": response.content, "errors": []}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GeoSightState)

    graph.add_node("geocode", node_geocode)
    graph.add_node("flood_risk", node_flood_risk)
    graph.add_node("protected_areas", node_protected_areas)
    graph.add_node("land_use", node_land_use)
    graph.add_node("vision", node_vision)
    graph.add_node("rag", node_rag)
    graph.add_node("synthesise", node_synthesise)

    graph.set_entry_point("geocode")
    graph.add_edge("geocode", "flood_risk")
    graph.add_edge("flood_risk", "protected_areas")
    graph.add_edge("protected_areas", "land_use")
    graph.add_conditional_edges(
        "land_use",
        should_run_vision,
        {"vision": "vision", "rag": "rag"},
    )
    graph.add_edge("vision", "rag")
    graph.add_edge("rag", "synthesise")
    graph.add_edge("synthesise", END)

    return graph.compile()
_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(postcode: str, image_bytes: bytes | None = None) -> GeoSightState:
    graph = get_graph()
    initial_state: GeoSightState = {
        "postcode": postcode,
        "image_bytes": image_bytes,
        "location": None,
        "flood_risk": None,
        "protected_areas": None,
        "land_use": None,
        "vision": None,
        "rag": None,
        "report": "",
        "errors": [],
    }
    return graph.invoke(initial_state)