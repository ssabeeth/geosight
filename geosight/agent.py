"""
GeoSight LangGraph pipeline
Runs: geocode → flood risk → protected areas → land use → (vision) → rag → synthesise
The tool order is fixed; only the vision step is conditional.
"""

import operator
import os
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from geosight import report
from geosight.rag.retriever import RAGResult, retrieve_many
from geosight.tools.flood_risk import FloodRiskResult, fetch_flood_risk
from geosight.tools.geocoder import GeoLocation, geocode_postcode
from geosight.tools.land_use import LandUseResult, fetch_land_use
from geosight.tools.protected_areas import ProtectedAreasResult, fetch_protected_areas
from geosight.tools.vision import VisionResult, describe_land_image, vision_model_name

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

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3.2:3b")


def llm_name() -> str:
    if os.getenv("GROQ_API_KEY"):
        return f"{GROQ_MODEL} (Groq)"
    return f"{OLLAMA_TEXT_MODEL} (Ollama)"


def _get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=groq_key,
            model=GROQ_MODEL,
            temperature=0.2,
            # Reasoning models spend part of this budget thinking.
            max_tokens=4096,
        )
    return ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=OLLAMA_TEXT_MODEL,
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
        return {"flood_risk": result, "errors": [f"Flood risk: {u} unavailable" for u in result.unavailable]}
    except Exception as e:
        return {"flood_risk": None, "errors": [f"Flood risk failed: {e}"]}


def node_protected_areas(state: GeoSightState) -> dict:
    if not state.get("location"):
        return {"protected_areas": None, "errors": []}
    try:
        result = fetch_protected_areas(state["location"].lat, state["location"].lon)
        return {
            "protected_areas": result,
            "errors": [f"Protected areas: {u} could not be checked" for u in result.unavailable],
        }
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
        return {"rag": retrieve_many(report.policy_queries(state)), "errors": []}
    except Exception as e:
        return {"rag": None, "errors": [f"RAG retrieval failed: {e}"]}


def node_vision(state: GeoSightState) -> dict:
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {
            "vision": VisionResult(
                description="No image provided.",
                model_used=vision_model_name(),
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
    if not state.get("location"):
        return {
            "report": f"# Land Intelligence Report: {state['postcode'].upper()}\n\n"
                      "The postcode could not be found, so no data was fetched.",
            "errors": [],
        }

    ratings = report.ratings_for(state)
    rag = state.get("rag")
    n_sources = len(rag.chunks) if rag else 0
    errors = []

    try:
        response = _get_llm().invoke([HumanMessage(content=report.build_prompt(state, ratings))])
        body = response.content if isinstance(response.content, str) else str(response.content)
        written_by = llm_name()
    except Exception as e:
        body = (
            "The language model could not write this report, so only the checked data is shown.\n\n"
            f"Error: {e}"
        )
        written_by = "no model (synthesis failed)"
        errors.append(f"Report synthesis failed: {e}")

    body, dropped = report.check_citations(body, n_sources)
    if n_sources and written_by != "no model (synthesis failed)" and not report.cites_any(body):
        errors.append("The report text cites none of the retrieved policy sources")
    if dropped:
        errors.append(f"Removed citations to sources that don't exist: {', '.join(f'[{n}]' for n in dropped)}")
    unsupported = report.unsupported_paragraph_refs(body, rag.context if rag else "")
    if unsupported:
        errors.append(f"Report quotes paragraph numbers not found in the sources: {', '.join(unsupported)}")
    errors += [f"Rating mismatch — {p}" for p in report.rating_contradictions(body, ratings)]

    return {"report": report.assemble_report(state, ratings, body, written_by), "errors": errors}


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
