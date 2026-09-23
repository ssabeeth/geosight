"""
Grounding for the written report.

The model writes the prose; everything a reader might take as fact is decided
here instead:
- RED/AMBER/GREEN ratings come from rules over the fetched data.
- Missing data is marked NOT AVAILABLE in the prompt, and listed under "Data gaps".
- The source list is built from what retrieval returned, and citation numbers
  in the model's text are checked against it.
"""

import re

from pydantic import BaseModel

from geosight.rag.retriever import RAGResult
from geosight.tools.flood_risk import FloodRiskResult
from geosight.tools.land_use import LandUseResult
from geosight.tools.protected_areas import ProtectedAreasResult
from geosight.tools.vision import VisionResult

NOT_AVAILABLE = "NOT AVAILABLE — this data could not be retrieved. Do not infer, estimate or describe it."
RATING_ICONS = {"RED": "🔴", "AMBER": "🟠", "GREEN": "🟢", "NOT RATED": "⚪"}


class Rating(BaseModel):
    topic: str
    rating: str  # RED, AMBER, GREEN or NOT RATED
    reason: str
    source: str


# ---------------------------------------------------------------------------
# Ratings — rules, not the model
# ---------------------------------------------------------------------------

def rate_flood(flood: FloodRiskResult | None) -> Rating:
    source = "Environment Agency"
    if flood is None:
        return Rating(
            topic="Flood risk", rating="NOT RATED", reason="Flood data could not be retrieved.", source=source,
        )
    if flood.severe_warnings or flood.warnings:
        return Rating(topic="Flood risk", rating="RED", reason="Flood warning in force nearby today.", source=source)
    if flood.flood_zone is None:
        return Rating(
            topic="Flood risk", rating="NOT RATED",
            reason="Flood zone lookup failed; live warnings alone say nothing about long-term risk.",
            source=source,
        )
    origin = f" ({' and '.join(flood.flood_sources)})" if flood.flood_sources else ""
    if flood.flood_zone == 3:
        return Rating(topic="Flood risk", rating="RED", reason=f"Site is in Flood Zone 3{origin}.", source=source)
    if flood.flood_zone == 2:
        return Rating(topic="Flood risk", rating="AMBER", reason=f"Site is in Flood Zone 2{origin}.", source=source)
    if flood.alerts:
        return Rating(
            topic="Flood risk", rating="AMBER",
            reason="Flood Zone 1, but a flood alert is in force nearby today.", source=source,
        )
    return Rating(
        topic="Flood risk", rating="GREEN",
        reason="Flood Zone 1 for rivers and sea. Surface water was not checked.", source=source,
    )


def rate_designations(pa: ProtectedAreasResult | None) -> Rating:
    source = "Natural England"
    topic = "Protected designations"
    if pa is None:
        return Rating(topic=topic, rating="NOT RATED", reason="Designation data could not be retrieved.", source=source)
    inside = [d.name for d in pa.designations if d.on_site]
    if inside:
        return Rating(topic=topic, rating="RED", reason=f"Site lies inside {', '.join(inside)}.", source=source)
    if pa.designations:
        return Rating(
            topic=topic, rating="AMBER",
            reason=f"{len(pa.designations)} within {pa.radius_km:g} km, none covering the site.", source=source,
        )
    if pa.unavailable:
        return Rating(
            topic=topic, rating="NOT RATED",
            reason=f"None found, but {', '.join(pa.unavailable)} could not be checked.", source=source,
        )
    return Rating(topic=topic, rating="GREEN", reason=f"None within {pa.radius_km:g} km.", source=source)


def ratings_for(state: dict) -> list[Rating]:
    return [rate_flood(state.get("flood_risk")), rate_designations(state.get("protected_areas"))]


# ---------------------------------------------------------------------------
# Retrieval queries — one per report topic
# ---------------------------------------------------------------------------

def policy_queries(state: dict) -> list[str]:
    flood: FloodRiskResult | None = state.get("flood_risk")
    pa: ProtectedAreasResult | None = state.get("protected_areas")
    lu: LandUseResult | None = state.get("land_use")

    queries = []
    if flood and flood.flood_zone in (2, 3):
        queries.append(f"development in Flood Zone {flood.flood_zone}: sequential test and exception test")
    else:
        queries.append("when a site-specific flood risk assessment is needed for development")

    types = {d.type for d in pa.designations} if pa else set()
    if "SSSI" in types:
        queries.append("development likely to harm a Site of Special Scientific Interest")
    if types & {"National Park", "AONB"}:
        queries.append("major development in National Parks and National Landscapes")
    if not types:
        queries.append("biodiversity net gain and protecting habitats in new development")

    uses = " ".join(lu.land_uses).lower() if lu else ""
    if "farmland" in uses:
        queries.append("best and most versatile agricultural land")
    elif "brownfield" in uses:
        queries.append("making effective use of brownfield land")
    elif any(w in uses for w in ("residential", "commercial", "retail", "industrial")):
        queries.append("making effective use of land in existing settlements")
    else:
        queries.append("development in the countryside and rural areas")
    return queries


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _flood_section(flood: FloodRiskResult | None) -> str:
    if flood is None:
        return NOT_AVAILABLE
    text = flood.summary
    if flood.nearby_stations:
        text += "\nNearby gauges: " + "; ".join(
            f"{s['name']} ({s['river']})" if s["river"] else s["name"] for s in flood.nearby_stations
        )
    return text + "\nNot checked: surface water flooding, groundwater flooding, historic flood records."


def _designations_section(pa: ProtectedAreasResult | None) -> str:
    if pa is None:
        return NOT_AVAILABLE
    text = pa.summary
    if pa.designations:
        text += "\n" + "\n".join(
            f"- {d.name} ({d.label}){' — covers the site' if d.on_site else ''}" for d in pa.designations
        )
    return text


def _land_use_section(lu: LandUseResult | None) -> str:
    if lu is None:
        return NOT_AVAILABLE
    if not (lu.land_uses or lu.natural_features or lu.waterways):
        return f"{lu.summary} Do not describe the land use beyond saying this."
    return lu.summary


def _photo_section(state: dict) -> str:
    vis: VisionResult | None = state.get("vision")
    if not state.get("image_bytes"):
        return (
            "No photograph was provided. "
            "Do not describe what the site looks like or claim any on-the-ground observation."
        )
    if vis is None or not vis.image_provided:
        return NOT_AVAILABLE
    return f"(Photo described by {vis.model_used})\n{vis.description}"


def build_prompt(state: dict, ratings: list[Rating]) -> str:
    loc = state["location"]
    rag: RAGResult | None = state.get("rag")
    policy = rag.context if rag and rag.chunks else NOT_AVAILABLE
    n_sources = len(rag.chunks) if rag else 0
    rating_lines = "\n".join(f"- {r.topic}: {r.rating} — {r.reason}" for r in ratings)

    return f"""You are a senior land and planning analyst writing a short site report.

RULES
1. Use only the facts in DATA. Do not add facts from general knowledge about the area.
2. Where DATA says NOT AVAILABLE, write "Not available" for that point and move on. Never guess.
3. The ratings below are fixed. Do not change them or add ratings of your own.
4. Cite policy only as [n], where n is between 1 and {n_sources}. Do not quote page,
   paragraph or section numbers yourself, and do not write a list of sources.
5. Live flood warnings are about today only. Never treat "no warnings" as low flood risk.

## DATA

### Location
{loc.display_name}

### Ratings (fixed)
{rating_lines}

### Flood risk
{_flood_section(state.get("flood_risk"))}

### Protected designations
{_designations_section(state.get("protected_areas"))}

### Land use within 500 m (OpenStreetMap)
{_land_use_section(state.get("land_use"))}

### Site photograph
{_photo_section(state)}

### Policy extracts
{policy}

## OUTPUT
Write these sections in Markdown, a short paragraph each. No title, no table.

## 1. Location Overview
## 2. Flood Risk Assessment
## 3. Protected Designations & Ecological Constraints
## 4. Land Use & Character
## 5. Planning Policy Context
## 6. Key Considerations
"""


# ---------------------------------------------------------------------------
# Checks on what the model wrote
# ---------------------------------------------------------------------------

_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_PARAGRAPH_REF = re.compile(r"(?:§|\bparagraphs?\s+|\bpara\.?\s*)(\d+)", re.IGNORECASE)


def _expand(group: str) -> list[int]:
    nums = []
    for part in re.split(r"\s*,\s*", group):
        bounds = re.split(r"\s*[–-]\s*", part)
        if len(bounds) == 2:
            lo, hi = int(bounds[0]), int(bounds[1])
            nums.extend(range(lo, hi + 1) if hi - lo < 20 else [lo, hi])
        else:
            nums.append(int(bounds[0]))
    return nums


def check_citations(text: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop citation numbers that are not in the source list. Returns (text, dropped)."""
    dropped: set[int] = set()

    def fix(match: re.Match) -> str:
        nums = _expand(match.group(1))
        good = [n for n in nums if 1 <= n <= n_sources]
        dropped.update(n for n in nums if not 1 <= n <= n_sources)
        return f"[{', '.join(map(str, good))}]" if good else ""

    return _CITATION.sub(fix, text), sorted(dropped)


def cites_any(text: str) -> bool:
    return bool(_CITATION.search(text))


def unsupported_paragraph_refs(text: str, context: str) -> list[str]:
    """Paragraph numbers the model quoted that appear nowhere in the retrieved text."""
    return sorted({n for n in _PARAGRAPH_REF.findall(text) if not re.search(rf"\b{n}\b", context)}, key=int)


def rating_contradictions(text: str, ratings: list[Rating]) -> list[str]:
    """Rating words in the model's flood or designation section that disagree with the rules."""
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    keys = {"Flood risk": "flood", "Protected designations": "designation"}
    problems = []
    for r in ratings:
        key = keys.get(r.topic)
        for section in sections:
            heading = section.split("\n", 1)[0].lower()
            if key and key in heading:
                used = set(re.findall(r"\b(RED|AMBER|GREEN)\b", section))
                wrong = used - {r.rating}
                if wrong:
                    problems.append(f"{r.topic}: model wrote {', '.join(sorted(wrong))}, data says {r.rating}")
    return problems


def data_gaps(state: dict) -> list[str]:
    gaps = []
    flood: FloodRiskResult | None = state.get("flood_risk")
    pa: ProtectedAreasResult | None = state.get("protected_areas")
    if flood is None:
        gaps.append("Flood risk: Environment Agency data could not be retrieved.")
    else:
        gaps += [f"Flood risk: {u} could not be retrieved." for u in flood.unavailable]
    if pa is None:
        gaps.append("Protected designations: Natural England data could not be retrieved.")
    elif pa.unavailable:
        gaps.append(f"Protected designations: could not check {', '.join(pa.unavailable)}.")
    if state.get("land_use") is None:
        gaps.append("Land use: OpenStreetMap could not be reached.")
    if state.get("image_bytes") and not (state.get("vision") and state["vision"].image_provided):
        gaps.append("Site photograph: image analysis failed.")
    rag: RAGResult | None = state.get("rag")
    if not rag or not rag.chunks:
        gaps.append("Planning policy: document retrieval failed.")
    gaps.append(
        "Not checked by GeoSight: surface water and groundwater flooding, heritage, contamination, planning history."
    )
    return gaps


# ---------------------------------------------------------------------------
# Final assembly
# ---------------------------------------------------------------------------

def ratings_table(ratings: list[Rating]) -> str:
    rows = "\n".join(
        f"| {r.topic} | {RATING_ICONS[r.rating]} {r.rating} | {r.reason} | {r.source} |" for r in ratings
    )
    return "| Topic | Rating | Why | Source |\n|---|---|---|---|\n" + rows


def sources_list(rag: RAGResult | None) -> str:
    if not rag or not rag.chunks:
        return "None retrieved."
    lines = []
    for i, c in enumerate(rag.chunks, 1):
        name = f"[{c.label}]({c.url})" if c.url else c.label
        lines.append(f"- **[{i}]** {name} — relevance {c.score:.2f}")
    return "\n".join(lines)


def assemble_report(state: dict, ratings: list[Rating], body: str, written_by: str) -> str:
    loc = state.get("location")
    gaps = "\n".join(f"- {g}" for g in data_gaps(state))
    return f"""# Land Intelligence Report: {state['postcode'].upper()}
{loc.display_name if loc else ''}

{ratings_table(ratings)}

{body.strip()}

## Data gaps
{gaps}

## Policy sources
{sources_list(state.get("rag"))}

*Prose written by {written_by}. The ratings, data gaps and source list are generated from the data, not by the model.*
"""
