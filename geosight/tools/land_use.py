"""
Land use tool — queries OpenStreetMap via Overpass API.
Free, no key required. Returns land use, natural features, and waterways nearby.
"""

import requests
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

LAND_USE_LABELS = {
    "farmland": "Agricultural farmland",
    "forest": "Woodland / forest",
    "grass": "Grassland",
    "meadow": "Meadow",
    "orchard": "Orchard",
    "allotments": "Allotments",
    "residential": "Residential area",
    "commercial": "Commercial area",
    "industrial": "Industrial area",
    "retail": "Retail area",
    "conservation": "Conservation area",
    "nature_reserve": "Nature reserve",
    "recreation_ground": "Recreation ground",
    "village_green": "Village green",
    "brownfield": "Brownfield / previously developed",
    "greenfield": "Greenfield",
}

NATURAL_LABELS = {
    "wood": "Woodland",
    "water": "Open water",
    "wetland": "Wetland",
    "heath": "Heathland",
    "scrub": "Scrubland",
    "grassland": "Grassland",
    "cliff": "Cliff / escarpment",
    "beach": "Beach / coastal",
    "mud": "Mudflat / estuary",
}


class LandUseResult(BaseModel):
    lat: float
    lon: float
    radius_m: int
    land_uses: list[str] = []
    natural_features: list[str] = []
    waterways: list[str] = []
    summary: str = ""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_land_use(lat: float, lon: float, radius_m: int = 500) -> LandUseResult:
    """
    Query OpenStreetMap Overpass API for land use and natural features near a point.
    """
    query = f"""
    [out:json][timeout:25];
    (
      way["landuse"](around:{radius_m},{lat},{lon});
      way["natural"](around:{radius_m},{lat},{lon});
      way["waterway"](around:{radius_m},{lat},{lon});
      relation["landuse"](around:{radius_m},{lat},{lon});
    );
    out tags;
    """

    resp = requests.post(OVERPASS_URL, data={"data": query}, timeout=30)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

    land_uses: set[str] = set()
    natural_features: set[str] = set()
    waterways: set[str] = set()

    for el in elements:
        tags = el.get("tags", {})
        lu = tags.get("landuse")
        nat = tags.get("natural")
        ww = tags.get("waterway")
        name = tags.get("name")

        if lu and lu in LAND_USE_LABELS:
            label = LAND_USE_LABELS[lu]
            if name:
                label = f"{label} ({name})"
            land_uses.add(label)

        if nat and nat in NATURAL_LABELS:
            label = NATURAL_LABELS[nat]
            if name:
                label = f"{label} ({name})"
            natural_features.add(label)

        if ww:
            ww_name = name or ww.replace("_", " ").title()
            waterways.add(ww_name)

    if not land_uses and not natural_features:
        summary = f"No detailed land use data found within {radius_m}m."
    else:
        parts = []
        if land_uses:
            parts.append(f"Land use: {', '.join(sorted(land_uses))}")
        if natural_features:
            parts.append(f"Natural features: {', '.join(sorted(natural_features))}")
        if waterways:
            parts.append(f"Waterways: {', '.join(sorted(waterways))}")
        summary = ". ".join(parts) + "."

    return LandUseResult(
        lat=lat,
        lon=lon,
        radius_m=radius_m,
        land_uses=sorted(land_uses),
        natural_features=sorted(natural_features),
        waterways=sorted(waterways),
        summary=summary,
    )