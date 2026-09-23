"""
Protected areas tool — queries Natural England ArcGIS REST APIs.
Free, no API key required. Returns SSSIs, National Landscapes (AONBs), NNRs and
National Parks that cover the site or lie within a true radius of it.
"""

import requests
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

_NE = "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services"

DESIGNATION_SOURCES = {
    "SSSI": {
        "url": f"{_NE}/SSSI_England/FeatureServer/0/query",
        "label": "Site of Special Scientific Interest (SSSI)",
        "significance": "High — legal protection under Wildlife & Countryside Act 1981",
    },
    "AONB": {
        "url": f"{_NE}/Areas_of_Outstanding_Natural_Beauty_England/FeatureServer/0/query",
        "label": "National Landscape (formerly AONB)",
        "significance": "High — landscape protection, planning constraints apply",
    },
    "NNR": {
        "url": f"{_NE}/National_Nature_Reserves_England/FeatureServer/0/query",
        "label": "National Nature Reserve (NNR)",
        "significance": "Very High — managed for nature conservation",
    },
    "National Park": {
        "url": f"{_NE}/National_Parks_England/FeatureServer/0/query",
        "label": "National Park",
        "significance": "High — strong planning restrictions",
    },
}


class Designation(BaseModel):
    type: str
    name: str
    label: str
    significance: str
    on_site: bool = False  # the site point falls inside it


class ProtectedAreasResult(BaseModel):
    lat: float
    lon: float
    radius_km: float
    designations: list[Designation] = []
    unavailable: list[str] = []  # designation types that could not be checked
    summary: str = ""


@retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
def _query_names(url: str, lat: float, lon: float, distance_m: float) -> set[str]:
    """Names of features within distance_m of the point (0 = covering the point)."""
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "NAME",
        "returnGeometry": "false",
        "f": "json",
    }
    if distance_m:
        params |= {"distance": distance_m, "units": "esriSRUnit_Meter"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:  # ArcGIS reports errors with HTTP 200
        raise RuntimeError(body["error"].get("message", "ArcGIS error"))
    names = {f.get("attributes", {}).get("NAME") or "Unnamed" for f in body.get("features", [])}
    return {n.title() if n.isupper() else n for n in names}  # "DARTMOOR" -> "Dartmoor"


def fetch_protected_areas(lat: float, lon: float, radius_km: float = 2.0) -> ProtectedAreasResult:
    found: list[Designation] = []
    unavailable: list[str] = []

    for key, config in DESIGNATION_SOURCES.items():
        try:
            nearby = _query_names(config["url"], lat, lon, radius_km * 1000)
            on_site = _query_names(config["url"], lat, lon, 0) if nearby else set()
        except Exception:
            unavailable.append(key)
            continue
        for name in sorted(nearby):
            found.append(Designation(
                type=key,
                name=name,
                label=config["label"],
                significance=config["significance"],
                on_site=name in on_site,
            ))

    if len(unavailable) == len(DESIGNATION_SOURCES):
        raise RuntimeError("every Natural England designation service failed")

    if not found:
        summary = f"No statutory designations found within {radius_km:g} km."
    else:
        types = sorted({d.type for d in found})
        inside = [d.name for d in found if d.on_site]
        summary = f"{len(found)} designation(s) within {radius_km:g} km: {', '.join(types)}."
        if inside:
            summary += f" The site lies inside: {', '.join(inside)}."
    if unavailable:
        summary += f" Could not check: {', '.join(unavailable)}."

    return ProtectedAreasResult(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        designations=found,
        unavailable=unavailable,
        summary=summary,
    )
