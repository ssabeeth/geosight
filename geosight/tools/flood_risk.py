"""
Flood risk tool — two Environment Agency sources, both free with no key:

1. Flood Map for Planning (Rivers and Sea): the long-term flood zone the site sits in.
   This is what planning decisions use.
2. Real Time Flood Monitoring: warnings in force today, plus nearby river gauges.

"No warnings today" says nothing about long-term risk, so the two are kept apart.
"""

import requests
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

EA_MONITORING = "https://environment.data.gov.uk/flood-monitoring"
EA_FLOOD_ZONES = (
    "https://environment.data.gov.uk/geoservices/datasets/"
    "04532375-a198-476e-985e-0579a0a11b47/ogc/features/v1/collections/"
    "Flood_Zones_2_3_Rivers_and_Sea/items"
)

FLOOD_ZONE_MEANING = {
    1: "low probability: less than 0.1% a year from rivers or the sea",
    2: "medium probability: 0.1–1% a year from rivers, 0.1–0.5% from the sea",
    3: "high probability: 1% or more a year from rivers, 0.5% or more from the sea",
}


class FloodRiskResult(BaseModel):
    lat: float
    lon: float
    flood_zone: int | None = None  # 1, 2 or 3; None if the lookup failed
    flood_sources: list[str] = []  # e.g. ["river"], ["sea"]
    nearby_stations: list[dict] = []
    live_checked: bool = False
    severe_warnings: int = 0
    warnings: int = 0
    alerts: int = 0
    unavailable: list[str] = []  # parts that could not be fetched
    summary: str = ""


@retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
def _fetch_flood_zone(lat: float, lon: float) -> tuple[int, list[str]]:
    """Flood zone at the point. Zone 3 lies inside Zone 2, so take the highest hit."""
    d = 0.0001  # ~10 m box around the point
    resp = requests.get(
        EA_FLOOD_ZONES,
        params={
            "bbox": f"{lon - d},{lat - d},{lon + d},{lat + d}",
            "f": "application/json",
            "limit": 20,
        },
        timeout=30,
    )
    resp.raise_for_status()
    zone, sources = 1, set()
    for feat in resp.json().get("features", []):
        props = feat.get("properties", {})
        z = {"FZ2": 2, "FZ3": 3}.get(props.get("flood_zone"))
        if z:
            zone = max(zone, z)
            if props.get("flood_source"):
                sources.add(props["flood_source"])
    return zone, sorted(sources)


@retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
def _fetch_live(lat: float, lon: float, radius_km: float) -> tuple[list[dict], list[dict]]:
    headers = {"Accept": "application/json"}
    warnings_resp = requests.get(
        f"{EA_MONITORING}/id/floods",
        params={"lat": lat, "long": lon, "dist": radius_km, "_limit": 20},
        headers=headers,
        timeout=60,
    )
    warnings_resp.raise_for_status()
    stations_resp = requests.get(
        f"{EA_MONITORING}/id/stations",
        params={"lat": lat, "long": lon, "dist": radius_km, "_limit": 5},
        headers=headers,
        timeout=60,
    )
    stations_resp.raise_for_status()
    return warnings_resp.json().get("items", []), stations_resp.json().get("items", [])


def fetch_flood_risk(lat: float, lon: float, radius_km: float = 5.0) -> FloodRiskResult:
    """
    Flood zone at the site plus live warnings within radius_km.
    Raises only if both sources fail; a partial failure is listed in `unavailable`.
    """
    result = FloodRiskResult(lat=lat, lon=lon)
    errors = []

    try:
        result.flood_zone, result.flood_sources = _fetch_flood_zone(lat, lon)
    except Exception as e:
        result.unavailable.append("flood zone (EA Flood Map for Planning)")
        errors.append(e)

    try:
        warnings_data, stations = _fetch_live(lat, lon, radius_km)
        result.live_checked = True
        result.severe_warnings = sum(1 for w in warnings_data if w.get("severityLevel") == 1)
        result.warnings = sum(1 for w in warnings_data if w.get("severityLevel") == 2)
        result.alerts = sum(1 for w in warnings_data if w.get("severityLevel") == 3)
        result.nearby_stations = [
            {
                "name": s.get("label", "Unknown"),
                "river": s.get("riverName") or "",
                "type": s.get("stationType", "N/A"),
            }
            for s in stations[:3]
        ]
    except Exception as e:
        result.unavailable.append("live flood warnings (EA Real Time Flood Monitoring)")
        errors.append(e)

    if len(errors) == 2:
        raise RuntimeError(f"both flood sources failed: {errors[0]}; {errors[1]}")

    parts = []
    if result.flood_zone is not None:
        source = f" ({' and '.join(result.flood_sources)})" if result.flood_sources else ""
        parts.append(f"Site is in Flood Zone {result.flood_zone}{source}, {FLOOD_ZONE_MEANING[result.flood_zone]}.")
    else:
        parts.append("Flood zone: NOT AVAILABLE.")
    if result.live_checked:
        if result.severe_warnings:
            parts.append(f"{result.severe_warnings} severe flood warning(s) in force within {radius_km:g} km today.")
        elif result.warnings:
            parts.append(f"{result.warnings} flood warning(s) in force within {radius_km:g} km today.")
        elif result.alerts:
            parts.append(f"{result.alerts} flood alert(s) in force within {radius_km:g} km today.")
        else:
            parts.append(f"No live flood warnings or alerts within {radius_km:g} km today.")
    else:
        parts.append("Live warnings: NOT AVAILABLE.")
    result.summary = " ".join(parts)
    return result
