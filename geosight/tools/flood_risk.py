"""
Flood risk tool — Environment Agency Real Time Flood Monitoring API.
Free, no key required.
"""

import requests
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

EA_BASE = "https://environment.data.gov.uk/flood-monitoring"


class FloodRiskResult(BaseModel):
    lat: float
    lon: float
    nearby_stations: list[dict] = []
    severe_warnings: int = 0
    warnings: int = 0
    alerts: int = 0
    summary: str = ""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_flood_risk(lat: float, lon: float, radius_km: float = 5.0) -> FloodRiskResult:
    """
    Fetch flood warnings and monitoring stations near a location.
    Uses the EA's open real-time flood monitoring API.
    """
    headers = {"Accept": "application/json"}

    # Fetch active flood warnings within radius
    warnings_resp = requests.get(
        f"{EA_BASE}/id/floods",
        params={"lat": lat, "long": lon, "dist": radius_km, "_limit": 20},
        headers=headers,
        timeout=10,
    )
    warnings_resp.raise_for_status()
    warnings_data = warnings_resp.json().get("items", [])

    severe = sum(1 for w in warnings_data if w.get("severityLevel") == 1)
    warn = sum(1 for w in warnings_data if w.get("severityLevel") == 2)
    alert = sum(1 for w in warnings_data if w.get("severityLevel") == 3)

    # Fetch nearby monitoring stations
    stations_resp = requests.get(
        f"{EA_BASE}/id/stations",
        params={"lat": lat, "long": lon, "dist": radius_km, "_limit": 5},
        headers=headers,
        timeout=10,
    )
    stations_resp.raise_for_status()
    stations = stations_resp.json().get("items", [])

    nearby = [
        {
            "name": s.get("label", "Unknown"),
            "river": s.get("riverName", "N/A"),
            "type": s.get("stationType", "N/A"),
        }
        for s in stations[:3]
    ]

    # Summarise
    if severe > 0:
        summary = f"⚠️ SEVERE: {severe} severe flood warning(s) active within {radius_km}km."
    elif warn > 0:
        summary = f"⚠️ WARNING: {warn} flood warning(s) active within {radius_km}km."
    elif alert > 0:
        summary = f"ℹ️ ALERT: {alert} flood alert(s) in force within {radius_km}km."
    else:
        summary = f"✅ No active flood warnings or alerts within {radius_km}km."

    return FloodRiskResult(
        lat=lat,
        lon=lon,
        nearby_stations=nearby,
        severe_warnings=severe,
        warnings=warn,
        alerts=alert,
        summary=summary,
    )