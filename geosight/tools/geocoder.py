"""Geocoding tool — converts UK postcode to coordinates via Nominatim (free, no key)."""

import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from pydantic import BaseModel


class GeoLocation(BaseModel):
    postcode: str
    lat: float
    lon: float
    display_name: str
    county: str | None = None
    country: str = "United Kingdom"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def geocode_postcode(postcode: str) -> GeoLocation:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": postcode,
        "countrycodes": "gb",
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
    }
    headers = {"User-Agent": "GeoSight/0.1 (github.com/ssabeeth/geosight)"}

    response = requests.get(url, params=params, headers=headers, timeout=10)
    response.raise_for_status()
    results = response.json()

    if not results:
        raise ValueError(f"Could not geocode postcode: {postcode!r}")

    r = results[0]
    addr = r.get("address", {})

    return GeoLocation(
        postcode=postcode.upper().strip(),
        lat=float(r["lat"]),
        lon=float(r["lon"]),
        display_name=r.get("display_name", ""),
        county=addr.get("county") or addr.get("state_district"),
    )