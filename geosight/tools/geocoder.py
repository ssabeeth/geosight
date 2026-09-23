"""
Geocoding tool — converts a UK postcode to coordinates.

postcodes.io (ONS postcode centroids, free, no key) is tried first. Nominatim is the
fallback: it rate-limits shared hosting such as Streamlit Community Cloud, where it
answered the live demo with HTTP 429.
"""

import requests
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

POSTCODES_IO = "https://api.postcodes.io/postcodes/"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "GeoSight/0.2 (+https://github.com/ssabeeth/geosight)"}


class GeoLocation(BaseModel):
    postcode: str
    lat: float
    lon: float
    display_name: str
    county: str | None = None
    country: str = "United Kingdom"


class PostcodeNotFound(ValueError):
    pass


# Retry network errors only; an unknown postcode won't improve on a second try.
_retry_network = retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)


@_retry_network
def _from_postcodes_io(postcode: str) -> GeoLocation:
    resp = requests.get(POSTCODES_IO + requests.utils.quote(postcode), headers=HEADERS, timeout=10)
    if resp.status_code == 404:
        raise PostcodeNotFound(f"Could not geocode postcode: {postcode!r}")
    resp.raise_for_status()
    r = resp.json()["result"]
    parish = r.get("parish") or ""
    place = r.get("admin_ward") if "unparished" in parish or not parish else parish
    parts = [r["postcode"], place, r.get("admin_district"), r.get("admin_county"), r.get("region"), r.get("country")]
    names = list(dict.fromkeys(p for p in parts if p))  # drop blanks and repeats, keep order
    return GeoLocation(
        postcode=r["postcode"],
        lat=r["latitude"],
        lon=r["longitude"],
        display_name=", ".join(names),
        county=r.get("admin_county") or r.get("admin_district"),
    )


@_retry_network
def _from_nominatim(postcode: str) -> GeoLocation:
    params = {
        "q": postcode,
        "countrycodes": "gb",
        "format": "json",
        "addressdetails": 1,
        "limit": 1,
    }
    response = requests.get(NOMINATIM, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    results = response.json()

    if not results:
        raise PostcodeNotFound(f"Could not geocode postcode: {postcode!r}")

    r = results[0]
    addr = r.get("address", {})

    return GeoLocation(
        postcode=postcode.upper().strip(),
        lat=float(r["lat"]),
        lon=float(r["lon"]),
        display_name=r.get("display_name", ""),
        county=addr.get("county") or addr.get("state_district"),
    )


def geocode_postcode(postcode: str) -> GeoLocation:
    postcode = postcode.strip()
    try:
        return _from_postcodes_io(postcode)
    except PostcodeNotFound:
        raise
    except requests.RequestException:
        return _from_nominatim(postcode)
