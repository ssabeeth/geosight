"""
Protected areas tool — queries Natural England / DEFRA ArcGIS REST APIs.
Free, no API key required. Returns SSSIs, AONBs, NNRs, and National Parks within radius.
"""

import json
import requests
from pydantic import BaseModel
from pyproj import Transformer
from tenacity import retry, stop_after_attempt, wait_fixed

_transformer = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True)

DESIGNATION_SOURCES = {
    "SSSI": {
        "url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/SSSI_England/FeatureServer/0/query",
        "label": "Site of Special Scientific Interest (SSSI)",
        "significance": "High — legal protection under Wildlife & Countryside Act 1981",
    },
    "AONB": {
        "url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/Areas_of_Outstanding_Natural_Beauty_England/FeatureServer/0/query",
        "label": "Area of Outstanding Natural Beauty (AONB)",
        "significance": "High — landscape protection, planning constraints apply",
    },
    "NNR": {
        "url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/National_Nature_Reserves_England/FeatureServer/0/query",
        "label": "National Nature Reserve (NNR)",
        "significance": "Very High — managed for nature conservation",
    },
    "National Park": {
        "url": "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/National_Parks_England/FeatureServer/0/query",
        "label": "National Park",
        "significance": "High — strong planning restrictions",
    },
}


class Designation(BaseModel):
    type: str
    name: str
    label: str
    significance: str


class ProtectedAreasResult(BaseModel):
    lat: float
    lon: float
    radius_km: float
    designations: list[Designation] = []
    summary: str = ""


def _build_envelope(easting: float, northing: float, radius_m: float) -> str:
    return json.dumps({
        "xmin": easting - radius_m,
        "ymin": northing - radius_m,
        "xmax": easting + radius_m,
        "ymax": northing + radius_m,
        "spatialReference": {"wkid": 27700}
    })


@retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
def _query_designations(url: str, envelope: str) -> list[dict]:
    resp = requests.get(
        url,
        params={
            'geometry': envelope,
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '27700',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': 'NAME',
            'returnGeometry': 'false',
            'f': 'json',
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get('features', [])


def fetch_protected_areas(lat: float, lon: float, radius_km: float = 2.0) -> ProtectedAreasResult:
    easting, northing = _transformer.transform(lon, lat)
    radius_m = radius_km * 1000
    envelope = _build_envelope(easting, northing, radius_m)
    found: list[Designation] = []

    for key, config in DESIGNATION_SOURCES.items():
        try:
            features = _query_designations(config["url"], envelope)
            for feat in features:
                name = feat.get("attributes", {}).get("NAME") or "Unnamed"
                found.append(Designation(
                    type=key,
                    name=name,
                    label=config["label"],
                    significance=config["significance"],
                ))
        except Exception:
            continue

    if not found:
        summary = f"✅ No statutory designations found within {radius_km}km."
    else:
        types = list({d.type for d in found})
        summary = (
            f"⚠️ {len(found)} protected designation(s) found within {radius_km}km: "
            f"{', '.join(types)}. Planning and land management activities may be restricted."
        )

    return ProtectedAreasResult(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        designations=found,
        summary=summary,
    )