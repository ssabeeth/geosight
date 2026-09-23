import pytest
from tenacity import wait_none

from geosight.rag.retriever import RAGResult, RetrievedChunk
from geosight.tools import flood_risk, geocoder, land_use, protected_areas
from geosight.tools.flood_risk import FloodRiskResult
from geosight.tools.geocoder import GeoLocation
from geosight.tools.land_use import LandUseResult
from geosight.tools.protected_areas import Designation, ProtectedAreasResult

RETRYING = [
    flood_risk._fetch_flood_zone,
    flood_risk._fetch_live,
    protected_areas._query_names,
    land_use._overpass,
    geocoder._from_postcodes_io,
    geocoder._from_nominatim,
]


@pytest.fixture(autouse=True)
def no_retry_wait():
    """Keep tenacity's retries but skip the sleeps between them."""
    saved = [f.retry.wait for f in RETRYING]
    for f in RETRYING:
        f.retry.wait = wait_none()
    yield
    for f, wait in zip(RETRYING, saved, strict=True):
        f.retry.wait = wait


@pytest.fixture
def location():
    return GeoLocation(
        postcode="SP6 1EF", lat=50.9323, lon=-1.7827,
        display_name="SP6 1EF, Fordingbridge, New Forest, Hampshire", county="Hampshire",
    )


@pytest.fixture
def flood_zone3():
    return FloodRiskResult(
        lat=50.9323, lon=-1.7827, flood_zone=3, flood_sources=["river"], live_checked=True,
        nearby_stations=[{"name": "Fordingbridge", "river": "River Avon", "type": ""}],
        summary="Site is in Flood Zone 3. No live flood warnings or alerts within 5 km today.",
    )


@pytest.fixture
def designations_nearby():
    return ProtectedAreasResult(
        lat=50.9323, lon=-1.7827, radius_km=2.0,
        designations=[Designation(type="SSSI", name="River Avon System SSSI", label="SSSI", significance="High")],
        summary="1 designation(s) within 2 km: SSSI.",
    )


@pytest.fixture
def land_use_mixed():
    return LandUseResult(
        lat=50.9323, lon=-1.7827, radius_m=500,
        land_uses=["Residential area"], waterways=["River Avon"],
        summary="Land use: Residential area. Waterways: River Avon.",
    )


@pytest.fixture
def rag_two_sources():
    chunks = [
        RetrievedChunk(text="Paragraph 170. Inappropriate development in areas at risk of flooding should be avoided.",
                       source="NPPF", page=48, score=0.7),
        RetrievedChunk(text="A sequential test is needed in Flood Zones 2 and 3.",
                       source="Flood risk PPG", section="The sequential test", score=0.6),
    ]
    return RAGResult(queries=["q"], chunks=chunks,
                     context="\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1)))
