"""Tool tests against mocked HTTP. `responses` refuses any request that isn't mocked."""

import base64
import json
from urllib.parse import parse_qs, urlparse

import pytest
import responses

from geosight.tools import flood_risk, land_use, protected_areas, vision
from geosight.tools.geocoder import geocode_postcode

NOMINATIM = "https://nominatim.openstreetmap.org/search"
FLOODS = f"{flood_risk.EA_MONITORING}/id/floods"
STATIONS = f"{flood_risk.EA_MONITORING}/id/stations"


# --- geocoder -------------------------------------------------------------------

@responses.activate
def test_geocode_parses_nominatim():
    responses.get(NOMINATIM, json=[{
        "lat": "50.9323", "lon": "-1.7827", "display_name": "SP6 1EF, Fordingbridge",
        "address": {"county": "Hampshire"},
    }])
    loc = geocode_postcode("sp6 1ef ")
    assert (loc.postcode, loc.lat, loc.lon, loc.county) == ("SP6 1EF", 50.9323, -1.7827, "Hampshire")


@responses.activate
def test_geocode_unknown_postcode_raises():
    responses.get(NOMINATIM, json=[])
    with pytest.raises(ValueError, match="Could not geocode"):
        geocode_postcode("ZZ99 9ZZ")


# --- flood risk -----------------------------------------------------------------

def _zones(*zones):
    return {"features": [{"properties": {"flood_zone": z, "flood_source": "river"}} for z in zones]}


def _mock_live(warnings=(), stations=()):
    responses.get(FLOODS, json={"items": [{"severityLevel": s} for s in warnings]})
    responses.get(STATIONS, json={"items": [{"label": n, "riverName": "River Avon"} for n in stations]})


@responses.activate
def test_flood_zone_takes_highest_zone():
    responses.get(flood_risk.EA_FLOOD_ZONES, json=_zones("FZ2", "FZ3"))
    _mock_live(stations=["Fordingbridge"])
    result = flood_risk.fetch_flood_risk(50.93, -1.78)
    assert result.flood_zone == 3
    assert result.flood_sources == ["river"]
    assert result.nearby_stations[0]["name"] == "Fordingbridge"
    assert "Flood Zone 3" in result.summary
    assert result.unavailable == []


@responses.activate
def test_no_flood_zone_features_means_zone_1():
    responses.get(flood_risk.EA_FLOOD_ZONES, json=_zones())
    _mock_live()
    result = flood_risk.fetch_flood_risk(50.63, -3.85)
    assert result.flood_zone == 1
    assert "No live flood warnings" in result.summary


@responses.activate
def test_live_warnings_are_counted_by_severity():
    responses.get(flood_risk.EA_FLOOD_ZONES, json=_zones())
    _mock_live(warnings=[1, 2, 2, 3, 4])
    result = flood_risk.fetch_flood_risk(51.0, -1.0)
    assert (result.severe_warnings, result.warnings, result.alerts) == (1, 2, 1)


@responses.activate
def test_flood_zone_failure_is_reported_not_hidden():
    responses.get(flood_risk.EA_FLOOD_ZONES, status=503)
    _mock_live()
    result = flood_risk.fetch_flood_risk(51.0, -1.0)
    assert result.flood_zone is None
    assert result.live_checked
    assert result.unavailable == ["flood zone (EA Flood Map for Planning)"]
    assert "Flood zone: NOT AVAILABLE" in result.summary


@responses.activate
def test_both_flood_sources_failing_raises():
    responses.get(flood_risk.EA_FLOOD_ZONES, status=503)
    responses.get(FLOODS, status=503)
    with pytest.raises(RuntimeError, match="both flood sources failed"):
        flood_risk.fetch_flood_risk(51.0, -1.0)


# --- protected areas -------------------------------------------------------------

def _arcgis(nearby: dict, on_site: dict, failing=()):
    """Callback answering each layer; queries with a distance are 'nearby', without are 'on site'."""
    def callback(request):
        params = parse_qs(urlparse(request.url).query)
        layer = next(k for k, v in protected_areas.DESIGNATION_SOURCES.items() if request.url.startswith(v["url"]))
        if layer in failing:
            return 200, {}, '{"error": {"code": 400, "message": "Invalid query"}}'
        names = (nearby if "distance" in params else on_site).get(layer, [])
        return 200, {}, json.dumps({"features": [{"attributes": {"NAME": n}} for n in names]})
    for config in protected_areas.DESIGNATION_SOURCES.values():
        responses.add_callback(responses.GET, config["url"], callback=callback)


@responses.activate
def test_designations_within_radius_and_on_site():
    _arcgis(nearby={"SSSI": ["North Dartmoor SSSI"], "National Park": ["DARTMOOR"]},
            on_site={"National Park": ["DARTMOOR"]})
    result = protected_areas.fetch_protected_areas(50.63, -3.85)
    by_name = {d.name: d for d in result.designations}
    assert by_name["Dartmoor"].on_site  # upper-case names are tidied
    assert not by_name["North Dartmoor SSSI"].on_site
    assert "The site lies inside: Dartmoor" in result.summary


@responses.activate
def test_designation_queries_use_a_true_radius():
    _arcgis(nearby={}, on_site={})
    protected_areas.fetch_protected_areas(50.63, -3.85, radius_km=2.0)
    params = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert params["geometryType"] == ["esriGeometryPoint"]
    assert params["distance"] == ["2000.0"]
    assert params["units"] == ["esriSRUnit_Meter"]


@responses.activate
def test_failed_designation_layer_is_listed():
    _arcgis(nearby={}, on_site={}, failing=("SSSI",))
    result = protected_areas.fetch_protected_areas(50.63, -3.85)
    assert result.unavailable == ["SSSI"]
    assert "Could not check: SSSI" in result.summary


@responses.activate
def test_every_designation_layer_failing_raises():
    _arcgis(nearby={}, on_site={}, failing=tuple(protected_areas.DESIGNATION_SOURCES))
    with pytest.raises(RuntimeError):
        protected_areas.fetch_protected_areas(50.63, -3.85)


# --- land use ---------------------------------------------------------------------

@responses.activate
def test_land_use_falls_back_to_mirror_and_identifies_itself():
    main, mirror = land_use.OVERPASS_URLS
    responses.post(main, status=406)
    responses.post(mirror, json={"elements": [
        {"tags": {"landuse": "farmland"}},
        {"tags": {"natural": "wood", "name": "Hyde Wood"}},
        {"tags": {"waterway": "river", "name": "River Avon"}},
        {"tags": {"landuse": "something_unmapped"}},
    ]})
    result = land_use.fetch_land_use(50.93, -1.78)
    assert result.land_uses == ["Agricultural farmland"]
    assert result.natural_features == ["Woodland (Hyde Wood)"]
    assert result.waterways == ["River Avon"]
    assert responses.calls[0].request.headers["User-Agent"].startswith("GeoSight/")


@responses.activate
def test_land_use_with_nothing_mapped():
    responses.post(land_use.OVERPASS_URLS[0], json={"elements": []})
    result = land_use.fetch_land_use(50.93, -1.78, radius_m=500)
    assert result.summary == "OpenStreetMap has no mapped land use within 500 m."


# --- vision ---------------------------------------------------------------------------

def test_vision_without_image_calls_nothing():
    result = vision.describe_land_image(image_path=None, image_bytes=None)
    assert result.image_provided is False


@pytest.mark.parametrize("header,mime", [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
    (b"\xff\xd8\xff\xe0", "image/jpeg"),
])
def test_mime_type_detection(header, mime):
    assert vision._mime_type(header + b"rest") == mime


def test_groq_vision_uses_configured_model_and_strips_thinking(monkeypatch):
    sent = {}

    class FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)
            message = type("M", (), {"content": "<think>hmm</think>\nFlat grassland with a hedge."})
            return type("R", (), {"choices": [type("C", (), {"message": message})]})

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})

    import groq
    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    result = vision.describe_land_image(image_path=None, image_bytes=b"\x89PNG....")
    assert sent["model"] == vision.GROQ_VISION_MODEL
    image_url = sent["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url == "data:image/png;base64," + base64.b64encode(b"\x89PNG....").decode()
    assert result.description == "Flat grassland with a hedge."
    assert result.model_used.endswith("(Groq)")
