import pytest

from geosight import report
from geosight.tools.flood_risk import FloodRiskResult
from geosight.tools.protected_areas import Designation, ProtectedAreasResult


def _flood(**kw):
    return FloodRiskResult(lat=0, lon=0, live_checked=True, **kw)


def _pa(designations=(), unavailable=()):
    return ProtectedAreasResult(lat=0, lon=0, radius_km=2.0,
                                designations=list(designations), unavailable=list(unavailable))


def _designation(name, on_site=False):
    return Designation(type="SSSI", name=name, label="SSSI", significance="High", on_site=on_site)


# --- ratings ---------------------------------------------------------------------

@pytest.mark.parametrize("flood,expected", [
    (None, "NOT RATED"),
    (_flood(flood_zone=3), "RED"),
    (_flood(flood_zone=2), "AMBER"),
    (_flood(flood_zone=1), "GREEN"),
    (_flood(flood_zone=1, alerts=1), "AMBER"),
    (_flood(flood_zone=1, warnings=1), "RED"),
    # The old bug: "no warnings today" must never become GREEN on its own.
    (_flood(flood_zone=None), "NOT RATED"),
])
def test_flood_rating(flood, expected):
    assert report.rate_flood(flood).rating == expected


@pytest.mark.parametrize("pa,expected", [
    (None, "NOT RATED"),
    (_pa([_designation("Dartmoor", on_site=True)]), "RED"),
    (_pa([_designation("River Avon System SSSI")]), "AMBER"),
    (_pa(unavailable=["SSSI"]), "NOT RATED"),
    (_pa(), "GREEN"),
])
def test_designation_rating(pa, expected):
    assert report.rate_designations(pa).rating == expected


# --- prompt ------------------------------------------------------------------------

def test_prompt_marks_missing_data(location, flood_zone3, rag_two_sources):
    state = {"postcode": "SP6 1EF", "location": location, "flood_risk": flood_zone3,
             "protected_areas": None, "land_use": None, "image_bytes": None, "vision": None,
             "rag": rag_two_sources}
    prompt = report.build_prompt(state, report.ratings_for(state))
    land_use_section = prompt.split("### Land use")[1].split("###")[0]
    assert report.NOT_AVAILABLE in land_use_section
    assert "No photograph was provided" in prompt
    assert "between 1 and 2" in prompt
    assert "Flood risk: RED" in prompt
    assert "Not checked: surface water" in prompt


def test_prompt_marks_failed_photo_analysis(location, rag_two_sources):
    state = {"postcode": "X", "location": location, "image_bytes": b"img", "vision": None, "rag": rag_two_sources}
    photo_section = report.build_prompt(state, []).split("### Site photograph")[1].split("###")[0]
    assert report.NOT_AVAILABLE in photo_section


def test_policy_queries_follow_the_data(flood_zone3, designations_nearby, land_use_mixed):
    queries = report.policy_queries(
        {"flood_risk": flood_zone3, "protected_areas": designations_nearby, "land_use": land_use_mixed}
    )
    assert queries[0].startswith("development in Flood Zone 3")
    assert "Site of Special Scientific Interest" in queries[1]
    assert "existing settlements" in queries[2]


# --- checks on the model's text --------------------------------------------------------

def test_citations_outside_the_source_list_are_removed():
    text, dropped = report.check_citations("A [1]. B [7]. C [2, 9]. D [1-3].", n_sources=2)
    assert text == "A [1]. B . C [2]. D [1, 2]."
    assert dropped == [3, 7, 9]


def test_cites_any():
    assert report.cites_any("as required [2].")
    assert not report.cites_any("no citations here")


def test_paragraph_numbers_must_appear_in_the_sources(rag_two_sources):
    text = "See paragraph 170 and NPPF §6."
    assert report.unsupported_paragraph_refs(text, rag_two_sources.context) == ["6"]


def test_rating_contradiction_is_flagged():
    ratings = [report.Rating(topic="Flood risk", rating="RED", reason="", source="")]
    body = "## 1. Location\nFine.\n## 2. Flood Risk Assessment\nFlood risk is GREEN.\n"
    assert report.rating_contradictions(body, ratings) == ["Flood risk: model wrote GREEN, data says RED"]
    assert report.rating_contradictions(body.replace("GREEN", "RED"), ratings) == []


# --- assembly ------------------------------------------------------------------------------

def test_report_has_fixed_ratings_gaps_and_sources(location, flood_zone3, rag_two_sources):
    state = {"postcode": "sp6 1ef", "location": location, "flood_risk": flood_zone3,
             "protected_areas": None, "land_use": None, "image_bytes": None, "rag": rag_two_sources}
    text = report.assemble_report(state, report.ratings_for(state), "## 1. Location Overview\nText [1].",
                                  written_by="test-model")
    assert text.startswith("# Land Intelligence Report: SP6 1EF")
    assert "| Flood risk | 🔴 RED | Site is in Flood Zone 3 (river). |" in text
    assert "| Protected designations | ⚪ NOT RATED |" in text
    assert "- Land use: OpenStreetMap could not be reached." in text
    assert "**[1]** NPPF, p.48" in text
    assert '**[2]** Flood risk PPG, section "The sequential test"' in text
    assert "Prose written by test-model" in text
