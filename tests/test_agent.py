"""The whole graph with every tool and the model faked."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from geosight import agent


@pytest.fixture
def fake_tools(monkeypatch, location, flood_zone3, designations_nearby, rag_two_sources):
    monkeypatch.setattr(agent, "geocode_postcode", lambda postcode: location)
    monkeypatch.setattr(agent, "fetch_flood_risk", lambda lat, lon: flood_zone3)
    monkeypatch.setattr(agent, "fetch_protected_areas", lambda lat, lon: designations_nearby)
    monkeypatch.setattr(agent, "retrieve_many", lambda queries: rag_two_sources)

    def land_use_down(lat, lon):
        raise ConnectionError("Overpass unreachable")
    monkeypatch.setattr(agent, "fetch_land_use", land_use_down)


def _model(monkeypatch, text):
    monkeypatch.setattr(agent, "_get_llm", lambda: FakeListChatModel(responses=[text]))


def test_report_is_grounded_and_checked(monkeypatch, fake_tools):
    _model(monkeypatch, "## 2. Flood Risk Assessment\nFlood Zone 3 applies [1]. See also [9] and §6.")
    result = agent.run_agent("SP6 1EF")

    text = result["report"]
    assert "| Flood risk | 🔴 RED |" in text
    assert "Flood Zone 3 applies [1]." in text
    assert "[9]" not in text
    assert "- Land use: OpenStreetMap could not be reached." in text
    assert result["errors"] == [
        "Land use failed: Overpass unreachable",
        "Removed citations to sources that don't exist: [9]",
        "Report quotes paragraph numbers not found in the sources: 6",
    ]


def test_contradicting_the_rating_is_flagged(monkeypatch, fake_tools):
    _model(monkeypatch, "## 2. Flood Risk Assessment\nRisk is GREEN [1].")
    result = agent.run_agent("SP6 1EF")
    assert "Rating mismatch — Flood risk: model wrote GREEN, data says RED" in result["errors"]


def test_report_citing_nothing_is_flagged(monkeypatch, fake_tools):
    _model(monkeypatch, "## 1. Location Overview\nA town.")
    result = agent.run_agent("SP6 1EF")
    assert "The report text cites none of the retrieved policy sources" in result["errors"]


def test_model_failure_still_returns_the_checked_data(monkeypatch, fake_tools):
    def broken():
        raise RuntimeError("model_decommissioned")
    monkeypatch.setattr(agent, "_get_llm", broken)

    result = agent.run_agent("SP6 1EF")
    assert "| Flood risk | 🔴 RED |" in result["report"]
    assert "could not write this report" in result["report"]
    assert "Report synthesis failed: model_decommissioned" in result["errors"]


def test_unknown_postcode_stops_before_the_model(monkeypatch):
    def not_found(postcode):
        raise ValueError(f"Could not geocode postcode: {postcode!r}")
    monkeypatch.setattr(agent, "geocode_postcode", not_found)
    monkeypatch.setattr(agent, "_get_llm", lambda: pytest.fail("model should not be called"))

    result = agent.run_agent("ZZ99 9ZZ")
    assert "could not be found" in result["report"]
    assert result["errors"] == ["Geocoding failed: Could not geocode postcode: 'ZZ99 9ZZ'"]


def test_groq_is_used_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    assert agent.llm_name() == f"{agent.GROQ_MODEL} (Groq)"
    monkeypatch.delenv("GROQ_API_KEY")
    assert agent.llm_name() == f"{agent.OLLAMA_TEXT_MODEL} (Ollama)"
