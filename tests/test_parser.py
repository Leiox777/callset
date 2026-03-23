import json
from pathlib import Path

import pytest

from callset.parser.json_tools import parse_json_tools
from callset.parser.openapi import parse_openapi

FIXTURES = Path(__file__).parent / "fixtures"


class TestJsonToolsParser:
    def test_parse_valid_tools(self):
        tools = parse_json_tools(FIXTURES / "sample_tools.json")
        assert len(tools) == 3
        assert tools[0].name == "search_flights"
        assert tools[0].required_params == ["departure_city", "arrival_city", "date"]
        assert tools[0].optional_params == []
        assert tools[0].response_schema is not None

    def test_required_optional_split(self):
        tools = parse_json_tools(FIXTURES / "sample_tools.json")
        book = next(t for t in tools if t.name == "book_flight")
        assert "flight_id" in book.required_params
        assert "passenger_name" in book.required_params
        assert book.optional_params == []

    def test_tags(self):
        tools = parse_json_tools(FIXTURES / "sample_tools.json")
        assert tools[0].tags == ["flights"]
        assert tools[2].tags == ["bookings"]

    def test_missing_name_raises(self, tmp_path):
        bad = [{"description": "no name"}]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="missing 'name'"):
            parse_json_tools(p)

    def test_not_array_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"name": "oops"}))
        with pytest.raises(ValueError, match="JSON array"):
            parse_json_tools(p)


class TestOpenAPIParser:
    def test_parse_sample_spec(self):
        tools, metadata = parse_openapi(FIXTURES / "sample_openapi.yaml")
        assert len(tools) == 3
        assert metadata["title"] == "SkyRoute Travel API"

    def test_tool_names(self):
        tools, _ = parse_openapi(FIXTURES / "sample_openapi.yaml")
        names = {t.name for t in tools}
        assert names == {"search_flights", "book_flight", "get_booking"}

    def test_search_flights_params(self):
        tools, _ = parse_openapi(FIXTURES / "sample_openapi.yaml")
        search = next(t for t in tools if t.name == "search_flights")
        assert "departure" in search.required_params
        assert "arrival" in search.required_params
        assert "date" in search.required_params
        assert "max_stops" in search.optional_params

    def test_ref_resolution(self):
        tools, _ = parse_openapi(FIXTURES / "sample_openapi.yaml")
        search = next(t for t in tools if t.name == "search_flights")
        # Response schema should be resolved (no $ref remaining)
        assert search.response_schema is not None
        flights_items = search.response_schema["properties"]["flights"]["items"]
        assert "properties" in flights_items
        assert "id" in flights_items["properties"]

    def test_request_body_merged(self):
        tools, _ = parse_openapi(FIXTURES / "sample_openapi.yaml")
        book = next(t for t in tools if t.name == "book_flight")
        props = book.parameters["properties"]
        # Should have both path param and body params
        assert "flight_id" in props
        assert "passenger_name" in props
        assert "email" in props

    def test_tags(self):
        tools, _ = parse_openapi(FIXTURES / "sample_openapi.yaml")
        search = next(t for t in tools if t.name == "search_flights")
        assert search.tags == ["flights"]
