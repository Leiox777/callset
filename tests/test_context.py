from callset.context import build_context
from callset.models import ToolDef


def _make_tools():
    return [
        ToolDef(
            name="search_flights",
            description="Search flights",
            parameters={"type": "object", "properties": {"date": {"type": "string"}}},
            required_params=["date"],
            response_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}, "airline": {"type": "string"}},
            },
            tags=["flights"],
        ),
        ToolDef(
            name="book_flight",
            description="Book a flight",
            parameters={
                "type": "object",
                "properties": {"flight_id": {"type": "string"}},
            },
            required_params=["flight_id"],
            tags=["flights"],
        ),
        ToolDef(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            required_params=["city"],
            tags=["weather"],
        ),
    ]


class TestBuildContext:
    def test_domain_from_title(self):
        ctx = build_context(_make_tools(), spec_title="SkyRoute API")
        assert ctx.domain == "SkyRoute API"

    def test_domain_from_tags(self):
        ctx = build_context(_make_tools())
        # "flights" appears twice, "weather" once
        assert ctx.domain == "flights"

    def test_description_from_spec(self):
        ctx = build_context(_make_tools(), spec_description="A travel API")
        assert ctx.description == "A travel API"

    def test_description_generated(self):
        ctx = build_context(_make_tools())
        assert "search_flights" in ctx.description

    def test_tool_chain_inference(self):
        ctx = build_context(_make_tools())
        # search_flights returns 'id', book_flight takes 'flight_id'
        # Should infer chain: search_flights → book_flight via flight_id
        chain_tuples = [(src, tgt) for src, tgt, _ in ctx.tool_chains]
        assert ("search_flights", "book_flight") in chain_tuples

    def test_no_self_chains(self):
        ctx = build_context(_make_tools())
        for src, tgt, _ in ctx.tool_chains:
            assert src != tgt
