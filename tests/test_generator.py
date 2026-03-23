import json

import pytest

from callset.generator.providers import LLMProvider
from callset.generator.seeds import generate_seeds
from callset.generator.conversations import generate_conversation
from callset.models import APIContext, ExampleType, ToolDef


class MockProvider(LLMProvider):
    """A mock LLM provider that returns canned responses."""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-model"

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = self._responses[-1] if self._responses else "[]"
        self._call_count += 1
        return resp


def _make_context():
    tools = [
        ToolDef(
            name="search_flights",
            description="Search flights",
            parameters={
                "type": "object",
                "properties": {
                    "departure": {"type": "string"},
                    "arrival": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["departure", "arrival", "date"],
            },
            required_params=["departure", "arrival", "date"],
            response_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            tags=["flights"],
        ),
    ]
    return APIContext(
        domain="Travel",
        description="A travel API",
        tools=tools,
        tool_chains=[],
    )


class TestGenerateSeeds:
    def test_parses_json_array(self):
        provider = MockProvider([
            json.dumps(["Search NYC to LAX", "Find cheap flights"]),
            json.dumps(["Book then cancel"]),
            json.dumps(["Ask about hotel"]),
            json.dumps(["Search past date"]),
            json.dumps(["Ask about weather"]),
        ])
        dist = {
            ExampleType.HAPPY: 50,
            ExampleType.MULTI_STEP: 20,
            ExampleType.CLARIFICATION: 10,
            ExampleType.ERROR: 10,
            ExampleType.REFUSAL: 10,
        }
        seeds = generate_seeds(_make_context(), dist, 10, provider)
        assert len(seeds) > 0
        assert all("type" in s and "scenario" in s for s in seeds)

    def test_handles_markdown_fences(self):
        provider = MockProvider([
            '```json\n["scenario 1", "scenario 2"]\n```',
            '```json\n["multi"]\n```',
            '```json\n["clarify"]\n```',
            '```json\n["error"]\n```',
            '```json\n["refuse"]\n```',
        ])
        dist = {
            ExampleType.HAPPY: 50,
            ExampleType.MULTI_STEP: 20,
            ExampleType.CLARIFICATION: 10,
            ExampleType.ERROR: 10,
            ExampleType.REFUSAL: 10,
        }
        seeds = generate_seeds(_make_context(), dist, 10, provider)
        assert len(seeds) > 0


class TestGenerateConversation:
    def test_parses_valid_conversation(self):
        conversation = {
            "messages": [
                {"role": "system", "content": "You are a travel assistant."},
                {"role": "user", "content": "Find flights"},
                {
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"departure": "JFK", "arrival": "LAX", "date": "2026-03-15"}'
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"id": "FL123"}'},
                {"role": "assistant", "content": "Found flight FL123."},
            ]
        }
        provider = MockProvider([json.dumps(conversation)])
        seed = {"type": ExampleType.HAPPY, "scenario": "Search for flights"}
        result = generate_conversation(seed, _make_context(), provider)
        assert "messages" in result
        assert len(result["messages"]) == 5
