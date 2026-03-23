import pytest

from callset.models import ToolDef, ValidationResult
from callset.validator import validate_conversation
from callset.validator.format import validate_format
from callset.validator.schema import validate_schema
from callset.validator.semantic import validate_semantic


SAMPLE_TOOLS = [
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
    ),
]


def _valid_conversation():
    return {
        "messages": [
            {"role": "system", "content": "You are a travel assistant."},
            {"role": "user", "content": "Find flights from NYC to LAX on March 15"},
            {
                "role": "assistant",
                "content": "Let me search for flights.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"departure": "JFK", "arrival": "LAX", "date": "2026-03-15"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"flights": [{"airline": "Delta", "price": 350}]}',
            },
            {
                "role": "assistant",
                "content": "I found a Delta flight for $350.",
            },
        ]
    }


class TestFormatValidator:
    def test_valid(self):
        result = validate_format(_valid_conversation())
        assert result.valid

    def test_missing_messages(self):
        result = validate_format({"data": []})
        assert not result.valid

    def test_invalid_role(self):
        conv = {"messages": [{"role": "alien", "content": "hi"}]}
        result = validate_format(conv)
        assert not result.valid

    def test_invalid_arguments_json(self):
        conv = _valid_conversation()
        conv["messages"][2]["tool_calls"][0]["function"]["arguments"] = "not json{{"
        result = validate_format(conv)
        assert not result.valid

    def test_tool_calls_only_on_assistant(self):
        conv = {"messages": [{"role": "user", "content": "hi", "tool_calls": []}]}
        result = validate_format(conv)
        assert not result.valid


class TestSchemaValidator:
    def test_valid(self):
        result = validate_schema(_valid_conversation(), SAMPLE_TOOLS)
        assert result.valid

    def test_unknown_tool(self):
        conv = _valid_conversation()
        conv["messages"][2]["tool_calls"][0]["function"]["name"] = "nonexistent"
        result = validate_schema(conv, SAMPLE_TOOLS)
        assert not result.valid
        assert any("unknown tool" in e for e in result.errors)

    def test_missing_required_param(self):
        conv = _valid_conversation()
        conv["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"departure": "JFK"}'
        result = validate_schema(conv, SAMPLE_TOOLS)
        assert not result.valid

    def test_extra_param(self):
        conv = _valid_conversation()
        conv["messages"][2]["tool_calls"][0]["function"]["arguments"] = (
            '{"departure": "JFK", "arrival": "LAX", "date": "2026-03-15", "extra": "bad"}'
        )
        result = validate_schema(conv, SAMPLE_TOOLS)
        assert not result.valid

    def test_wrong_type(self):
        conv = _valid_conversation()
        conv["messages"][2]["tool_calls"][0]["function"]["arguments"] = (
            '{"departure": 123, "arrival": "LAX", "date": "2026-03-15"}'
        )
        result = validate_schema(conv, SAMPLE_TOOLS)
        assert not result.valid


class TestSemanticValidator:
    def test_valid_happy(self):
        result = validate_semantic(_valid_conversation(), "happy")
        assert result.valid

    def test_tool_before_user(self):
        conv = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
                },
                {"role": "user", "content": "hi"},
            ]
        }
        result = validate_semantic(conv, "happy")
        assert not result.valid

    def test_refusal_with_tool_calls(self):
        conv = _valid_conversation()
        result = validate_semantic(conv, "refusal")
        assert not result.valid
        assert any("Refusal" in e for e in result.errors)

    def test_clarification_warns_if_no_question(self):
        # Assistant directly calls tool without asking first
        conv = {
            "messages": [
                {"role": "user", "content": "Book a flight"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
                },
            ]
        }
        result = validate_semantic(conv, "clarification")
        assert len(result.warnings) > 0


class TestCombinedValidator:
    def test_valid_conversation(self):
        result = validate_conversation(_valid_conversation(), SAMPLE_TOOLS, "happy")
        assert result.valid

    def test_strict_mode_promotes_warnings(self):
        conv = _valid_conversation()
        # Remove tool_call_id to trigger a warning
        del conv["messages"][3]["tool_call_id"]
        result = validate_conversation(conv, SAMPLE_TOOLS, "happy", strict=True)
        assert not result.valid
