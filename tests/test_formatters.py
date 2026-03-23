import json

from callset.formatter import format_conversation
from callset.formatter.openai_fmt import format_openai
from callset.formatter.hermes_fmt import format_hermes
from callset.formatter.chatml_fmt import format_chatml
from callset.formatter.raw_fmt import format_raw
from callset.models import ToolDef


SAMPLE_TOOLS = [
    ToolDef(
        name="search_flights",
        description="Search flights",
        parameters={
            "type": "object",
            "properties": {
                "departure": {"type": "string"},
                "arrival": {"type": "string"},
            },
        },
        required_params=["departure", "arrival"],
    ),
]


def _sample_conversation():
    return {
        "messages": [
            {"role": "system", "content": "You are a travel assistant."},
            {"role": "user", "content": "Find flights from NYC to LAX"},
            {
                "role": "assistant",
                "content": "Searching...",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"departure": "JFK", "arrival": "LAX"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"flights": []}'},
            {"role": "assistant", "content": "No flights found."},
        ]
    }


class TestOpenAIFormatter:
    def test_has_messages_and_tools(self):
        result = format_openai(_sample_conversation(), SAMPLE_TOOLS)
        assert "messages" in result
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "search_flights"

    def test_arguments_are_strings(self):
        # Test with dict arguments that should be stringified
        conv = _sample_conversation()
        conv["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"departure": "JFK", "arrival": "LAX"}
        result = format_openai(conv, SAMPLE_TOOLS)
        args = result["messages"][2]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"departure": "JFK", "arrival": "LAX"}


class TestHermesFormatter:
    def test_has_text_field(self):
        result = format_hermes(_sample_conversation(), SAMPLE_TOOLS)
        assert "text" in result

    def test_contains_xml_tags(self):
        result = format_hermes(_sample_conversation(), SAMPLE_TOOLS)
        text = result["text"]
        assert "<tools>" in text
        assert "</tools>" in text
        assert "<tool_call>" in text
        assert "<tool_response>" in text

    def test_contains_im_tokens(self):
        result = format_hermes(_sample_conversation(), SAMPLE_TOOLS)
        text = result["text"]
        assert "<|im_start|>system" in text
        assert "<|im_start|>user" in text
        assert "<|im_start|>assistant" in text


class TestChatMLFormatter:
    def test_has_text_field(self):
        result = format_chatml(_sample_conversation(), SAMPLE_TOOLS)
        assert "text" in result

    def test_contains_im_tokens(self):
        result = format_chatml(_sample_conversation(), SAMPLE_TOOLS)
        text = result["text"]
        assert "<|im_start|>" in text
        assert "<|im_end|>" in text

    def test_contains_tool_call_marker(self):
        result = format_chatml(_sample_conversation(), SAMPLE_TOOLS)
        assert "[tool_call]" in result["text"]


class TestRawFormatter:
    def test_has_messages(self):
        result = format_raw(_sample_conversation(), SAMPLE_TOOLS)
        assert "messages" in result

    def test_all_messages_have_role_and_content(self):
        result = format_raw(_sample_conversation(), SAMPLE_TOOLS)
        for msg in result["messages"]:
            assert "role" in msg
            assert "content" in msg

    def test_tool_calls_inlined(self):
        result = format_raw(_sample_conversation(), SAMPLE_TOOLS)
        assistant_with_tools = result["messages"][2]
        assert "[tool_calls:" in assistant_with_tools["content"]


class TestFormatDispatcher:
    def test_all_formats(self):
        for fmt in ("openai", "hermes", "chatml", "raw"):
            result = format_conversation(_sample_conversation(), SAMPLE_TOOLS, fmt)
            assert result is not None

    def test_unknown_format_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown format"):
            format_conversation(_sample_conversation(), SAMPLE_TOOLS, "unknown")
