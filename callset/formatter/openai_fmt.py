from __future__ import annotations

import json

from callset.models import ToolDef


def format_openai(conversation: dict, tools: list[ToolDef]) -> dict:
    """Format a conversation in OpenAI function calling format."""
    messages = conversation.get("messages", [])

    # Ensure tool_call arguments are JSON strings
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments")
            if isinstance(args, dict):
                func["arguments"] = json.dumps(args)

    # Build the tools array
    tools_array = []
    for t in tools:
        tools_array.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        })

    return {
        "messages": messages,
        "tools": tools_array,
    }
