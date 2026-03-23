from __future__ import annotations

import json

from callset.models import ToolDef


def format_raw(conversation: dict, tools: list[ToolDef]) -> dict:
    """Format a conversation in minimal raw format."""
    messages = conversation.get("messages", [])
    output_messages = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            # Inline tool calls into content
            tc_text = json.dumps([
                {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", "{}")}
                for tc in tool_calls
            ])
            combined = f"{content}\n[tool_calls: {tc_text}]" if content else f"[tool_calls: {tc_text}]"
            output_messages.append({"role": role, "content": combined})
        else:
            output_messages.append({"role": role, "content": content})

    return {"messages": output_messages}
