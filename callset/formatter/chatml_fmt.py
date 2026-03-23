from __future__ import annotations

import json

from callset.models import ToolDef


def format_chatml(conversation: dict, tools: list[ToolDef]) -> dict:
    """Format a conversation in ChatML format with tool tokens."""
    messages = conversation.get("messages", [])
    parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "tool":
            # Tool results rendered as a special tool role
            tool_call_id = msg.get("tool_call_id", "")
            parts.append(f"<|im_start|>tool ({tool_call_id})\n{content}<|im_end|>")
        elif role == "assistant" and tool_calls:
            tc_blocks = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args)
                tc_blocks.append(json.dumps({"name": name, "arguments": json.loads(args)}))
            text = content or ""
            if text:
                text += "\n"
            text += "\n".join(f"[tool_call] {block}" for block in tc_blocks)
            parts.append(f"<|im_start|>assistant\n{text}<|im_end|>")
        else:
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    return {"text": "\n".join(parts)}
