from __future__ import annotations

import json

from callset.models import ToolDef


def format_hermes(conversation: dict, tools: list[ToolDef]) -> dict:
    """Format a conversation in Hermes XML tag format."""
    messages = conversation.get("messages", [])

    # Build tools XML block
    tools_defs = []
    for t in tools:
        tools_defs.append({
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        })
    tools_xml = f"<tools>\n{json.dumps(tools_defs, indent=2)}\n</tools>"

    parts = [tools_xml, ""]

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tc_parts.append(
                        f"<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>"
                    )
                text = content or ""
                if text:
                    text += "\n"
                text += "\n".join(tc_parts)
                parts.append(f"<|im_start|>assistant\n{text}<|im_end|>")
            else:
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        elif role == "tool":
            parts.append(f"<tool_response>\n{content}\n</tool_response>")

    return {"text": "\n".join(parts)}
