from __future__ import annotations

import re

from callset.models import ValidationResult


def validate_semantic(conversation: dict, example_type: str) -> ValidationResult:
    """Lightweight semantic checks on a conversation."""
    errors: list[str] = []
    warnings: list[str] = []

    messages = conversation.get("messages", [])

    # Find indices of different message types
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    tool_call_indices = [i for i, m in enumerate(messages) if m.get("tool_calls")]
    tool_result_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]

    # Check: user message must precede any tool call
    if tool_call_indices and user_indices:
        if user_indices[0] > tool_call_indices[0]:
            errors.append("Tool call appears before any user message")
    elif tool_call_indices and not user_indices:
        errors.append("Tool calls present but no user messages")

    # Type-specific checks
    if example_type == "clarification":
        if len(user_indices) < 2:
            errors.append("Clarification example must have at least 2 user turns")
        # Assistant should have a non-tool-call message before any tool call
        assistant_no_tool = [
            i for i, m in enumerate(messages)
            if m.get("role") == "assistant" and not m.get("tool_calls")
        ]
        if tool_call_indices and assistant_no_tool:
            if assistant_no_tool[0] > tool_call_indices[0]:
                errors.append(
                    "Clarification example: assistant must ask a question before making tool calls"
                )
        elif tool_call_indices and not assistant_no_tool:
            errors.append(
                "Clarification example: assistant never asks a clarifying question"
            )

    if example_type == "multi_step":
        if len(tool_call_indices) < 2:
            errors.append("Multi-step example must contain at least 2 tool calls")

    if example_type == "refusal":
        # Should NOT have any tool calls
        if tool_call_indices:
            errors.append("Refusal example should not contain tool calls")

    # Check: assistant response after tool result references something from the result
    for ti in tool_result_indices:
        tool_content = messages[ti].get("content", "")
        # Find next assistant message after this tool result
        next_assistant = None
        for m in messages[ti + 1:]:
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                next_assistant = m
                break
        if next_assistant and tool_content:
            assistant_content = next_assistant.get("content", "")
            # Extract words/numbers from tool result (at least 3 chars)
            tokens = set(re.findall(r"\b\w{3,}\b", str(tool_content)))
            overlap = tokens & set(re.findall(r"\b\w{3,}\b", str(assistant_content)))
            if not overlap:
                warnings.append(
                    "Assistant response after tool result shares no words with the result"
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
