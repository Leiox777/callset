from __future__ import annotations

import json

from callset.models import ValidationResult

VALID_ROLES = {"system", "user", "assistant", "tool"}


def validate_format(conversation: dict, output_format: str = "openai") -> ValidationResult:
    """Validate the structural format of a conversation."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(conversation, dict):
        return ValidationResult(valid=False, errors=["Conversation must be a dict"])

    messages = conversation.get("messages")
    if not isinstance(messages, list):
        return ValidationResult(valid=False, errors=["Conversation must have a 'messages' list"])

    if len(messages) == 0:
        errors.append("Conversation has no messages")

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"Message {i} is not a dict")
            continue

        role = msg.get("role")
        if role not in VALID_ROLES:
            errors.append(f"Message {i} has invalid role: {role!r}")

        if "content" not in msg and "tool_calls" not in msg:
            errors.append(f"Message {i} has neither 'content' nor 'tool_calls'")

        # Validate tool_calls structure
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            if role != "assistant":
                errors.append(f"Message {i}: only assistant messages can have tool_calls")
            if not isinstance(tool_calls, list):
                errors.append(f"Message {i}: tool_calls must be a list")
            else:
                for j, tc in enumerate(tool_calls):
                    if not isinstance(tc, dict):
                        errors.append(f"Message {i}, tool_call {j}: must be a dict")
                        continue
                    func = tc.get("function", {})
                    if not func.get("name"):
                        errors.append(f"Message {i}, tool_call {j}: missing function.name")
                    args = func.get("arguments")
                    if args is not None and isinstance(args, str):
                        try:
                            json.loads(args)
                        except json.JSONDecodeError:
                            errors.append(f"Message {i}, tool_call {j}: arguments is not valid JSON")

        # Validate tool response messages
        if role == "tool" and "tool_call_id" not in msg:
            warnings.append(f"Message {i}: tool message missing 'tool_call_id'")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
