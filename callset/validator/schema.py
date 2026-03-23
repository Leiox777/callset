from __future__ import annotations

import json

import jsonschema

from callset.models import ToolDef, ValidationResult


def validate_schema(conversation: dict, tools: list[ToolDef]) -> ValidationResult:
    """Validate tool calls in a conversation against tool schemas."""
    errors: list[str] = []
    warnings: list[str] = []

    tool_lookup = {t.name: t for t in tools}
    messages = conversation.get("messages", [])

    for i, msg in enumerate(messages):
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue

        for j, tc in enumerate(tool_calls):
            func = tc.get("function", {})
            name = func.get("name", "")
            prefix = f"Message {i}, tool_call {j} ({name})"

            # Check tool exists
            tool_def = tool_lookup.get(name)
            if not tool_def:
                errors.append(f"{prefix}: unknown tool '{name}'")
                continue

            # Parse arguments
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    errors.append(f"{prefix}: arguments is not valid JSON")
                    continue
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                errors.append(f"{prefix}: arguments must be a JSON string or dict")
                continue

            if not isinstance(args, dict):
                errors.append(f"{prefix}: parsed arguments must be an object")
                continue

            # Validate against JSON Schema
            schema = tool_def.parameters.copy()
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = False

            try:
                jsonschema.validate(instance=args, schema=schema)
            except jsonschema.ValidationError as e:
                errors.append(f"{prefix}: {e.message}")

            # Check required params explicitly
            for param in tool_def.required_params:
                if param not in args:
                    errors.append(f"{prefix}: missing required parameter '{param}'")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
