from __future__ import annotations

from callset.models import ToolDef, ValidationResult
from callset.validator.format import validate_format
from callset.validator.schema import validate_schema
from callset.validator.semantic import validate_semantic


def validate_conversation(
    conversation: dict,
    tools: list[ToolDef],
    example_type: str,
    output_format: str = "openai",
    strict: bool = False,
) -> ValidationResult:
    """Run all three validation layers and merge results."""
    result = validate_format(conversation, output_format)
    result = result.merge(validate_schema(conversation, tools))
    result = result.merge(validate_semantic(conversation, example_type))

    if strict and result.warnings:
        result = ValidationResult(
            valid=False,
            errors=result.errors + [f"(strict) {w}" for w in result.warnings],
            warnings=[],
        )

    return result
