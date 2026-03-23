from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExampleType(str, Enum):
    HAPPY = "happy"
    MULTI_STEP = "multi_step"
    CLARIFICATION = "clarification"
    ERROR = "error"
    REFUSAL = "refusal"


DEFAULT_DISTRIBUTION: dict[ExampleType, int] = {
    ExampleType.HAPPY: 40,
    ExampleType.MULTI_STEP: 20,
    ExampleType.CLARIFICATION: 15,
    ExampleType.ERROR: 15,
    ExampleType.REFUSAL: 10,
}


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict  # Full JSON Schema object
    required_params: list[str] = field(default_factory=list)
    optional_params: list[str] = field(default_factory=list)
    response_schema: dict | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class APIContext:
    domain: str
    description: str
    tools: list[ToolDef]
    tool_chains: list[tuple[str, str, str]] = field(default_factory=list)
    # Each tuple is (source_tool, target_tool, linking_param)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )
