from __future__ import annotations

import json
import math

from callset.generator.providers import LLMProvider
from callset.models import APIContext, ExampleType
from callset.prompts import (
    SEED_SYSTEM_PROMPT,
    SEED_USER_TEMPLATE,
    TYPE_INSTRUCTIONS,
)


def _build_tool_summary(context: APIContext) -> str:
    """Build a concise summary of available tools."""
    lines = []
    for tool in context.tools:
        req = ", ".join(tool.required_params) or "none"
        opt = ", ".join(tool.optional_params) or "none"
        lines.append(f"- {tool.name}: {tool.description} (required: {req}; optional: {opt})")
    return "\n".join(lines)


def _build_type_instructions(example_type: ExampleType, context: APIContext) -> str:
    """Build type-specific instructions with context placeholders filled."""
    template = TYPE_INSTRUCTIONS[example_type]

    if example_type == ExampleType.MULTI_STEP:
        chains_str = ", ".join(
            f"{src} → {tgt} (via {param})" for src, tgt, param in context.tool_chains
        ) or "no explicit chains inferred"
        return template.format(tool_chains=chains_str)

    if example_type == ExampleType.CLARIFICATION:
        req_params = "; ".join(
            f"{t.name}: {', '.join(t.required_params)}"
            for t in context.tools if t.required_params
        )
        return template.format(required_params=req_params)

    return template


def _parse_seeds_response(response: str) -> list[str]:
    """Parse the LLM response into a list of scenario strings."""
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def generate_seeds(
    context: APIContext,
    distribution: dict[ExampleType, int],
    num_examples: int,
    provider: LLMProvider,
) -> list[dict]:
    """Generate scenario seeds for all example types.

    Returns a list of {"type": ExampleType, "scenario": str} dicts.
    """
    seeds: list[dict] = []
    tool_summary = _build_tool_summary(context)
    api_summary = f"{context.domain}: {context.description}"

    for example_type, pct in distribution.items():
        count = max(1, math.ceil(num_examples * pct / 100))

        type_instructions = _build_type_instructions(example_type, context)

        prompt = SEED_USER_TEMPLATE.format(
            api_context_summary=api_summary,
            tool_definitions_summary=tool_summary,
            n=count,
            example_type=example_type.value,
            type_specific_instructions=type_instructions,
        )

        # Try up to 2 times to get valid JSON
        for attempt in range(2):
            try:
                response = provider.generate(
                    SEED_SYSTEM_PROMPT, prompt, temperature=0.9
                )
                scenarios = _parse_seeds_response(response)
                break
            except (json.JSONDecodeError, ValueError):
                if attempt == 1:
                    raise ValueError(
                        f"Failed to parse seed scenarios for {example_type.value} "
                        f"after 2 attempts. Last response: {response[:200]}"
                    )

        for scenario in scenarios[:count]:
            seeds.append({"type": example_type, "scenario": scenario})

    return seeds
