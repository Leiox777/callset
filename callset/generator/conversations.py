from __future__ import annotations

import json
import logging

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from callset.generator.providers import LLMProvider
from callset.models import APIContext, ExampleType, ToolDef
from callset.prompts import (
    CONVERSATION_SYSTEM_PROMPT,
    CONVERSATION_USER_TEMPLATE,
    OPENAI_FORMAT_TEMPLATE,
    TYPE_RULES,
)
from callset.validator import validate_conversation

logger = logging.getLogger(__name__)


def _build_tools_json(tools: list[ToolDef]) -> str:
    """Build full tool definitions as JSON for the prompt."""
    tool_defs = []
    for t in tools:
        tool_def = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        if t.response_schema:
            tool_def["function"]["response_schema"] = t.response_schema
        tool_defs.append(tool_def)
    return json.dumps(tool_defs, indent=2)


def _parse_conversation_response(response: str) -> dict:
    """Parse the LLM response into a conversation dict."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def generate_conversation(
    seed: dict,
    context: APIContext,
    provider: LLMProvider,
    system_prompt: str | None = None,
) -> dict:
    """Generate a single conversation from a scenario seed."""
    example_type: ExampleType = seed["type"]
    scenario: str = seed["scenario"]

    if not system_prompt:
        system_prompt = f"You are a helpful {context.domain} assistant."

    type_rules = TYPE_RULES[example_type]
    tools_json = _build_tools_json(context.tools)

    prompt = CONVERSATION_USER_TEMPLATE.format(
        system_prompt=system_prompt,
        full_tool_definitions_json=tools_json,
        scenario_seed=scenario,
        example_type=example_type.value,
        format_template=OPENAI_FORMAT_TEMPLATE,
        type_specific_rules=type_rules,
    )

    response = provider.generate(CONVERSATION_SYSTEM_PROMPT, prompt, temperature=0.7)
    return _parse_conversation_response(response)


def generate_all(
    seeds: list[dict],
    context: APIContext,
    provider: LLMProvider,
    max_retries: int = 2,
    strict: bool = False,
    system_prompt: str | None = None,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Generate and validate all conversations.

    Returns (conversations, stats) where stats tracks generation metrics.
    """
    conversations: list[dict] = []
    stats = {
        "generated": 0,
        "passed": 0,
        "failed": 0,
        "retries": 0,
        "failure_reasons": {},
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Generating conversations", total=len(seeds))

        for seed in seeds:
            conversation = None
            example_type: ExampleType = seed["type"]

            for attempt in range(1 + max_retries):
                stats["generated"] += 1
                if attempt > 0:
                    stats["retries"] += 1

                try:
                    conv = generate_conversation(
                        seed, context, provider, system_prompt
                    )
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Parse error on attempt {attempt + 1}: {e}")
                    _record_failure(stats, "malformed_json")
                    continue

                result = validate_conversation(
                    conv, context.tools, example_type.value, strict=strict
                )

                if result.valid:
                    conversation = conv
                    break
                else:
                    for error in result.errors:
                        _record_failure(stats, error)
                    if verbose:
                        logger.info(
                            f"Validation failed (attempt {attempt + 1}): {result.errors}"
                        )

            if conversation:
                conversations.append(conversation)
                stats["passed"] += 1
            else:
                stats["failed"] += 1

            progress.update(task, advance=1)

    return conversations, stats


def _record_failure(stats: dict, reason: str) -> None:
    """Increment a failure reason counter."""
    reasons = stats["failure_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
