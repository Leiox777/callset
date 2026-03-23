from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _process_seed(
    seed: dict,
    context: APIContext,
    provider: LLMProvider,
    max_retries: int,
    strict: bool,
    system_prompt: str | None,
    verbose: bool,
) -> tuple[dict | None, dict]:
    """Process a single seed: generate, validate, and retry if needed.

    Returns (conversation_or_None, local_stats).
    """
    local_stats = {"generated": 0, "passed": 0, "failed": 0, "retries": 0, "failure_reasons": {}, "type_counts": {}}
    conversation = None
    example_type: ExampleType = seed["type"]

    for attempt in range(1 + max_retries):
        local_stats["generated"] += 1
        if attempt > 0:
            local_stats["retries"] += 1

        try:
            conv = generate_conversation(seed, context, provider, system_prompt)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Parse error on attempt {attempt + 1}: {e}")
            _record_failure(local_stats, "malformed_json")
            continue

        result = validate_conversation(
            conv, context.tools, example_type.value, strict=strict
        )

        if result.valid:
            conversation = conv
            break
        else:
            for error in result.errors:
                _record_failure(local_stats, error)
            if verbose:
                logger.info(f"Validation failed (attempt {attempt + 1}): {result.errors}")

    if conversation:
        local_stats["passed"] = 1
        local_stats["type_counts"][example_type.value] = 1
    else:
        local_stats["failed"] = 1

    return conversation, local_stats


def _merge_stats(stats: dict, local_stats: dict, lock: threading.Lock) -> None:
    """Merge a worker's local stats into the shared stats dict."""
    with lock:
        stats["generated"] += local_stats["generated"]
        stats["passed"] += local_stats["passed"]
        stats["failed"] += local_stats["failed"]
        stats["retries"] += local_stats["retries"]
        for reason, count in local_stats["failure_reasons"].items():
            stats["failure_reasons"][reason] = stats["failure_reasons"].get(reason, 0) + count
        for etype, count in local_stats.get("type_counts", {}).items():
            stats["type_counts"][etype] = stats["type_counts"].get(etype, 0) + count


def generate_all(
    seeds: list[dict],
    context: APIContext,
    provider: LLMProvider,
    max_retries: int = 2,
    strict: bool = False,
    system_prompt: str | None = None,
    verbose: bool = False,
    workers: int = 1,
) -> tuple[list[dict], dict]:
    """Generate and validate all conversations.

    Args:
        workers: Number of parallel threads for generation (default 1).

    Returns (conversations, stats) where stats tracks generation metrics.
    """
    stats = {
        "generated": 0,
        "passed": 0,
        "failed": 0,
        "retries": 0,
        "failure_reasons": {},
        "type_counts": {},
    }
    lock = threading.Lock()
    # Pre-allocate to preserve seed ordering in the output
    results: list[dict | None] = [None] * len(seeds)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Generating conversations", total=len(seeds))

        def _worker(index: int, seed: dict) -> None:
            conv, local_stats = _process_seed(
                seed, context, provider, max_retries, strict, system_prompt, verbose
            )
            results[index] = conv
            _merge_stats(stats, local_stats, lock)
            progress.update(task, advance=1)

        if workers <= 1:
            for i, seed in enumerate(seeds):
                _worker(i, seed)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_worker, i, seed): i
                    for i, seed in enumerate(seeds)
                }
                for future in as_completed(futures):
                    future.result()  # propagate exceptions

    conversations = [r for r in results if r is not None]
    return conversations, stats


def _record_failure(stats: dict, reason: str) -> None:
    """Increment a failure reason counter."""
    reasons = stats["failure_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
