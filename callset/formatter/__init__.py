from __future__ import annotations

from typing import Callable

from callset.formatter.chatml_fmt import format_chatml
from callset.formatter.hermes_fmt import format_hermes
from callset.formatter.openai_fmt import format_openai
from callset.formatter.raw_fmt import format_raw
from callset.models import ToolDef

FORMATTERS: dict[str, Callable[[dict, list[ToolDef]], dict]] = {
    "openai": format_openai,
    "hermes": format_hermes,
    "chatml": format_chatml,
    "raw": format_raw,
}


def format_conversation(conversation: dict, tools: list[ToolDef], fmt: str) -> dict:
    """Format a conversation using the specified output format."""
    formatter = FORMATTERS.get(fmt)
    if not formatter:
        raise ValueError(f"Unknown format: {fmt}. Choose from: {', '.join(FORMATTERS)}")
    return formatter(conversation, tools)
