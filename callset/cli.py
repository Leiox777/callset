from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from callset import __version__
from callset.context import build_context
from callset.formatter import format_conversation
from callset.generator.conversations import generate_all
from callset.generator.providers import get_provider
from callset.generator.seeds import generate_seeds
from callset.models import DEFAULT_DISTRIBUTION, ExampleType
from callset.parser.json_tools import parse_json_tools
from callset.parser.openapi import parse_openapi

console = Console()


def _parse_distribution(value: str) -> dict[ExampleType, int]:
    """Parse a distribution string like 'happy:40,multi_step:20,...'."""
    dist: dict[ExampleType, int] = {}
    for part in value.split(","):
        part = part.strip()
        if ":" not in part:
            raise click.BadParameter(f"Invalid distribution entry: {part!r}. Expected 'type:pct'.")
        name, pct_str = part.split(":", 1)
        try:
            example_type = ExampleType(name.strip())
        except ValueError:
            valid = ", ".join(e.value for e in ExampleType)
            raise click.BadParameter(f"Unknown example type: {name!r}. Valid types: {valid}")
        try:
            pct = int(pct_str.strip())
        except ValueError:
            raise click.BadParameter(f"Invalid percentage for {name}: {pct_str!r}")
        dist[example_type] = pct

    total = sum(dist.values())
    if total != 100:
        raise click.BadParameter(f"Distribution must sum to 100, got {total}")
    return dist


@click.command()
@click.option("--spec", "spec_path", type=click.Path(exists=True), default=None, help="OpenAPI spec file (YAML or JSON)")
@click.option("--tools", "tools_path", type=click.Path(exists=True), default=None, help="JSON file with tool definitions")
@click.option("--examples", default=500, type=int, help="Number of examples to generate")
@click.option("--distribution", default=None, type=str, help="Example type distribution (e.g. 'happy:40,multi_step:20,...')")
@click.option("--output", default="dataset.jsonl", type=click.Path(), help="Output file path")
@click.option("--format", "fmt", default="openai", type=click.Choice(["openai", "hermes", "chatml", "raw"]), help="Output format")
@click.option("--provider", default="openai", type=click.Choice(["openai", "anthropic"]), help="LLM provider")
@click.option("--model", default=None, type=str, help="Model to use for generation")
@click.option("--api-key", default=None, type=str, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
@click.option("--strict", is_flag=True, help="Reject examples with warnings")
@click.option("--max-retries", default=2, type=int, help="Max regeneration attempts per failed example")
@click.option("--seed", "random_seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--verbose", is_flag=True, help="Print progress and validation details")
@click.option("--dry-run", is_flag=True, help="Parse spec and show inferred context without generating")
@click.option("--export-tools", "export_tools_path", default=None, type=click.Path(), help="Export parsed tools as OpenAI-format JSON and exit")
@click.option("--system-prompt", default=None, type=str, help="Custom system prompt for the assistant")
@click.option("--personas", default=None, type=str, help="Comma-separated user persona descriptions")
@click.option("--workers", default=1, type=int, help="Number of parallel threads for generation (default: 1)")
def main(
    spec_path, tools_path, examples, distribution, output, fmt,
    provider, model, api_key, strict, max_retries, random_seed,
    verbose, dry_run, export_tools_path, system_prompt, personas, workers,
):
    """Generate validated tool-calling training data from API specifications."""
    # Configure logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Validate input: exactly one of --spec or --tools
    if not spec_path and not tools_path:
        raise click.UsageError("Provide one of --spec or --tools.")
    if spec_path and tools_path:
        raise click.UsageError("Provide only one of --spec or --tools, not both.")

    # Set random seed
    if random_seed is not None:
        random.seed(random_seed)

    # Step 1: Parse spec
    metadata = {}
    if spec_path:
        tools, metadata = parse_openapi(Path(spec_path))
        console.print(f"[bold]Parsed OpenAPI spec:[/bold] {spec_path}")
    else:
        tools = parse_json_tools(Path(tools_path))
        console.print(f"[bold]Parsed tool definitions:[/bold] {tools_path}")

    console.print(f"  Found {len(tools)} tools")

    # Step 2: Build context
    context = build_context(
        tools,
        spec_title=metadata.get("title", ""),
        spec_description=metadata.get("description", ""),
    )

    # Parse distribution
    dist = _parse_distribution(distribution) if distribution else DEFAULT_DISTRIBUTION

    # Export tools as OpenAI-format JSON and exit
    if export_tools_path:
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
            tool_defs.append(tool_def)
        export_path = Path(export_tools_path)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(tool_defs, f, indent=2, ensure_ascii=False)
        console.print(f"[bold green]Exported {len(tool_defs)} tools to {export_path}[/bold green]")
        return

    # Dry run: show context and exit
    if dry_run:
        _print_dry_run(context, dist, examples)
        return

    # Step 3-6: Generate
    console.print(f"\n[bold]Generating {examples} examples...[/bold]")

    llm = get_provider(provider, model, api_key)
    console.print(f"  Provider: {provider} ({llm.model_name})")

    # Generate seeds
    console.print("  Generating scenario seeds...")
    seeds = generate_seeds(context, dist, examples, llm)
    console.print(f"  Generated {len(seeds)} seeds")

    # Expand seeds into conversations
    conversations, stats = generate_all(
        seeds, context, llm,
        max_retries=max_retries,
        strict=strict,
        system_prompt=system_prompt,
        verbose=verbose,
        workers=workers,
    )

    # Step 7: Format and write output
    output_path = Path(output)
    with open(output_path, "w", encoding="utf-8") as f:
        # Write metadata line
        meta = {
            "_meta": {
                "tool": "callset",
                "version": __version__,
                "spec": spec_path or tools_path,
                "examples": len(conversations),
                "pass_rate": round(stats["passed"] / max(stats["generated"], 1), 3),
                "format": fmt,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for conv in conversations:
            formatted = format_conversation(conv, context.tools, fmt)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    # Print summary
    console.print(f"\n[bold green]Done![/bold green] Wrote {len(conversations)} examples to {output_path}")
    _print_stats(stats)


def _print_dry_run(context, dist, num_examples):
    """Print dry-run context summary."""
    console.print(f"\n[bold]API:[/bold] {context.domain}")
    console.print(f"[bold]Description:[/bold] {context.description}")
    console.print(f"\n[bold]Tools:[/bold] {len(context.tools)}")
    for t in context.tools:
        req = len(t.required_params)
        opt = len(t.optional_params)
        console.print(f"  - {t.name} ({req} required, {opt} optional params)")

    if context.tool_chains:
        console.print(f"\n[bold]Inferred tool chains:[/bold]")
        for src, tgt, param in context.tool_chains:
            console.print(f"  - {src} -> {tgt} (via {param})")

    console.print(f"\n[bold]Would generate:[/bold] {num_examples} examples")
    for etype, pct in dist.items():
        count = max(1, round(num_examples * pct / 100))
        console.print(f"  - {etype.value}: {count}")


def _print_stats(stats):
    """Print generation statistics."""
    table = Table(title="Generation Stats")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Generated", str(stats["generated"]))
    table.add_row("Passed", str(stats["passed"]))
    table.add_row("Failed", str(stats["failed"]))
    table.add_row("Retries", str(stats["retries"]))

    total = stats["generated"]
    rate = stats["passed"] / max(total, 1) * 100
    table.add_row("Pass Rate", f"{rate:.1f}%")
    console.print(table)

    if stats.get("type_counts"):
        total_passed = stats["passed"]
        console.print("\n[bold]Example distribution:[/bold]")
        for etype, count in sorted(stats["type_counts"].items()):
            pct = count / max(total_passed, 1) * 100
            console.print(f"  {etype}: {count} ({pct:.1f}%)")

    if stats["failure_reasons"]:
        console.print("\n[bold]Failure reasons:[/bold]")
        for reason, count in sorted(stats["failure_reasons"].items(), key=lambda x: -x[1]):
            console.print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
