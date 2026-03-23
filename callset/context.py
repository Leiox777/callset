from __future__ import annotations

from collections import Counter

from callset.models import APIContext, ToolDef


def _infer_tool_chains(tools: list[ToolDef]) -> list[tuple[str, str, str]]:
    """Infer tool chains by matching response fields to input parameters.

    Looks for patterns like:
    - Exact match: tool A returns 'flight_id', tool B takes 'flight_id'
    - ID suffix: tool A returns 'id', tool B takes '{toolA_singular}_id'
    """
    chains: list[tuple[str, str, str]] = []

    for source in tools:
        if not source.response_schema:
            continue
        response_props = source.response_schema.get("properties", {})
        if not response_props:
            # Check if response is an array with item properties
            items = source.response_schema.get("items", {})
            response_props = items.get("properties", {})

        if not response_props:
            continue

        response_fields = set(response_props.keys())

        for target in tools:
            if target.name == source.name:
                continue
            target_params = set(target.parameters.get("properties", {}).keys())

            matched = False
            for resp_field in response_fields:
                if matched:
                    break
                # Exact match
                if resp_field in target_params:
                    chains.append((source.name, target.name, resp_field))
                    matched = True
                    continue
                # source returns 'id', target takes something like '{noun}_id'
                if resp_field == "id":
                    for param in target_params:
                        if param.endswith("_id"):
                            chains.append((source.name, target.name, param))
                            matched = True
                            break

    return chains


def _infer_domain(tools: list[ToolDef], spec_title: str) -> str:
    """Infer the API domain from spec title or most common tag."""
    if spec_title:
        return spec_title
    tags: list[str] = []
    for tool in tools:
        tags.extend(tool.tags)
    if tags:
        most_common = Counter(tags).most_common(1)[0][0]
        return most_common
    return "API"


def build_context(
    tools: list[ToolDef],
    spec_title: str = "",
    spec_description: str = "",
) -> APIContext:
    """Build an APIContext from parsed tool definitions."""
    domain = _infer_domain(tools, spec_title)

    if spec_description:
        description = spec_description
    else:
        tool_names = ", ".join(t.name for t in tools)
        description = f"{domain} API with tools: {tool_names}"

    tool_chains = _infer_tool_chains(tools)

    return APIContext(
        domain=domain,
        description=description,
        tools=tools,
        tool_chains=tool_chains,
    )
