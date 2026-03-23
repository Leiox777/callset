from __future__ import annotations

import json
from pathlib import Path

from callset.models import ToolDef


def parse_json_tools(path: Path) -> list[ToolDef]:
    """Parse a JSON file containing tool definitions into ToolDef objects."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of tool definitions, got {type(data).__name__}")

    tools: list[ToolDef] = []
    for i, tool in enumerate(data):
        if not isinstance(tool, dict):
            raise ValueError(f"Tool at index {i} is not an object")

        name = tool.get("name")
        if not name:
            raise ValueError(f"Tool at index {i} is missing 'name'")

        description = tool.get("description", "")
        parameters = tool.get("parameters", {"type": "object", "properties": {}})

        required = parameters.get("required", [])
        all_props = list(parameters.get("properties", {}).keys())
        optional = [p for p in all_props if p not in required]

        tools.append(ToolDef(
            name=name,
            description=description,
            parameters=parameters,
            required_params=required,
            optional_params=optional,
            response_schema=tool.get("response_schema"),
            tags=tool.get("tags", []),
        ))

    return tools
