from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from callset.models import ToolDef


def _resolve_ref(ref: str, spec: dict) -> dict:
    """Resolve a local $ref like '#/components/schemas/Foo'."""
    if not ref.startswith("#/"):
        raise ValueError(f"Only local $ref supported, got: {ref}")
    parts = ref.lstrip("#/").split("/")
    node = spec
    for part in parts:
        node = node[part]
    return _resolve_refs_deep(node, spec)


def _resolve_refs_deep(obj: Any, spec: dict) -> Any:
    """Recursively resolve all $ref occurrences in a schema object."""
    if isinstance(obj, dict):
        if "$ref" in obj:
            return _resolve_ref(obj["$ref"], spec)
        return {k: _resolve_refs_deep(v, spec) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs_deep(item, spec) for item in obj]
    return obj


def _path_to_name(method: str, path: str) -> str:
    """Generate a tool name from HTTP method and path: GET /users/{id} -> get_users_by_id."""
    slug = re.sub(r"\{(\w+)\}", r"by_\1", path)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug).strip("_").lower()
    return f"{method.lower()}_{slug}"


def _build_parameters_schema(
    path_params: list[dict],
    query_params: list[dict],
    request_body: dict | None,
    spec: dict,
) -> tuple[dict, list[str]]:
    """Build a unified JSON Schema object from path/query params and request body.

    Returns (schema_dict, required_list).
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in path_params + query_params:
        param = _resolve_refs_deep(param, spec)
        name = param["name"]
        schema = param.get("schema", {"type": "string"})
        schema = _resolve_refs_deep(schema, spec)
        if "description" not in schema and "description" in param:
            schema["description"] = param["description"]
        properties[name] = schema
        if param.get("required", False):
            required.append(name)

    if request_body:
        content = request_body.get("content", {})
        json_content = content.get("application/json", {})
        body_schema = json_content.get("schema", {})
        body_schema = _resolve_refs_deep(body_schema, spec)
        if body_schema.get("type") == "object" and "properties" in body_schema:
            properties.update(body_schema["properties"])
            required.extend(body_schema.get("required", []))
        elif body_schema:
            properties["body"] = body_schema
            if request_body.get("required", False):
                required.append("body")

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema, required


def _extract_response_schema(responses: dict, spec: dict) -> dict | None:
    """Extract the response schema from the 200 or 201 response."""
    for code in ("200", "201", 200, 201):
        resp = responses.get(code)
        if resp:
            resp = _resolve_refs_deep(resp, spec)
            content = resp.get("content", {})
            json_content = content.get("application/json", {})
            schema = json_content.get("schema")
            if schema:
                return _resolve_refs_deep(schema, spec)
    return None


def parse_openapi(path: Path) -> tuple[list[ToolDef], dict]:
    """Parse an OpenAPI spec file into ToolDef objects.

    Returns (tools, metadata) where metadata has 'title' and 'description'.
    """
    with open(path) as f:
        spec = yaml.safe_load(f)

    info = spec.get("info", {})
    metadata = {
        "title": info.get("title", ""),
        "description": info.get("description", ""),
    }

    tools: list[ToolDef] = []
    paths = spec.get("paths", {})

    for path_str, path_item in paths.items():
        # Collect path-level parameters
        path_level_params = path_item.get("parameters", [])

        for method in ("get", "post", "put", "patch", "delete"):
            operation = path_item.get(method)
            if not operation:
                continue

            operation_id = operation.get("operationId")
            name = operation_id if operation_id else _path_to_name(method, path_str)

            description = operation.get("summary", "") or operation.get("description", "")

            # Separate path and query params
            op_params = operation.get("parameters", [])
            all_params = path_level_params + op_params
            all_params = [_resolve_refs_deep(p, spec) for p in all_params]

            path_params = [p for p in all_params if p.get("in") == "path"]
            query_params = [p for p in all_params if p.get("in") == "query"]

            request_body = operation.get("requestBody")
            if request_body:
                request_body = _resolve_refs_deep(request_body, spec)

            parameters, required = _build_parameters_schema(
                path_params, query_params, request_body, spec
            )

            all_props = list(parameters.get("properties", {}).keys())
            optional = [p for p in all_props if p not in required]

            response_schema = _extract_response_schema(
                operation.get("responses", {}), spec
            )

            tags = operation.get("tags", [])

            tools.append(ToolDef(
                name=name,
                description=description,
                parameters=parameters,
                required_params=required,
                optional_params=optional,
                response_schema=response_schema,
                tags=tags,
            ))

    return tools, metadata
