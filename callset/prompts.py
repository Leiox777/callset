from __future__ import annotations

from callset.models import ExampleType

SEED_SYSTEM_PROMPT = "You are generating training data scenarios for a tool-calling AI assistant."

SEED_USER_TEMPLATE = """\
API Context:
{api_context_summary}

Available Tools:
{tool_definitions_summary}

Generate {n} diverse, realistic scenario descriptions for the following \
example type: {example_type}

{type_specific_instructions}

Return as a JSON array of scenario strings. Return only valid JSON, no markdown or explanation."""

TYPE_INSTRUCTIONS: dict[ExampleType, str] = {
    ExampleType.HAPPY: (
        "Each scenario should involve a user asking something that one of the "
        "available tools can answer directly. Vary the tools used, the user's "
        "tone (casual, formal, urgent), and the specificity of the request."
    ),
    ExampleType.MULTI_STEP: (
        "Each scenario should require calling two or more tools in sequence. "
        "Use the tool chains: {tool_chains}. The user may state the full intent "
        "upfront or reveal it incrementally."
    ),
    ExampleType.CLARIFICATION: (
        "Each scenario should have the user making a vague or incomplete request. "
        "The required parameters they omit: {required_params}. The assistant "
        "should ask for the missing information."
    ),
    ExampleType.ERROR: (
        "Each scenario should result in a tool returning an error or empty result. "
        "Reasons: invalid input, resource not found, past dates, conflicting parameters."
    ),
    ExampleType.REFUSAL: (
        "Each scenario should have the user asking for something plausible in "
        "this domain but not supported by any available tool. The assistant "
        "should decline helpfully."
    ),
}

CONVERSATION_SYSTEM_PROMPT = "You are generating a training example for a tool-calling AI assistant."

CONVERSATION_USER_TEMPLATE = """\
System Prompt: {system_prompt}

Available Tools:
{full_tool_definitions_json}

Scenario: {scenario_seed}
Example Type: {example_type}

Generate a complete multi-turn conversation in the following JSON format:

{format_template}

Rules:
- The user message should sound natural, not like a command to call a function
- The assistant should decide when to call tools based on the conversation, not be told to
- Tool call arguments must exactly match the parameter names and types in the tool definitions
- When a parameter description indicates a specific format (e.g. "IATA code", "YYYY-MM-DD", "ISO 8601"), arguments MUST use that format, not a natural language equivalent. Use "CDG" not "Paris".
- Only include optional parameters in tool call arguments if the user explicitly mentioned them or they can be directly inferred from the user's words. Do NOT invent default values for optional parameters.
- Use dates in the near future (2026-2027) for all generated examples
- Simulate realistic tool results based on the response schema provided
- The assistant's final response should naturally incorporate the tool result
- {type_specific_rules}

Return only valid JSON. No markdown, no explanation."""

TYPE_RULES: dict[ExampleType, str] = {
    ExampleType.HAPPY: (
        "This is a straightforward single tool call. The user provides enough "
        "information for the assistant to call a tool immediately."
    ),
    ExampleType.MULTI_STEP: (
        "This requires at least 2 tool calls in separate assistant messages. "
        "The second tool call MUST use information from the first tool's result. "
        "The user should respond between tool calls (e.g., confirming a selection). "
        "Example flow: user asks → assistant searches → results → user picks one → "
        "assistant books with the selected ID → assistant summarizes."
    ),
    ExampleType.CLARIFICATION: (
        "The user's initial message is vague or missing required parameters. "
        "The assistant must ask for clarification before making any tool call."
    ),
    ExampleType.ERROR: (
        "The tool result MUST contain an error indicator (error field, empty result set, "
        "or HTTP-style error message). Do NOT simulate a successful result. "
        "The assistant MUST acknowledge the error and suggest an alternative or ask "
        "the user to correct their input."
    ),
    ExampleType.REFUSAL: (
        "The user asks for something outside the API's capabilities. The assistant "
        "should NOT call any tools. Instead, explain what it can help with."
    ),
}

OPENAI_FORMAT_TEMPLATE = """\
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "tool_name", "arguments": "{\\"param\\": \\"value\\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\\"result\\": ...}"},
    {"role": "assistant", "content": "..."}
  ]
}"""
