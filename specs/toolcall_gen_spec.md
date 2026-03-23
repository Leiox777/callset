# toolcall-gen

Generate validated tool calling training data from your API spec.

```
$ toolcall-gen --spec openapi.yaml --examples 1000 --output dataset.jsonl
```

Input: an OpenAPI spec or JSON tool definitions.
Output: a JSONL training dataset of multi-turn conversations with validated tool calls.

---

## What It Does

You give it your API definition. It generates realistic conversations where a user asks for things, an assistant calls your tools, tools return results, and the assistant responds naturally. Every generated tool call is validated against your schema. Bad examples are rejected and regenerated. The output is ready for fine-tuning with Unsloth, Axolotl, TRL, or the OpenAI fine-tuning API.

---

## Input Formats

### OpenAPI Spec (YAML or JSON)

```
$ toolcall-gen --spec api.yaml
```

The tool parses the OpenAPI spec and extracts:

- API title and description → used to infer the domain and seed realistic user queries
- Each endpoint → becomes a tool definition (operation ID as tool name, summary as description)
- Parameters → tool parameters with types, required flags, descriptions, enums, defaults
- Request body schemas → additional tool parameters for POST/PUT endpoints
- Response schemas → used to generate realistic simulated tool results
- Server URL and tags → additional domain context

### JSON Tool Definitions

For non-REST APIs, MCP servers, or custom function signatures:

```
$ toolcall-gen --tools tools.json
```

```json
[
  {
    "name": "search_flights",
    "description": "Search for available flights between two airports",
    "parameters": {
      "type": "object",
      "properties": {
        "departure_city": {
          "type": "string",
          "description": "IATA airport code"
        },
        "arrival_city": {
          "type": "string",
          "description": "IATA airport code"
        },
        "date": {
          "type": "string",
          "description": "Departure date in YYYY-MM-DD format"
        }
      },
      "required": ["departure_city", "arrival_city", "date"]
    }
  }
]
```

This is the same format as OpenAI's function calling schema. MCP tool definitions can be converted to this format trivially since they use JSON Schema for parameters.

### Python Functions (stretch goal, not MVP)

```
$ toolcall-gen --module my_tools.py
```

Parse function signatures, type hints, and docstrings into tool definitions. Similar to how TRL's `get_json_schema` works. Not in the first release — add it if people ask.

---

## Output Formats

### OpenAI Function Calling Format (default)

```
$ toolcall-gen --spec api.yaml --format openai
```

```json
{
  "messages": [
    {"role": "system", "content": "You are a travel booking assistant..."},
    {"role": "user", "content": "I need to fly from New York to Tokyo in mid-March"},
    {"role": "assistant", "content": "Let me search for available flights.", "tool_calls": [
      {"id": "call_1", "type": "function", "function": {"name": "search_flights", "arguments": "{\"departure_city\": \"JFK\", \"arrival_city\": \"NRT\", \"date\": \"2026-03-15\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"flights\": [{\"airline\": \"JAL\", \"price\": 1250, \"departure\": \"2026-03-15T10:30:00\"}]}"},
    {"role": "assistant", "content": "I found a JAL flight departing March 15th at 10:30 AM for $1,250. Would you like me to book it?"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "search_flights", "description": "...", "parameters": {...}}}
  ]
}
```

### Hermes Format

```
$ toolcall-gen --spec api.yaml --format hermes
```

Uses `<tools>`, `<tool_call>`, `<tool_response>` XML tags in the conversation. Compatible with NousResearch Hermes models and many open-source tool calling fine-tunes.

### ChatML with Tool Tokens

```
$ toolcall-gen --spec api.yaml --format chatml
```

Standard ChatML with `<|im_start|>` tokens, tool calls as JSON blocks within assistant turns.

### Raw JSONL

```
$ toolcall-gen --spec api.yaml --format raw
```

Minimal format with just the conversation turns as a list. User handles tokenization and template formatting.

---

## Example Types

Every generated dataset contains a mix of five conversation types. The distribution is configurable.

### 1. Happy Path (default: 40%)

Straightforward single tool call. User asks for something, assistant calls the right tool, tool returns results, assistant responds.

```
User: "What flights are available from Boston to London on April 5th?"
Assistant: [calls search_flights] → results → "I found 4 flights..."
```

### 2. Multi-Step (default: 20%)

Sequential tool calls where one result informs the next. Inferred from API structure — if the spec has `search_flights` and `book_flight`, the tool generates search-then-book conversations.

```
User: "Find me a flight to Tokyo and book the cheapest one"
Assistant: [calls search_flights] → results → "The cheapest is $1,250 on JAL..."
User: "Book that one"
Assistant: [calls book_flight with flight_id from previous result] → confirmation
```

The tool infers multi-step chains by analyzing parameter dependencies between endpoints. If `book_flight` takes a `flight_id` parameter and `search_flights` returns objects with an `id` field, they form a natural chain.

### 3. Clarification (default: 15%)

User provides incomplete information. The assistant asks for required parameters before calling.

```
User: "Book me a flight"
Assistant: "I'd be happy to help. Where would you like to fly to, and when?"
User: "Tokyo, sometime next week"
Assistant: [calls search_flights with inferred parameters]
```

Generated by looking at required parameters and deliberately omitting some from the user query. Every required parameter is an opportunity for a clarification turn.

### 4. Error Handling (default: 15%)

Tool returns an error or empty result. The assistant handles it gracefully.

```
User: "Find flights from Boston to Atlantis"
Assistant: [calls search_flights] → error: "Unknown airport code"
Assistant: "I couldn't find an airport matching 'Atlantis'. Could you check the city name?"
```

```
User: "Any flights to London on December 25th?"
Assistant: [calls search_flights] → empty results
Assistant: "No flights are available on December 25th. Would you like me to check nearby dates?"
```

Error shapes are inferred from response schemas (error fields, status codes) and common API failure patterns.

### 5. Refusal (default: 10%)

User asks for something outside the API's capabilities. The assistant responds helpfully without hallucinating a tool.

```
User: "What's the weather like in Tokyo?"
Assistant: "I'm a travel booking assistant, so I can't check weather forecasts. I can help you search for flights or hotels in Tokyo though."
```

Generated by creating user queries for plausible but unsupported actions based on the API's domain. If it's a travel API, weather, restaurant recommendations, and visa requirements are plausible questions the user might ask but the API can't answer.

### Distribution Control

```
$ toolcall-gen --spec api.yaml \
    --distribution "happy:50,multi_step:20,clarification:10,error:10,refusal:10"
```

Percentages must sum to 100. Default is 40/20/15/15/10.

---

## Generation Pipeline

### Step 1: Parse Spec

Read the OpenAPI spec or JSON tool definitions. Extract:

```python
@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict          # JSON Schema
    required_params: list[str]
    optional_params: list[str]
    response_schema: dict | None
    tags: list[str]

@dataclass
class APIContext:
    domain: str               # "travel booking", "e-commerce", etc.
    description: str          # from spec info.description
    tools: list[ToolDef]
    tool_chains: list[tuple]  # inferred multi-step sequences
```

Tool chains are inferred by matching output fields of one endpoint to input parameters of another. If `search_flights` returns `{id, airline, price}` and `book_flight` takes `flight_id`, that's a chain.

### Step 2: Build API Context Summary

Generate a one-paragraph domain description from the spec metadata. This seeds the LLM so generated conversations sound like real users of this API.

```
"This is a travel booking API called SkyRoute. It offers 8 tools:
search for flights, book flights, cancel bookings, search hotels,
book hotels, get booking details, list airports, and check flight
status. Typical users are travelers planning trips."
```

If the spec has a good description, use it directly. If not, have the LLM summarize from the tool definitions.

### Step 3: Generate Scenario Seeds

For each example to generate, create a brief scenario description based on the example type:

```python
seeds = [
    {"type": "happy", "scenario": "User searches for flights from Chicago to Miami in June"},
    {"type": "multi_step", "scenario": "User searches for flights, selects one, then books it"},
    {"type": "clarification", "scenario": "User says 'book a hotel' without specifying city or dates"},
    {"type": "error", "scenario": "User searches for flights on a past date"},
    {"type": "refusal", "scenario": "User asks for restaurant recommendations in Paris"},
]
```

Scenario generation uses the API context to produce diverse, realistic seeds. The LLM is prompted with the tool definitions and asked to generate N diverse user scenarios for each type.

### Step 4: Expand Seeds into Conversations

Each seed is expanded into a full multi-turn conversation using the LLM. The prompt includes:

- The API context summary
- All tool definitions (full JSON schemas)
- The scenario seed and type
- The output format specification
- Instructions for simulating realistic tool results based on response schemas

Single LLM call per conversation. The model generates the entire exchange including user turns, assistant reasoning, tool calls with arguments, simulated tool results, and assistant responses.

### Step 5: Validate

Every generated conversation goes through validation:

**Format validation:**
- Is the output valid JSON?
- Does it have the expected structure (messages array, correct roles)?
- Are tool calls properly formatted?

**Schema validation (the critical differentiator):**
- Does each tool call reference a tool that exists in the spec?
- Are all required parameters present?
- Do parameter types match the schema (string for string, number for number)?
- Are enum values valid (if the parameter has an enum constraint)?
- Are there no extra parameters that don't exist in the schema?

**Semantic validation (lightweight):**
- Does the assistant response reference the tool result? (prevent disconnected responses)
- Is there at least one user turn before a tool call? (prevent unprompted tool calls)
- For clarification examples, does the assistant actually ask before calling? (prevent skipping clarification)

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]        # what failed
    warnings: list[str]      # suspicious but not invalid

def validate_conversation(conversation: dict, tools: list[ToolDef]) -> ValidationResult:
    ...
```

### Step 6: Reject and Regenerate

Failed conversations are discarded and new ones are generated in their place. Track the pass rate. A healthy pipeline should pass 65-85% of generated examples on the first attempt. Lower than 50% suggests the LLM is struggling with the tool definitions (they may be too complex or poorly described). Higher than 95% suggests validation is too lenient.

Log the rejection reasons so users can see what's failing:

```
Generated: 1200 | Passed: 987 | Failed: 213 (82.3% pass rate)
Failures: 
  - wrong parameter type: 89
  - missing required param: 54
  - hallucinated tool: 38
  - malformed JSON: 32
```

### Step 7: Write Output

Write validated examples to JSONL in the specified format. One conversation per line.

Include a generation metadata comment at the top of the file:

```json
{"_meta": {"tool": "toolcall-gen", "version": "0.1.0", "spec": "openapi.yaml", "examples": 1000, "pass_rate": 0.823, "format": "openai", "generated_at": "2026-03-22T14:30:00Z"}}
```

---

## CLI Interface

```
toolcall-gen [OPTIONS]

Input (one required):
  --spec PATH          OpenAPI spec file (YAML or JSON)
  --tools PATH         JSON file with tool definitions

Generation:
  --examples N         Number of examples to generate (default: 500)
  --distribution STR   Example type distribution (default: "happy:40,multi_step:20,clarification:15,error:15,refusal:10")
  --system-prompt STR  Custom system prompt for the assistant (optional, inferred from spec if omitted)
  --personas STR       Comma-separated user persona descriptions (optional)

Output:
  --output PATH        Output file path (default: dataset.jsonl)
  --format FORMAT      Output format: openai, hermes, chatml, raw (default: openai)

LLM:
  --provider PROVIDER  LLM provider: openai, anthropic (default: openai)
  --model MODEL        Model to use for generation (default: gpt-4o)
  --api-key KEY        API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)

Validation:
  --strict             Reject examples with warnings (not just errors)
  --max-retries N      Max regeneration attempts per failed example (default: 2)

Other:
  --seed N             Random seed for reproducible generation
  --verbose            Print generation progress and validation details
  --dry-run            Parse spec and show inferred context without generating
```

### Dry Run

```
$ toolcall-gen --spec api.yaml --dry-run

API: SkyRoute Travel Booking API
Domain: Travel and hospitality
Tools: 8
  - search_flights (3 required, 2 optional params)
  - book_flight (2 required params)
  - cancel_booking (1 required param)
  - search_hotels (2 required, 3 optional params)
  - book_hotel (3 required params)
  - get_booking (1 required param)
  - list_airports (0 required, 1 optional param)
  - check_flight_status (1 required param)

Inferred tool chains:
  - search_flights → book_flight (via flight_id)
  - search_hotels → book_hotel (via hotel_id)
  - book_flight → get_booking (via booking_id)
  - book_hotel → get_booking (via booking_id)
  - get_booking → cancel_booking (via booking_id)

Would generate: 500 examples
  - happy: 200
  - multi_step: 100
  - clarification: 75
  - error: 75
  - refusal: 50
```

This lets the user verify the tool parsed their spec correctly before spending API credits on generation.

---

## LLM Prompts

Two main prompts power the pipeline.

### Scenario Seed Prompt

Generates diverse scenario descriptions for each example type.

```
You are generating training data scenarios for a tool-calling AI assistant.

API Context:
{api_context_summary}

Available Tools:
{tool_definitions_summary}

Generate {n} diverse, realistic scenario descriptions for the following
example type: {example_type}

{type_specific_instructions}

Return as a JSON array of scenario strings.
```

Type-specific instructions:

- **happy**: "Each scenario should involve a user asking something that one of the available tools can answer directly. Vary the tools used, the user's tone (casual, formal, urgent), and the specificity of the request."
- **multi_step**: "Each scenario should require calling two or more tools in sequence. Use the tool chains: {inferred_chains}. The user may state the full intent upfront or reveal it incrementally."
- **clarification**: "Each scenario should have the user making a vague or incomplete request. The required parameters they omit: {required_params_per_tool}. The assistant should ask for the missing information."
- **error**: "Each scenario should result in a tool returning an error or empty result. Reasons: invalid input, resource not found, past dates, conflicting parameters."
- **refusal**: "Each scenario should have the user asking for something plausible in this domain but not supported by any available tool. The assistant should decline helpfully."

### Conversation Expansion Prompt

Expands a scenario seed into a full multi-turn conversation.

```
You are generating a training example for a tool-calling AI assistant.

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
- Simulate realistic tool results based on the response schema provided
- The assistant's final response should naturally incorporate the tool result
- {type_specific_rules}

Return only valid JSON. No markdown, no explanation.
```

---

## Architecture

```
toolcall_gen/
├── __init__.py
├── cli.py                 # Click/argparse CLI entry point
├── parser/
│   ├── __init__.py
│   ├── openapi.py         # Parse OpenAPI spec → ToolDef list
│   └── json_tools.py      # Parse JSON tool definitions → ToolDef list
├── context.py             # Build APIContext from parsed tools
├── generator/
│   ├── __init__.py
│   ├── seeds.py           # Generate scenario seeds per type
│   ├── conversations.py   # Expand seeds into full conversations
│   └── providers.py       # LLM provider abstraction (OpenAI, Anthropic)
├── validator/
│   ├── __init__.py
│   ├── format.py          # JSON structure validation
│   ├── schema.py          # Tool call schema validation
│   └── semantic.py        # Lightweight semantic checks
├── formatter/
│   ├── __init__.py
│   ├── openai_fmt.py      # OpenAI function calling format
│   ├── hermes_fmt.py      # Hermes XML format
│   ├── chatml_fmt.py      # ChatML with tool tokens
│   └── raw_fmt.py         # Minimal format
└── models.py              # ToolDef, APIContext, ValidationResult dataclasses
```

~800-1000 lines total for the MVP. The validator and parser are the core — everything else is LLM calls and formatting.

---

## Dependencies

```toml
[project]
name = "toolcall-gen"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.0",         # LLM generation
    "pyyaml>=6.0",         # OpenAPI YAML parsing
    "jsonschema>=4.0",     # Schema validation
    "click>=8.0",          # CLI framework
    "rich>=13.0",          # Terminal output formatting
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.30"]

[project.scripts]
toolcall-gen = "toolcall_gen.cli:main"
```

Five dependencies. No heavy ML frameworks. Installs in seconds.

---

## What Success Looks Like

In two weeks after posting:

**Strong signal (proceed to product):**
- People file GitHub issues with feature requests ("add MCP support", "support Gorilla format", "handle nested object parameters")
- Someone posts "I used toolcall-gen to create 2,000 examples and fine-tuned my model, here are benchmark results"
- Multiple people ask for a hosted version or UI

**Moderate signal (keep iterating):**
- 50+ stars, a few issues, some forks
- People try it but report quality problems — this means the problem is real but the solution needs work

**Weak signal (move on):**
- Stars but no issues (people bookmarked it but didn't use it)
- No one mentions using the output for actual training
- Silence in communities after initial post

---

## What Is NOT in the MVP

- UI of any kind
- Web app or hosted service
- User accounts or payments
- MCP-specific tooling (add if people ask)
- Python function parsing (add if people ask)
- Actual API execution for validation (schema-only for now)
- DPO/preference pair generation (add if people ask)
- Batch generation with progress resume
- Custom scenario templates
- Multi-language support

Each of these is a natural extension if the core validates. None are needed to test whether people want this.
