# callset

Generate validated tool-calling training data from API specifications. Provide an OpenAPI spec or JSON tool definitions and callset produces a JSONL dataset of realistic, multi-turn conversations ready for fine-tuning language models.

## Installation

Requires Python >= 3.10.

```bash
git clone <repo-url>
cd callset
pip install -e .                     # OpenAI provider only
pip install -e ".[anthropic]"        # With Anthropic support
```

## Quick Start

Set your API key:

```bash
# Linux / macOS
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Windows (cmd)
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

Generate a dataset:

```bash
callset --spec api.yaml --examples 1000 --output dataset.jsonl
```

Preview what will be generated without using API credits:

```bash
callset --spec api.yaml --dry-run
```

## Usage

### Input Sources

callset accepts two input formats (one is required):

- `--spec PATH` — An OpenAPI 3.x spec file (YAML or JSON)
- `--tools PATH` — A JSON file with OpenAI-compatible tool definitions

### Example Types

Generated conversations are distributed across five scenario types:

| Type | Default % | Description |
|------|-----------|-------------|
| Happy path | 40 | Straightforward single tool call |
| Multi-step | 20 | Sequential tool calls using inferred chains |
| Clarification | 15 | Assistant asks for missing required parameters |
| Error handling | 15 | Tool returns an error; assistant handles gracefully |
| Refusal | 10 | User asks for an unsupported capability |

Customize the distribution with `--distribution`:

```bash
callset --spec api.yaml --distribution "happy:50,multi_step:25,clarification:10,error:10,refusal:5"
```

### Output Formats

Choose a format with `--format`:

| Format | Description |
|--------|-------------|
| `openai` (default) | OpenAI function calling format |
| `hermes` | NousResearch Hermes XML-tagged format |
| `chatml` | Standard ChatML with tool call blocks |
| `raw` | Minimal format with inline tool calls |

### CLI Reference

```
callset [OPTIONS]

Input (one required):
  --spec PATH              OpenAPI spec file (YAML/JSON)
  --tools PATH             JSON tool definitions file

Generation:
  --examples INT           Number of examples to generate (default: 500)
  --distribution TEXT      Example type percentages
  --system-prompt TEXT     Custom system prompt (auto-inferred if omitted)
  --personas TEXT          Comma-separated user persona descriptions

Output:
  --output PATH            Output file path (default: dataset.jsonl)
  --format CHOICE          openai | hermes | chatml | raw

LLM:
  --provider CHOICE        openai | anthropic (default: openai)
  --model TEXT             Model identifier (default: provider default)
  --api-key TEXT           API key (overrides environment variable)

Validation:
  --strict                 Reject examples with warnings
  --max-retries INT        Max regeneration attempts per failed example (default: 2)

Performance:
  --workers INT            Parallel threads for generation (default: 1)

Other:
  --seed INT               Random seed for reproducibility
  --verbose                Print progress and validation details
  --dry-run                Parse spec and show context without generating
```

## Examples

```bash
# Generate with Anthropic
callset --spec travel_api.yaml \
    --provider anthropic \
    --examples 500 \
    --output travel_data.jsonl

# Use JSON tool definitions with Hermes format
callset --tools my_tools.json --format hermes --output hermes_data.jsonl

# Reproducible generation with verbose output
callset --spec api.yaml --seed 42 --verbose --max-retries 3

# Strict validation
callset --spec api.yaml --strict --examples 200
```

## Sample Output

Each line in the output JSONL is a complete conversation. Here's a happy path example (formatted for readability):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful travel assistant."},
    {"role": "user", "content": "I need a flight from Paris to Tokyo on March 15th."},
    {"role": "assistant", "content": null, "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "search_flights",
          "arguments": "{\"departure\": \"CDG\", \"arrival\": \"NRT\", \"date\": \"2026-03-15\"}"
        }
      }
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"flights\": [{\"flight_id\": \"FL-9281\", \"airline\": \"Air France\", \"price\": 847.00, \"departure_time\": \"10:30\", \"arrival_time\": \"06:45+1\"}]}"},
    {"role": "assistant", "content": "I found a flight from CDG to NRT on March 15th. Air France FL-9281 departs at 10:30 and arrives at 06:45 the next day for $847. Would you like to book it?"}
  ]
}
```

And a multi-step example where the user searches, then books:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful travel assistant."},
    {"role": "user", "content": "Find me hotels in Berlin for next weekend."},
    {"role": "assistant", "content": null, "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "search_hotels",
          "arguments": "{\"location\": \"Berlin\", \"check_in\": \"2026-03-28\", \"check_out\": \"2026-03-30\"}"
        }
      }
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"hotels\": [{\"hotel_id\": \"H-4820\", \"name\": \"Hotel Adlon\", \"price_per_night\": 195.00}, {\"hotel_id\": \"H-7731\", \"name\": \"Motel One\", \"price_per_night\": 79.00}]}"},
    {"role": "assistant", "content": "I found 2 options in Berlin for March 28-30:\n1. Hotel Adlon - $195/night\n2. Motel One - $79/night\nWhich one would you like to book?"},
    {"role": "user", "content": "Book the Motel One please."},
    {"role": "assistant", "content": null, "tool_calls": [
      {
        "id": "call_2",
        "type": "function",
        "function": {
          "name": "book_hotel",
          "arguments": "{\"hotel_id\": \"H-7731\", \"check_in\": \"2026-03-28\", \"check_out\": \"2026-03-30\"}"
        }
      }
    ]},
    {"role": "tool", "tool_call_id": "call_2", "content": "{\"booking_id\": \"BK-19204\", \"status\": \"confirmed\", \"total\": 158.00}"},
    {"role": "assistant", "content": "Your booking at Motel One is confirmed! Booking ID: BK-19204, total: $158.00 for 2 nights. Enjoy Berlin!"}
  ]
}
```

## Validation

Every generated conversation passes through three validation layers:

1. **Format** — JSON structure, role ordering, tool call shape
2. **Schema** — Tool calls checked against the source JSON Schema (required params, types, enums)
3. **Semantic** — Type-specific rules (e.g., clarification examples must ask before calling, refusals must not contain tool calls)

Conversations that fail validation are discarded and regenerated up to `--max-retries` times.

## License

See [LICENSE](LICENSE) for details.
