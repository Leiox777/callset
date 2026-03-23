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

## Validation

Every generated conversation passes through three validation layers:

1. **Format** — JSON structure, role ordering, tool call shape
2. **Schema** — Tool calls checked against the source JSON Schema (required params, types, enums)
3. **Semantic** — Type-specific rules (e.g., clarification examples must ask before calling, refusals must not contain tool calls)

Conversations that fail validation are discarded and regenerated up to `--max-retries` times.

## License

See [LICENSE](LICENSE) for details.
