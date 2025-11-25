# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DreamLog is a Prolog-like logic programming language with S-expression syntax and LLM integration. It implements **wake-sleep cycles** for continuous self-improvement: exploiting existing knowledge during "wake" and exploring new abstractions through "dreaming." When undefined predicates are encountered during query evaluation, it can automatically generate relevant facts and rules using LLMs.

## Commands

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_prefix_parser.py -v
python -m pytest tests/test_unification.py -v

# Run with coverage
python -m pytest tests/ --cov=dreamlog

# Skip slow/integration/LLM tests
python -m pytest tests/ -m "not slow"
python -m pytest tests/ -m "not llm"
```

### Linting & Formatting
```bash
black dreamlog/           # Format code
mypy dreamlog/            # Type check
pylint dreamlog/          # Lint
```

### Running the TUI
```bash
# Interactive TUI (main interface)
python -m dreamlog.tui

# With LLM support
python -m dreamlog.tui --llm

# With specific model
python -m dreamlog.tui --llm --provider ollama --model phi4-mini:latest

# Load knowledge base on startup
python -m dreamlog.tui --load knowledge.dl
```

### Pythonic API
```python
from dreamlog.pythonic import dreamlog

kb = dreamlog()  # or dreamlog(llm_provider="ollama")
kb.fact("parent", "john", "mary") \
  .fact("parent", "mary", "alice") \
  .rule("grandparent", ["X", "Z"]).when("parent", ["X", "Y"]).and_("parent", ["Y", "Z"]).build()

for result in kb.query("grandparent", "X", "alice"):
    print(result["X"])  # john
```

## Architecture

### Core Flow
`pythonic.py` → `engine.py` → `evaluator.py` → `unification.py` / `knowledge.py`

When LLM is enabled: `evaluator.py` → `llm_hook.py` → `llm_providers.py`

### Key Components

- **Terms** (`terms.py`): Immutable `Atom`, `Variable`, `Compound` classes with factory functions `atom()`, `var()`, `compound()`
- **Parser** (`prefix_parser.py`): `parse_s_expression()` for `(parent john mary)`, `parse_prefix_notation()` for JSON arrays
- **Unification** (`unification.py`): Functional API with `unify()`, `match()`, `subsumes()`
- **Evaluator** (`evaluator.py`): `PrologEvaluator` implements SLD resolution with backtracking
- **Engine** (`engine.py`): `DreamLogEngine` combines KB, evaluator, and LLM hook
- **Pythonic API** (`pythonic.py`): `DreamLog` class with fluent method chaining
- **TUI** (`tui.py`): Rich terminal interface with `/ask`, `/find-all`, `/facts`, `/rules`, etc.
- **Config** (`config.py`): `DreamLogConfig` with YAML/JSON support, searches `~/.dreamlog/config.yaml`

### LLM Integration
- `llm_hook.py`: Hook called when evaluator encounters undefined predicates
- `llm_providers.py`: Provider protocol and `create_provider()` factory
- `llm_retry_wrapper.py`: Automatic retry with JSON parsing correction
- `prompt_template_system.py`: RAG-based example selection for prompts

## Development Guidelines

### No Backward Compatibility
This project prioritizes code purity. Break old APIs freely, remove deprecated code immediately, don't add compatibility shims.

### Syntax
S-expressions are the canonical format:
- Facts: `(parent john mary)`
- Rules: `(grandparent X Z) :- (parent X Y), (parent Y Z)`
- Variables: uppercase (X, Y, Person)

### Adding LLM Providers
1. Implement the `LLMProvider` protocol in `llm_providers.py`
2. Provider must implement `generate(prompt: str) -> str` and `get_metadata() -> dict`
3. Register in `create_provider()` factory

### Testing
Use `MockLLMProvider` from `tests/mock_provider.py` for deterministic LLM tests.

## Configuration

Config is loaded from (in order): `~/.dreamlog/config.yaml`, `~/.dreamlog/config.json`, `./dreamlog_config.yaml`

```yaml
provider:
  provider: ollama
  model: phi4-mini:latest
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 500

sampling:
  max_facts: 20
  max_rules: 15
  strategy: related  # "related", "random", or "recent"

llm_enabled: false
debug_enabled: false
```