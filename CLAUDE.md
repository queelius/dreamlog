# CLAUDE.md - DreamLog Project Guide

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DreamLog is a Prolog-like logic programming language with JSON syntax and LLM integration for automatic knowledge generation. When undefined terms are encountered during query evaluation, it can automatically generate relevant facts and rules using LLMs.

## Commands

### Testing
```bash
# Run all tests (NOTE: Many tests need updating to new APIs)
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_prefix_parser.py -v  # New parser tests
python -m pytest tests/test_llm.py -v

# Run with coverage
python -m pytest tests/ --cov=dreamlog
```

### Running the REPL
```bash
# Interactive REPL
python -m dreamlog.repl

# With LLM support
python -m dreamlog.repl --llm

# Load knowledge base on startup
python -m dreamlog.repl --kb knowledge.json
```

### Running Integrations
```bash
# REST API Server with REPL endpoint
python integrations/api/dreamlog_api_server.py --port 8000

# MCP Server
python integrations/mcp/dreamlog_mcp_server.py --port 8765

# VS Code Language Server
python integrations/vscode/dreamlog_language_server.py
```

## Architecture

### Core Components

1. **Terms System** (`dreamlog/terms.py`)
   - Base class `Term` with `Atom`, `Variable`, `Compound` implementations
   - Factory functions: `atom()`, `var()`, `compound()`
   - Uses prefix notation internally

2. **Parser** (`dreamlog/prefix_parser.py`)
   - `parse_s_expression()`: Parse S-expressions like `(parent john mary)`
   - `parse_prefix_notation()`: Parse JSON arrays like `["parent", "john", "mary"]`
   - `term_to_sexp()`, `term_to_prefix_json()`: Convert terms to strings

3. **Knowledge Base** (`dreamlog/knowledge.py`)
   - `Fact`: Ground terms
   - `Rule`: Head-body implications
   - `KnowledgeBase`: Storage with functor-based indexing

4. **Unification** (`dreamlog/unification.py`)
   - Functional API: `unify()`, `match()`, `subsumes()`
   - Multiple unification modes (standard, one-way, subsumption)
   - Debugging and constraint support
   - `Unifier` class for stateful operations

5. **Query Evaluator** (`dreamlog/evaluator.py`)
   - `PrologEvaluator`: SLD resolution with backtracking
   - Returns `Solution` objects with variable bindings

6. **LLM Integration**
   - `llm_hook.py`: Hook system for automatic knowledge generation
   - `llm_providers.py`: Base provider classes
   - `llm_http_provider.py`: HTTP-based providers (OpenAI, Anthropic, Ollama)
   - `llm_prompt_templates.py`: Customizable prompt templates
   - `llm_config.py`: Configuration management

7. **Main Engine** (`dreamlog/engine.py`)
   - `DreamLogEngine`: High-level API combining all components
   - Methods: `add_fact()`, `add_rule()`, `query()`, `save_to_prefix()`, `load_from_prefix()`

8. **REPL** (`dreamlog/repl.py`)
   - Interactive command-line interface
   - Commands for loading/saving, querying, adding facts/rules
   - Syntax highlighting and tab completion

### Key Design Patterns

- **Immutable Terms**: All term objects are immutable for safe sharing
- **Generator-based Evaluation**: Query results are yielded lazily
- **Hook System**: LLM integration through hooks, not hardcoded
- **Protocol-based Providers**: LLM providers follow a protocol for easy extension

## Development Guidelines

### IMPORTANT: No Backward Compatibility
**This project prioritizes code purity and simplicity. Backward compatibility is NOT needed or desired.**
- When refactoring, feel free to break old APIs
- Remove deprecated code paths immediately
- Don't add compatibility shims or aliases
- Keep the codebase clean and maintainable

### Current Format: S-expressions and Prefix Notation
The project has migrated from mixed JSON format to clean S-expression/prefix notation:
- S-expressions: `(parent john mary)`, `(grandparent X Z)`
- JSON prefix: `["parent", "john", "mary"]`
- Old mixed format with `{"functor": ..., "args": ...}` is DEPRECATED

### Adding New LLM Providers
1. Implement the `LLMProvider` protocol in `dreamlog/llm_providers.py`
2. Use the new `HTTPLLMAdapter` in `dreamlog/llm_http_provider.py` for HTTP-based providers
3. Supported providers: OpenAI, Anthropic, Ollama, Custom HTTP endpoints

### Testing
- Tests may need updating when APIs change - that's fine
- Use `MockLLMProvider` for deterministic testing
- Focus on testing current functionality, not backward compatibility

### Core Modules
- `prefix_parser.py`: Main parser for S-expressions and prefix notation
- `unification.py`: Enhanced with functional API (unify, match, subsumes)
- `llm_prompt_templates.py`: Customizable prompt templates
- `repl.py`: Interactive REPL for command-line usage
- `llm_http_provider.py`: Lightweight HTTP-based LLM integration

### Integrations
Located in `integrations/` folder:
- MCP (Model Context Protocol) server
- REST API with WebSocket REPL
- Jupyter notebook magic commands
- VS Code Language Server Protocol
- LLM fine-tuning data generator

## Current Configuration

- **LLM Config**: `llm_config.json` - Configured for Ollama with phi3:7b model
- **Test Config**: `pytest.ini` - Test paths and markers configured
- **Dependencies**: Minimal runtime deps (requests), pytest for testing