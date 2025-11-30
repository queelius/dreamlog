# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DreamLog is a Prolog-like logic programming language with S-expression syntax and LLM integration. It implements **wake-sleep cycles** inspired by neuroscience: during wake phases, it exploits existing knowledge to answer queries efficiently; during sleep phases, it explores optimizations through "dreaming" to discover abstractions, compress redundancies, and generalize patterns.

## Commands

### Testing
```bash
# Run all tests (276+ tests, ~50% coverage target)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_prefix_parser.py -v
python -m pytest tests/test_unification.py -v

# Run specific test class or method
python -m pytest tests/test_unification.py::TestAdvancedUnification -v
python -m pytest tests/test_knowledge_advanced.py::TestKnowledgeBaseAdvanced::test_rule_operations -v

# Run with coverage
python -m pytest tests/ --cov=dreamlog --cov-report=term-missing

# Run specific modules with coverage
python -m pytest tests/test_terms.py tests/test_unification.py --cov=dreamlog.terms --cov=dreamlog.unification --cov-report=term-missing

# Run excluding slow tests
python -m pytest tests/ -m "not slow" -v

# Run only LLM integration tests
python -m pytest tests/ -m llm -v

# Run only integration tests
python -m pytest tests/ -m integration -v
```

### Running the TUI
```bash
# Interactive TUI (Terminal User Interface)
python -m dreamlog.tui
# or
dreamlog

# With LLM support
python -m dreamlog.tui --llm

# Load knowledge base on startup
python -m dreamlog.tui --kb knowledge.json

# With custom config
python -m dreamlog.tui --config myconfig.json
```

### Running Integrations
```bash
# MCP Server (Model Context Protocol)
python integrations/mcp/dreamlog_mcp_server.py --port 8765
```

## Architecture

### Core Components

1. **Terms System** (`dreamlog/terms.py`)
   - Immutable term objects: `Atom`, `Variable`, `Compound`
   - Factory functions: `atom()`, `var()`, `compound()`
   - Internal prefix notation representation

2. **Parser** (`dreamlog/prefix_parser.py`)
   - `parse_s_expression()`: Parse S-expressions like `(parent john mary)`
   - `parse_prefix_notation()`: Parse JSON arrays like `["parent", "john", "mary"]`
   - `term_to_sexp()`, `term_to_prefix_json()`: Convert terms to strings

3. **Knowledge Base** (`dreamlog/knowledge.py`)
   - `Fact`: Ground terms
   - `Rule`: Head-body implications with Horn clause logic
   - `KnowledgeBase`: Storage with functor-based indexing for efficient retrieval

4. **Unification** (`dreamlog/unification.py`)
   - Functional API: `unify()`, `match()`, `subsumes()`
   - Multiple unification modes (standard, one-way, subsumption)
   - `Unifier` class for stateful operations
   - Robinson's unification algorithm with occurs check

5. **Query Evaluator** (`dreamlog/evaluator.py`)
   - `PrologEvaluator`: SLD resolution with backtracking
   - Generator-based lazy evaluation
   - Returns `Solution` objects with variable bindings

6. **LLM Integration**
   - `llm_hook.py`: Hook system for automatic rule generation when undefined predicates are encountered
   - `llm_providers.py`: Base provider protocol
   - `llm_http_provider.py`: HTTP-based providers (OpenAI, Anthropic, Ollama)
   - `llm_prompt_templates.py`: Customizable prompt templates
   - `llm_response_parser.py`: Parse LLM-generated rules
   - `llm_judge.py`: LLM-based verification and validation
   - `correction_retry.py`: Correction-based retry logic for robust parsing
   - `config.py`: Unified configuration management
   - **Important**: LLMs generate rules only, never facts (by design)

7. **Main Engine** (`dreamlog/engine.py`)
   - `DreamLogEngine`: High-level API combining all components
   - Methods: `add_fact()`, `add_rule()`, `query()`, `save_to_prefix()`, `load_from_prefix()`
   - Entry point for most operations

8. **Pythonic API** (`dreamlog/pythonic.py`)
   - `DreamLog` class: Fluent, chainable Python interface
   - `QueryResult`: Pythonic access to bindings with dict-like and attribute access
   - `RuleBuilder`: Fluent rule construction with `.when()`, `.and_()`, `.build()`
   - Seamless Python integration (iteration, context managers, dataframes)

9. **TUI** (`dreamlog/tui.py`)
   - Comprehensive Terminal User Interface with rich formatting
   - Commands for knowledge management, queries, sleep-phase operations
   - LLM and embedding model control
   - Debugging and tracing support

10. **Wake-Sleep Architecture** (`dreamlog/kb_dreamer.py`)
    - `KnowledgeBaseDreamer`: Knowledge optimization through "dreaming"
    - `DreamSession`: Results from dream cycles (compression ratios, insights, verification)
    - `DreamInsight`: Individual optimization discoveries (compression, abstraction, generalization)
    - `DreamVerification`: Behavior preservation verification
    - **Wake Phase**: Efficient query answering using existing knowledge (exploitation)
    - **Sleep Phase**: Knowledge optimization through dreaming (exploration)

### Key Design Patterns

- **Immutable Terms**: All term objects are immutable for safe sharing across threads
- **Generator-based Evaluation**: Query results yielded lazily for memory efficiency
- **Hook System**: LLM integration through hooks, not hardcoded dependencies
- **Protocol-based Providers**: LLM and embedding providers follow protocols for easy extension
- **Functional Unification API**: Stateless functions alongside stateful `Unifier` class

## Development Guidelines

### IMPORTANT: No Backward Compatibility
**This project prioritizes code purity and simplicity. Backward compatibility is NOT needed or desired.**
- Break old APIs freely when refactoring
- Remove deprecated code immediately
- Never add compatibility shims or aliases
- Keep the codebase clean and maintainable

### Current Format: S-expressions and Prefix Notation
The project uses clean S-expression/prefix notation:
- **S-expressions**: `(parent john mary)`, `(grandparent X Z)`
- **JSON prefix**: `["parent", "john", "mary"]`
- **Rules**: `(grandparent X Z) :- (parent X Y), (parent Y Z)`
- Old mixed format with `{"functor": ..., "args": ...}` is DEPRECATED and should not be used

### Testing Strategy
- Use `MockLLMProvider` from `tests/mock_provider.py` for deterministic testing
- Tests should focus on **behavior, not implementation** (avoid testing private attributes like `_dimension`)
- Tests may need updating when APIs change - that's acceptable
- Add new tests for new features
- Run coverage to identify gaps
- Use `@pytest.mark.integration` for end-to-end LLM tests

### Key Implementation Details

**Unification**: Robinson's algorithm with occurs check. Three modes:
- Standard: Full bidirectional unification
- One-way: Like pattern matching (second term is pattern)
- Subsumption: Check if one term is more general than another

**Query Evaluation**: SLD resolution (Selection rule with Linear resolution for Definite clauses) with depth-first search and backtracking. Goals evaluated left-to-right.

**LLM Hook**: When an undefined predicate is queried, the hook:
1. Samples relevant context from KB
2. Constructs a prompt using templates
3. Sends to LLM provider
4. Parses response (with retry if malformed)
5. Adds generated **rules** to KB (facts are ignored by design)
6. Retries the query

**Configuration**: Unified system in `config.py` supporting both YAML and JSON:
- `DreamLogConfig`: Top-level configuration
- `LLMProviderConfig`: Provider-specific settings
- `LLMSamplingConfig`: Context sampling for prompts

## Configuration Files

- **pyproject.toml**: Project metadata, dependencies, test configuration
- **pytest.ini**: Test markers (slow, integration, llm)
- **llm_config.json**: LLM provider configuration (typically Ollama with phi3:7b)

## Examples

See `examples/wake_sleep_demo.py` for a comprehensive demonstration of the wake-sleep cycle architecture.
