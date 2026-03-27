# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DreamLog is a Prolog-like logic programming language with S-expression syntax and LLM integration. It implements **wake-sleep cycles** inspired by DreamCoder: during wake phases, it answers queries via SLD resolution (optionally generating rules via LLM for undefined predicates); during sleep phases, it compresses the knowledge base through anti-unification, subsumption elimination, predicate invention, and body pattern extraction. The core thesis is that compression is learning (Solomonoff induction): the shortest KB that preserves deductive closure is the best generalization.

## Commands

### Testing
```bash
# Run all tests (689 tests, 80% coverage enforced by pyproject.toml)
python -m pytest tests/ -v

# Run specific test file, class, or method
python -m pytest tests/test_unification.py -v
python -m pytest tests/test_sleep_cycle.py::TestOperationD -v

# Coverage is ON by default (pyproject.toml addopts includes --cov)
# Filter by marker
python -m pytest tests/ -m "not slow" -v    # Exclude slow tests
python -m pytest tests/ -m llm -v           # Only LLM integration tests
```

### Makefile Shortcuts
```bash
make test          # Quick regression (terms + unification only)
make test-all      # Full suite with coverage
make format        # black dreamlog/ tests/
make lint          # pylint + mypy (strict mode)
make repl          # Start TUI
make repl-llm      # TUI with LLM backend
make docs-serve    # MkDocs at localhost:8000
```

### Running the TUI
```bash
dreamlog                              # Entry point (pyproject.toml scripts)
python -m dreamlog.tui --llm          # With LLM support
python -m dreamlog.tui --kb file.json # Load knowledge base on startup
```

## Architecture

### Layer 1: Term Algebra
`terms.py`, `types.py`, `factories.py`, `utils.py`

Immutable term objects (`Atom`, `Variable`, `Compound`). Factory functions in `factories.py` use variadic args: `compound("f", atom("a"), atom("b"))`. The `Compound` constructor takes a list: `Compound("f", [atom("a"), atom("b")])`. Do not confuse them.

### Layer 2: Knowledge Representation
`knowledge.py`, `prefix_parser.py`

`Fact` (ground terms) and `Rule` (head-body Horn clauses) stored in `KnowledgeBase` with functor/arity indexing. KB supports `copy()`/`restore_from()` for atomic rollback, value-based removal, `replace_facts()` for atomic swaps, and per-clause usage counters (`record_usage`/`get_usage`).

### Layer 3: Inference Engine
`unification.py`, `evaluator.py`

Robinson's unification with occurs check in three modes (standard, one-way/matching, subsumption). `clause_subsumes()` for rule-level subsumption (same-body-length). `PrologEvaluator` implements SLD resolution with:
- **`not/1`**: Negation as failure with floundering guard and unknown hook suppression
- **`call/N`**: Meta-predicate for runtime predicate dispatch (`call(parent, X, Y)` becomes `parent(X, Y)`)
- **`has_solution(term)`**: Generator-based derivability check (stops at first solution)
- Usage tracking: records which facts/rules fire during resolution

### Layer 4: LLM Integration Pipeline
When an undefined predicate is queried, the LLM pipeline activates:

1. **Hook** (`llm_hook.py`): Intercepts undefined predicates, orchestrates the pipeline
2. **Providers** (`llm_providers.py`, `llm_http_provider.py`): Protocol-based; supports OpenAI, Anthropic, Ollama
3. **Prompting** (`prompt_template_system.py`, `llm_prompt_templates.py`): Parameterized templates with few-shot learning
4. **RAG** (`example_retriever.py`): KB-aware semantic example retrieval for prompt context
5. **Embeddings** (`embedding_providers.py`, `tfidf_embedding_provider.py`): Ollama embeddings or local TF-IDF fallback
6. **Parsing** (`llm_response_parser.py`): Extracts rules from LLM output
7. **Validation** (`rule_validator.py`, `validation_feedback.py`): Structural validation with feedback generation
8. **Retry** (`correction_retry.py`, `llm_retry_wrapper.py`): Multi-strategy retry with JSON validation and correction feedback
9. **Judging** (`llm_judge.py`): LLM-based verification of generated rules
10. **Config** (`config.py`): Unified YAML/JSON configuration (`DreamLogConfig`, `LLMProviderConfig`, `LLMSamplingConfig`)

**Critical design constraint**: LLMs generate **rules only, never facts**.

### Layer 5: Sleep Cycle (the core innovation)
`kb_dreamer.py`, `anti_unification.py`, `skeleton.py`

`KnowledgeBaseDreamer().dream(kb)` runs six compression operations in sequence:

- **Operation A (Subsumption elimination)**: Remove rules subsumed by more general rules (same-body-length restriction). Remove facts subsumed by bodyless rules.
- **Operation B (Redundant fact pruning)**: Remove facts derivable from rules + other facts. Batch removal with one-at-a-time fallback for mutual dependencies.
- **Operation C (Fact generalization with exceptions)**: Group facts by functor/arity, partition by argument position (subgroup discovery), find guard predicates, compute exceptions, apply when MDL criterion passes. Uses `not/1` for exception clauses.
- **Operation D (Predicate invention)**: Extract skeleton fingerprints from rule sets (`skeleton.py`), group structurally identical predicates, build parameterized invented predicates with `call/N` dispatch. Discovers abstractions like transitive closure autonomously.
- **Operation E (Body pattern extraction)**: Find common contiguous sub-goal sequences across rule bodies, compute interface variables, extract as named predicates. Discovers abstractions like "grandparent chain."
- **Operation F (Dead clause pruning)**: Remove facts/rules with 0 usage after sufficient wake-phase queries. Uses per-clause frequency counters from the evaluator.

All operations use MDL (Minimum Description Length) scoring and are verified against a test suite of positive/negative queries with atomic rollback on failure.

**Key modules:**
- `anti_unification.py`: Plotkin's algorithm (dual of unification). `anti_unify`, `anti_unify_many`, `node_count`, `shared_structure` scoring.
- `skeleton.py`: Rule-set structural fingerprinting. `extract_skeleton` normalizes variable names, classifies functors as SELF/PARAM, produces hashable `RuleSetSkeleton`.
- `kb_dreamer.py`: The dreamer orchestrator, verification suite (`build_verification_suite`, `extend_verification_for_rules`), and all six operations.

### Layer 6: High-Level APIs
- `engine.py`: `DreamLogEngine` combines all components. Entry point for most operations.
- `pythonic.py`: `DreamLog` fluent chainable API with `RuleBuilder` (`.when()`, `.and_()`, `.build()`)
- `tui.py`: Terminal UI with `/dream`, `/dream --dry-run`, `/analyze`, `/dream-status` commands

### Integrations
- `integrations/mcp/dreamlog_mcp_server.py`: Model Context Protocol server

## Development Guidelines

### No Backward Compatibility
**This project prioritizes code purity and simplicity. Backward compatibility is NOT needed or desired.**
- Break old APIs freely when refactoring
- Remove deprecated code immediately
- Never add compatibility shims or aliases

### S-expression Format
- **S-expressions**: `(parent john mary)`, `(grandparent X Z)`
- **JSON prefix**: `["parent", "john", "mary"]`
- **Rules**: `(grandparent X Z) :- (parent X Y), (parent Y Z)`
- Old `{"functor": ..., "args": ...}` format is DEPRECATED. Do not use.

### Style
Python >=3.8, Black (88 chars), mypy strict mode on `dreamlog/`. Protocols in dedicated modules. Expose public API through `dreamlog/__init__.py`.

### Testing Strategy
- Use `MockLLMProvider` from `tests/mock_provider.py` for deterministic LLM testing
- Tests focus on **behavior, not implementation** (no testing private attributes)
- Mark with `@pytest.mark.slow`, `@pytest.mark.integration`, or `@pytest.mark.llm` as appropriate
- Coverage must stay above 80% (`--cov-fail-under=80` in pyproject.toml)
- Sleep cycle tests in `tests/test_sleep_cycle.py` cover all six operations with integration tests

### Design Specs and Plans
- Design specs in `docs/superpowers/specs/`
- Implementation plans in `docs/superpowers/plans/`
- Each feature follows: brainstorm, spec, review, plan, implement, verify
