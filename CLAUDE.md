# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DreamLog is a Prolog-like logic programming language with S-expression syntax and LLM integration. It implements **wake-sleep cycles** inspired by DreamCoder: during wake phases, it answers queries via SLD resolution (optionally generating rules via LLM for undefined predicates); during sleep phases, it compresses the knowledge base through anti-unification, subsumption elimination, predicate invention, and body pattern extraction. The core thesis is that compression is learning (Solomonoff induction): the shortest KB that preserves deductive closure is the best generalization.

## Commands

### Testing
```bash
# Run all tests (~931 collected, 80% coverage enforced by pyproject.toml)
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

### Running the TUI / MCP server
```bash
dreamlog                              # TUI entry point
dreamlog-mcp                          # MCP server entry point
python -m dreamlog.tui --llm          # TUI with LLM support
python -m dreamlog.tui --kb file.json # Load knowledge base on startup
```
Both entry points are registered in `pyproject.toml` under `[project.scripts]`.

## Architecture

### Layer 1: Term Algebra
`terms.py`, `types.py`, `factories.py`, `utils.py`

Immutable term objects (`Atom`, `Variable`, `Compound`). Factory functions in `factories.py` use variadic args: `compound("f", atom("a"), atom("b"))`. The `Compound` constructor takes a list: `Compound("f", [atom("a"), atom("b")])`. Do not confuse them.

### Layer 2: Knowledge Representation
`knowledge.py`, `prefix_parser.py`

`Fact` (ground terms) and `Rule` (head-body Horn clauses) stored in `KnowledgeBase` with functor/arity indexing. KB supports `copy()`/`restore_from()` for atomic rollback, value-based removal, `replace_facts()` for atomic swaps, and per-clause usage counters (`record_usage`/`get_usage`).

### Layer 3: Inference Engine
`unification.py`, `evaluator.py`, `proof_tree.py`

Robinson's unification with occurs check in three modes (standard, one-way/matching, subsumption). `clause_subsumes()` for rule-level subsumption (same-body-length). `PrologEvaluator` implements SLD resolution with:
- **`not/1`**: Negation as failure with floundering guard and unknown hook suppression
- **`call/N`**: Meta-predicate for runtime predicate dispatch (`call(parent, X, Y)` becomes `parent(X, Y)`)
- **`has_solution(term)`**: Generator-based derivability check (stops at first solution)
- **`query_with_proof(goals)`**: Yields `(Solution, ProofNode)` pairs; `proof_tree.py` defines `ProofNode` (goal, clause, children, depth) and `ProofLog` for collecting/analyzing proof structure.
- Usage tracking: records which facts/rules fire during resolution
- `max_total_calls`: per-evaluator call budget that persists across `query()` invocations on the same evaluator. Verification suites use one fresh evaluator per query so individual transitive queries cannot starve others.

### Layer 4: LLM Integration Pipeline
When an undefined predicate is queried, the LLM pipeline activates:

1. **Hook** (`llm_hook.py`): Intercepts undefined predicates, orchestrates the pipeline
2. **Providers** (`llm_client.py`): `LLMClient` wraps OpenAI/Anthropic/Ollama SDKs with lazy imports and per-provider default models. Default: Anthropic Haiku via `MY_ANTHROPIC_API_KEY`. Supports `provider`, `api_key_env`, `timeout`. `LLMUsage` tracks `calls`, `input_tokens`, `output_tokens`; `estimated_cost()` uses a `MODEL_PRICING` table.
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

`KnowledgeBaseDreamer().dream(kb)` runs eight compression operations (A-H) in sequence, with a ninth (Operation I, recursive closure discovery) available behind the `discover_recursion` flag:

- **Operation A (Subsumption elimination)**: Remove rules subsumed by more general rules (same-body-length restriction). Remove facts subsumed by bodyless rules.
- **Operation B (Redundant fact pruning)**: Remove facts derivable from rules + other facts. Batch removal with one-at-a-time fallback for mutual dependencies.
- **Operation C (Fact generalization with exceptions)**: Group facts by functor/arity, partition by argument position (subgroup discovery), find guard predicates, compute exceptions, apply when MDL criterion passes. Uses `not/1` for exception clauses.
- **Operation D (Predicate invention)**: Extract skeleton fingerprints from rule sets (`skeleton.py`), group structurally identical predicates, build parameterized invented predicates with `call/N` dispatch. Discovers abstractions like transitive closure autonomously.
- **Operation E (Body pattern extraction)**: Find common contiguous sub-goal sequences across rule bodies, compute interface variables, extract as named predicates. Discovers abstractions like "grandparent chain."
- **Operation F (Dead clause pruning)**: Remove facts/rules with 0 usage after sufficient wake-phase queries. Uses per-clause frequency counters from the evaluator.
- **Operation G (LLM-assisted compression)**: Ask LLM (default: Anthropic Haiku) to propose cross-functor rules the symbolic operations can't discover. Pipeline: (1) build prompt with facts + predicate counts + directionality constraints, (2) parse JSON response with nested term support (including `not/1`), (3) filter cyclic rules via DFS on functor dependency graph, (4) separate helper predicates from main rules, (5) evaluate each main rule (with helpers) against original KB, (6) false-positive check: reject rules that derive ground terms absent from KB, (7) verify combined set. Supports `not/1` via helper predicate pattern (e.g., `has_non_vegan(X) :- uses(X,Y), vegan(Y,false)` + `vegan_recipe(X) :- recipe(X), not(has_non_vegan(X))`).
- **Operation H (Lemma caching)**: Add frequently-derived terms as facts for faster resolution.
- **Operation I (Recursive closure discovery)**: Synthesize a right-recursive transitive-closure rule (e.g., `ancestor`) from base-relation facts, replacing the stored closure with a two-clause recursive definition. Flag-gated (`discover_recursion`, default off) so the default pipeline keeps zero drift; an open-world subset gate (`min_closure_coverage`, tau=0.5) lifts the exact-match restriction to recover partial closures.

All operations use MDL (Minimum Description Length) scoring and are verified against a test suite of positive/negative queries with atomic rollback on failure. Post-Op-G verification uses bounded evaluators (`max_total_calls`, scaled with KB size) to prevent combinatorial explosion from LLM-proposed rules that create resolution loops. `VerificationSuite.verify()` creates a fresh evaluator per query so one slow transitive query cannot starve the budget for subsequent ones.

**`KnowledgeBaseDreamer` constructor parameters:**
- `llm_client`: optional; if None, only symbolic ops (A-F, H) run.
- `min_group_size` (default 3): minimum facts per group before Op C triggers.
- `shared_structure_threshold` (default 0.1): anti-unification score threshold.
- `max_prompt_facts` (default 50): round-robin sample size for Op G's LLM prompt. Raise for KBs with many predicates (200+ works well for domains with 7-10 predicates; see `experiments/ex25_generalization.py`).
- `open_world` (default False): when True, Op G's false-positive check (reject rules that derive ground terms absent from the KB) is disabled. Closed-world default preserves the zero-drift safety guarantee for production compression; open-world is the opt-in for holdout-style evaluation where the goal is precisely to recover absent facts. **Important gotcha**: `open_world=True` only disables the FP enumeration check; the synthetic negative queries in `build_verification_suite` are still enforced, so it is a *partial* relaxation of closed-world.
- `discover_recursion` (default False): enable Operation I (symbolic recursive transitive-closure discovery). Off by default; flag-gated so the default pipeline has zero drift.
- `min_base_facts` (default 3): minimum base-relation facts before Operation I synthesizes a recursive rule (guards against spurious closure detection on tiny relations).
- `disable_op_c` (default False): skip Operation C (fact generalization). Off by default; used by the EX28 within-predicate LLM-only ablation to isolate the LLM's contribution.

**Key modules:**
- `anti_unification.py`: Plotkin's algorithm (dual of unification). `anti_unify`, `anti_unify_many`, `node_count`, `shared_structure` scoring.
- `skeleton.py`: Rule-set structural fingerprinting. `extract_skeleton` normalizes variable names, classifies functors as SELF/PARAM, produces hashable `RuleSetSkeleton`.
- `kb_dreamer.py`: the dreamer facade: schedule, verification suite, thin per-op orchestrators (method names preserved for tests/monkeypatching).
- `compression/`: the MDL machinery. `dl.py` (description length: the clause-count default plus a rename-invariant, parameter-free bits-based two-part code, selected via `dl_mode`; the bits code prices each clause by its symbol occurrences and each distinct symbol by its arity via an Elias gamma length), `proposal.py` (Proposal), `gate.py` (the single trial/verify/commit gate: apply_proposal, apply_batch_with_fallback, apply_batch_staged_combined), `policies.py` (per-op acceptance lifted verbatim), `generators/` (reduce=A+B, generalize=C, factor=D+E, closure=I, llm=G proposal stage), `maintenance.py` (F+H: cache pair outside the MDL objective).

### Layer 6: High-Level APIs
- `engine.py`: `DreamLogEngine` combines all components. Entry point for most operations.
- `pythonic.py`: `DreamLog` fluent chainable API with `RuleBuilder` (`.when()`, `.and_()`, `.build()`)
- `tui.py`: Terminal UI with `/dream`, `/dream --dry-run`, `/analyze`, `/dream-status` commands

### Examples and benchmarks
- `examples/*.dl`: S-expression KBs (e.g., `family.dl`, `family2.dl`). The `.dl` extension is just a convention for DreamLog source files; load with `dreamlog --kb examples/family.dl` or `kb.parse(open("...").read())`.
- `examples/wake_sleep_demo.py`, `examples/persistent_learning_demo.py`: end-to-end demos covering the full wake-sleep loop.
- `benchmarks/sleep_cycle_bench.py` + `benchmarks/baseline.json`: timing/compression baselines for sleep operations. Use these (not the experiments/) when comparing performance regressions.

### Integrations
- `integrations/mcp/dreamlog_mcp_server.py`: Model Context Protocol server (FastMCP v2). Exposes 5 tools (`assert`, `query`, `dream`, `explain`, `status`) and 2 resources (`dreamlog://kb`, `dreamlog://stats`). Configured via env vars: `DREAMLOG_STORE`, `DREAMLOG_LLM_BUDGET`, `DREAMLOG_DREAM_THRESHOLD`.
- `integrations/mcp/knowledge_store.py`: `KnowledgeStore` class wrapping `DreamLogEngine` with disk persistence (atomic writes via envelope format `{"version": 1, "kb": [...], "metadata": {...}}`), LLM budget tracking, dream-readiness advisory, and session metadata. Both `query()` and `explain()` accept a `max_total_calls` parameter (default 5000) to bound resolution after dreams discover broad-head rules.

### Experiments and paper
- `experiments/experiment_registry.yaml`: single source of truth for experimental provenance. Each entry has `title`, `date`, `script`, `status`, `motivation`, `method`, `key_result`, `implications`, and optional `depends_on`. Update it when adding a new experiment or revising an old one. The registry spans EX01-EX41; per-run records (meta/results/summary JSON capturing git commit, environment, package versions, seeds, LLM accounting, and output hashes) are written under `experiments/data/exNN/runs/<id>/` by the `experiments/_harness.py` `experiment_run` context manager (`data/` is gitignored).
- `experiments/ex25_generalization.py`: canonical generalization test on a family tree. Defines shared helpers (`build_kb`, `is_derivable`, `dream_kb`, `holdout_split`, `get_llm_client`) that sibling experiments (ex25b, ex25c) import rather than duplicate.
- `experiments/ex25b_novel_generalization.py`: same protocol on a synthetic crafting domain with invented terms (lumite, vexal, etc.) the LLM has never seen, plus a raw-LLM baseline.
- `experiments/ex25c_holdout_sweep.py`: holdout ratio sweep; uses `dream_kb(..., open_world=True)`.
- `paper/dreamlog_paper.tex`: the manuscript ("Compression Enables Generalization"). Build with `cd paper && pdflatex ... && bibtex ... && pdflatex ... (x2)`. References are in `paper/references.bib`.
- Paper has a Zenodo deposit: concept DOI `10.5281/zenodo.19490027`, v1 DOI `10.5281/zenodo.19490028`. The deposit is tracked in metafunctor's `pubs_db.json` under slug `dreamlog-compression`. Mint new versions via `mf pubs zenodo register dreamlog-compression --publish` (see `.zenodo.json` and `CITATION.cff`). The `mf` CLI lives in `~/github/repos/mf/`; install with `pip install -e ~/github/repos/mf`.
- `paper/slides/dreamlog_slides.tex`: 5-page Beamer deck for advisor coauthoring discussions (Madrid theme, no external deps).

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
- Sleep cycle tests in `tests/test_sleep_cycle.py` cover the sleep operations (A through I) with integration tests

### Design Specs and Plans
- Design specs in `docs/superpowers/specs/`
- Implementation plans in `docs/superpowers/plans/`
- Each feature follows: brainstorm, spec, review, plan, implement, verify
