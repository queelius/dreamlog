# Repository Guidelines

## Project Structure & Module Organization
DreamLog's core logic and LLM orchestration live in `dreamlog/` (e.g. `engine.py`, `llm_providers.py`, `prompt_template_system.py`). Tests reside in `tests/` and mirror module boundaries (`test_engine.py`, `test_unification.py`). MkDocs-friendly docs sit in `docs/`; rendered assets are ignored. Reference configurations (`dreamlog_config.yaml`, `llm_config.json`, `ollama_config.py`) and reusable prompts live at the repo root. Use `examples/` and `integrations/` for runnable showcases and external adapters; keep experimental notebooks or scripts inside `dev/` or `temp-experiments/`.

## Build, Test, and Development Commands
Run `make dev-install` the first time to create `.venv` and install dev tooling. Use `make test` for the focused regression suite and `make test-all` for full coverage. Format and lint with `make format` and `make lint` (Black, pylint, mypy). Start the REPL via `make repl` or `make repl-llm` when pairing with an LLM backend. Publish docs locally with `make docs-serve`.

## Coding Style & Naming Conventions
Target Python ≥3.8 with Black’s 88-character line limit and 4-space indentation; never commit unformatted code. Keep functions and variables in `snake_case`, classes in `PascalCase`, and module-level constants in `UPPER_SNAKE_CASE`. Preserve typing discipline—`mypy` runs in strict mode for `dreamlog/`, so add precise type hints and keep protocols in dedicated modules. Prefer small, composable functions; expose public entrypoints through `dreamlog.__init__` and register new providers in `llm_providers.py`.

## Testing Guidelines
Pytest drives the suite (`pytest.ini` enforces `test_*.py`, `Test*` classes, `test_*` functions). Coverage must stay above 80%; `make test-all` enforces `--cov=dreamlog --cov-fail-under=80`. Mark slower cases with `@pytest.mark.slow`, network-dependent ones with `integration` or `llm`, and skip them in quick runs. When adding features, pair new logic with deterministic unit tests and update fakes in `tests/mock_provider.py`.

## Commit & Pull Request Guidelines
Use imperative, sentence-case commit subjects that describe intent (e.g. `Add persistent learning API`). Group related changes per commit and include a short body documenting rationale when touching multiple subsystems. Pull requests should summarize behavioural impact, link issues, and mention configuration/documentation updates. Paste `make test` or `make test-all` output when the change is risky, and add screenshots or REPL transcripts if they clarify new agent workflows.

## Configuration & Integration Tips
Keep secrets out of version control; reference `.env` or local overrides instead of editing `llm_config.json`. Adjust `dreamlog_config.yaml` for datastore or agent defaults, and document non-default values in the PR description. If an integration requires network services (e.g., Ollama at `192.168.0.225`), call it out in `docs/` or `integrations/README.md` to avoid surprising contributors.
