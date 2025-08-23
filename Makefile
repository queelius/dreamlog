# DreamLog Project Makefile
# Simplifies common development tasks

.PHONY: help venv install test test-all test-ollama clean format lint repl docs

# Default target
help:
	@echo "DreamLog Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup:"
	@echo "  make venv          Create virtual environment"
	@echo "  make install       Install dependencies in venv"
	@echo "  make dev-install   Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run quick tests"
	@echo "  make test-all      Run all tests with coverage"
	@echo "  make test-ollama   Test Ollama integration (requires Ollama at 192.168.0.225)"
	@echo "  make test-prompt   Test prompt template system"
	@echo "  make test-retry    Test retry system"
	@echo ""
	@echo "Development:"
	@echo "  make repl          Start DreamLog REPL"
	@echo "  make format        Format code with black"
	@echo "  make lint          Run linting checks"
	@echo "  make clean         Clean up generated files"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          Generate documentation"
	@echo "  make docs-serve    Serve documentation locally"

# Virtual environment setup
venv:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Virtual environment created at .venv"
	@echo "Activate with: source .venv/bin/activate"

# Install dependencies
install: venv
	@echo "Installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Dependencies installed"

# Install with development dependencies
dev-install: venv
	@echo "Installing development dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install pytest pytest-cov black pylint mypy
	@echo "Development dependencies installed"

# Run quick tests
test:
	@echo "Running quick tests..."
	.venv/bin/python -m pytest tests/test_terms.py tests/test_unification.py -v

# Run all tests with coverage
test-all:
	@echo "Running all tests with coverage..."
	.venv/bin/python -m pytest tests/ -v --cov=dreamlog --cov-report=term-missing

# Test Ollama integration
test-ollama:
	@echo "Testing Ollama integration..."
	.venv/bin/python test_template_ollama.py

# Test prompt templates
test-prompt:
	@echo "Testing prompt template system..."
	.venv/bin/python test_prompt_templates.py

# Test retry system
test-retry:
	@echo "Testing retry system..."
	.venv/bin/python test_retry_system.py

# Start REPL
repl:
	@echo "Starting DreamLog REPL..."
	.venv/bin/python -m dreamlog.repl

# Start REPL with LLM support
repl-llm:
	@echo "Starting DreamLog REPL with LLM..."
	.venv/bin/python -m dreamlog.repl --llm

# Format code
format:
	@echo "Formatting code with black..."
	.venv/bin/black dreamlog/ tests/ *.py

# Lint code
lint:
	@echo "Running linting checks..."
	.venv/bin/pylint dreamlog/
	.venv/bin/mypy dreamlog/ --ignore-missing-imports

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf *.egg-info
	@echo "Cleanup complete"

# Generate documentation
docs:
	@echo "Generating documentation..."
	.venv/bin/mkdocs build

# Serve documentation
docs-serve:
	@echo "Serving documentation at http://localhost:8000..."
	.venv/bin/mkdocs serve

# Install and run everything for first-time setup
setup: dev-install
	@echo ""
	@echo "Setup complete! DreamLog is ready to use."
	@echo ""
	@echo "Activate the virtual environment with:"
	@echo "  source .venv/bin/activate"
	@echo ""
	@echo "Then try:"
	@echo "  make test      # Run tests"
	@echo "  make repl      # Start REPL"
	@echo ""

# Quick test of core functionality
smoke-test:
	@echo "Running smoke test..."
	@.venv/bin/python -c "from dreamlog import DreamLogEngine; e = DreamLogEngine(); print('✓ DreamLog imports successfully')"
	@.venv/bin/python -c "from dreamlog.pythonic import dreamlog; kb = dreamlog(); kb.fact('test', 'works'); print('✓ Pythonic API works')"
	@echo "Smoke test passed!"

# Run experiments
experiment:
	@echo "Running Ollama experiment..."
	.venv/bin/python experiment_ollama.py

# Check if requirements are up to date
check-deps:
	@echo "Checking for outdated dependencies..."
	.venv/bin/pip list --outdated

# Update requirements
update-deps:
	@echo "Updating dependencies..."
	.venv/bin/pip install --upgrade -r requirements.txt
	.venv/bin/pip freeze > requirements.txt
	@echo "Requirements updated"