# Contributing to DreamLog

Thank you for your interest in contributing to DreamLog! This document provides guidelines and instructions for contributing to the project.

## Philosophy

DreamLog prioritizes **code purity and simplicity** over backward compatibility. We welcome contributions that:
- Simplify and clean up the codebase
- Add meaningful new features
- Improve performance
- Enhance documentation
- Fix bugs

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of logic programming concepts
- Familiarity with S-expressions

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/queelius/dreamlog.git
   cd dreamlog
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

Follow the coding standards below and ensure your changes align with the project philosophy.

### 3. Write/Update Tests

- Add tests for new features in `tests/`
- Update existing tests if APIs change
- Ensure all tests pass: `pytest tests/ -v`

### 4. Update Documentation

- Update docstrings for any modified functions/classes
- Update relevant `.md` files in `docs/`
- Add examples if introducing new features

### 5. Run Quality Checks

```bash
# Run tests
pytest tests/ -v

# Check test coverage
pytest tests/ --cov=dreamlog --cov-report=html

# Format code (if using black)
black dreamlog/ tests/

# Type checking (if using mypy)
mypy dreamlog/
```

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add prompt template categories for dream exploration

- Add compression, abstraction, analogy categories
- Implement template sampling strategy
- Update tests for new functionality"
```

Commit message format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `style:` Code style changes

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Link to any related issues
- Examples of usage (if applicable)

## Coding Standards

### Python Style

- Follow PEP 8
- Use descriptive variable names
- Keep functions focused and small
- Document complex logic with comments

### DreamLog Specific

- **Immutable Terms**: Never modify term objects
- **Generator-based**: Use generators for query results
- **Prefix Notation**: Use S-expressions `(parent john mary)` or JSON arrays `["parent", "john", "mary"]`
- **No Backward Compatibility**: Feel free to break old APIs for cleaner code

### Example Code Style

```python
def parse_s_expression(text: str) -> Term:
    """
    Parse an S-expression into a Term.
    
    Args:
        text: S-expression string like "(parent john mary)"
        
    Returns:
        Parsed Term object
        
    Raises:
        ParseError: If syntax is invalid
    """
    # Implementation here
    pass
```

## Areas for Contribution

### High Priority

1. **Prompt Template Categories** (see `docs/FUTURE_VISION.md`)
   - Implement different dream categories
   - Add template sampling strategies

2. **Meta-Learning System**
   - Track prompt effectiveness
   - Evolve better prompts over time

3. **Grounding Mechanisms**
   - Implement ground truth anchors
   - Add user feedback integration

### Medium Priority

1. **Performance Optimizations**
   - Improve query evaluation speed
   - Optimize unification algorithm
   - Better indexing strategies

2. **New LLM Providers**
   - Add support for more LLM APIs
   - Implement local model support

3. **Integration Extensions**
   - GraphQL API
   - gRPC support
   - Additional IDE plugins

### Always Welcome

- Bug fixes
- Documentation improvements
- Test coverage increases
- Example programs
- Performance benchmarks

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_unification.py -v

# Run with coverage
pytest tests/ --cov=dreamlog --cov-report=term-missing

# Run only fast tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

```python
import pytest
from dreamlog.terms import atom, var, compound

class TestTerms:
    def test_atom_creation(self):
        a = atom("john")
        assert a.name == "john"
        assert str(a) == "john"
    
    def test_compound_creation(self):
        c = compound("parent", atom("john"), var("X"))
        assert c.functor == "parent"
        assert c.arity == 2
```

### Test Organization

- `test_*.py` files in `tests/` directory
- Group related tests in classes
- Use fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def dream(self, kb: KnowledgeBase, cycles: int = 1) -> DreamSession:
    """
    Run dream cycles to optimize knowledge base.
    
    Args:
        kb: Knowledge base to optimize
        cycles: Number of dream cycles to run
        
    Returns:
        DreamSession containing insights and metrics
        
    Example:
        >>> session = dreamer.dream(kb, cycles=3)
        >>> print(f"Compression: {session.compression_ratio}")
    """
```

### Documentation Files

- **API docs**: In `docs/api/`
- **Guides**: In `docs/guides/`
- **Examples**: In `docs/examples/`
- **Integration docs**: In `docs/integrations/`

## Breaking Changes

Since we **don't maintain backward compatibility**, feel free to:

1. **Rename functions/classes** for clarity
2. **Change API signatures** for simplicity
3. **Remove deprecated code** immediately
4. **Restructure modules** for better organization

Just ensure:
- Tests are updated
- Documentation reflects changes
- Commit message explains the breaking change

## Questions and Support

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive matters

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing opinions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

## Recognition

Contributors will be recognized in:
- The project README
- Release notes for their contributions
- The AUTHORS file (for significant contributions)

Thank you for helping make DreamLog better!