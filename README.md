# JLOG - JSON Prolog with LLM Integration

A modern implementation of a Prolog-like logic programming language using JSON syntax, with built-in hooks for Large Language Model (LLM) integration to automatically generate facts and rules.

## Features

- **JSON-based syntax** - Clean, structured representation of facts, rules, and queries
- **LLM Integration** - Automatic fact/rule generation when undefined terms are encountered
- **Complete Prolog semantics** - Unification with occurs check, SLD resolution, proper backtracking
- **Modern Python** - Type hints, dataclasses, clean architecture
- **Extensible** - Easy to add new LLM providers and customize behavior

## Quick Start

```python
from jlog import JLogEngine, atom, var, compound, Fact, Rule
from jlog import LLMHook, MockLLMProvider

# Create engine with LLM integration
llm_provider = MockLLMProvider()
llm_hook = LLMHook(llm_provider, knowledge_domain="family_relationships")
engine = JLogEngine(llm_hook=llm_hook)

# Add some facts
parent_fact = Fact(compound("parent", [atom("alice"), atom("bob")]))
engine.add_fact(parent_fact)

# Add a rule
grandparent_rule = Rule(
    head=compound("grandparent", [var("X"), var("Y")]),
    body=[
        compound("parent", [var("X"), var("Z")]),
        compound("parent", [var("Z"), var("Y")])
    ]
)
engine.add_rule(grandparent_rule)

# Query the knowledge base (queries are lists of terms)
query = [compound("grandparent", [var("A"), var("B")])]
solutions = engine.query(query)

for solution in solutions:
    print(f"A = {solution['A']}, B = {solution['B']}")
```

## Term Creation

JLOG provides factory functions for creating terms:

```python
from jlog import atom, var, compound

# Atoms (constants)
name = atom("alice")
number = atom(42)
boolean = atom(True)

# Variables
x = var("X")
anonymous = var()  # Creates unique variable name

# Compound terms (structures)
parent_relation = compound("parent", [atom("alice"), atom("bob")])
list_term = compound("list", [atom(1), atom(2), atom(3)])
```

## JSON Format Examples

### Facts
```json
{
  "type": "fact",
  "term": {
    "type": "compound",
    "functor": "parent",
    "args": [
      {"type": "atom", "value": "alice"},
      {"type": "atom", "value": "bob"}
    ]
  }
}
```

### Rules
```json
{
  "type": "rule",
  "head": {
    "type": "compound",
    "functor": "grandparent",
    "args": [
      {"type": "variable", "name": "X"},
      {"type": "variable", "name": "Y"}
    ]
  },
  "body": [
    {
      "type": "compound",
      "functor": "parent",
      "args": [
        {"type": "variable", "name": "X"},
        {"type": "variable", "name": "Z"}
      ]
    },
    {
      "type": "compound",
      "functor": "parent",
      "args": [
        {"type": "variable", "name": "Z"},
        {"type": "variable", "name": "Y"}
      ]
    }
  ]
}
```

## LLM Integration

The system can automatically generate facts and rules when encountering undefined terms:

```python
# Empty knowledge base
engine = JLogEngine(llm_hook=llm_hook)

# Query for undefined term
query = [compound("likes", [var("X"), atom("pizza")])]
solutions = engine.query(query)

# LLM automatically generates relevant facts about who likes pizza
# Solutions are returned based on generated knowledge
```

## Loading Knowledge from JSON

```python
# Load from JSON file
engine.load_from_json("knowledge_base.json")

# Load from JSON string
json_data = '''[
  {
    "type": "fact",
    "term": {
      "type": "compound",
      "functor": "parent",
      "args": [
        {"type": "atom", "value": "alice"},
        {"type": "atom", "value": "bob"}
      ]
    }
  }
]'''
engine.load_from_json_string(json_data)

# Save knowledge base
engine.save_to_json("output.json")
```

## Working with Substitutions

```python
from jlog import Substitution, unify

# Create terms
term1 = compound("likes", [var("X"), atom("pizza")])
term2 = compound("likes", [atom("alice"), var("Y")])

# Unify terms
result = unify(term1, term2, Substitution())
if result:
    print(f"Unified with substitution: {result}")
    # Apply substitution to a term
    unified_term = term1.apply_substitution(result)
    print(f"Result: {unified_term}")
```

## Running the Demo

```bash
# Basic demos
python demo.py

# Interactive mode
python demo.py --interactive
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_terms.py -v
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_llm.py -v
```

## LLM Providers

The system supports different LLM providers:

### Mock Provider (for testing)
```python
from jlog import MockLLMProvider
provider = MockLLMProvider()
```

### Custom LLM Provider
```python
from jlog import LLMProvider

class CustomLLMProvider(LLMProvider):
    def generate_response(self, prompt: str, context: dict) -> str:
        # Your LLM integration here
        return "generated_response"

provider = CustomLLMProvider()
```

## Advanced Features

### Cycle Detection
The engine automatically detects infinite recursion:

```python
# This would create infinite recursion without cycle detection
rule = Rule(
    head=compound("infinite", [var("X")]),
    body=[compound("infinite", [var("X")])]
)
engine.add_rule(rule)

# Query safely terminates
solutions = list(engine.query([compound("infinite", [atom("test")])]))
```

### Knowledge Base Indexing
Facts and rules are automatically indexed by functor for efficient retrieval:

```python
# Engine automatically builds indexes for fast lookup
engine.add_fact(Fact(compound("parent", [atom("a"), atom("b")])))
engine.add_fact(Fact(compound("likes", [atom("a"), atom("pizza")])))

# Queries only examine relevant facts/rules
solutions = list(engine.query([compound("parent", [var("X"), var("Y")])]))
```

## Architecture

- `jlog/terms.py` - Term data structures and operations
- `jlog/unification.py` - Unification algorithm with occurs check
- `jlog/knowledge_base.py` - Fact and rule storage with indexing
- `jlog/evaluator.py` - Query evaluation and SLD resolution
- `jlog/llm_hook.py` - LLM integration system
- `jlog/engine.py` - High-level engine API
- `jlog/__init__.py` - Public API exports
- `demo.py` - Demonstration scripts
- `tests/` - Comprehensive test suite

## Use Cases

1. **Knowledge Base Expansion** - Automatically expand domain knowledge using LLMs
2. **Educational Tools** - Interactive logic programming with AI assistance
3. **Rapid Prototyping** - Quickly build logic-based systems with AI-generated rules
4. **Research** - Explore the intersection of symbolic AI and neural language models

## License

MIT License - see LICENSE file for details.
