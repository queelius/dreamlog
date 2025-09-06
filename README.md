# DreamLog: Logic Programming with Wake-Sleep Cycles

> A reasoning engine that dreams to improve itself, inspired by how the brain consolidates knowledge during sleep.

## What is DreamLog?

DreamLog is a revolutionary logic programming language that implements **wake-sleep cycles** for continuous self-improvement. Like the human brain during REM sleep, DreamLog alternates between:

- 🌞 **Wake Phase**: Exploiting existing knowledge to answer queries efficiently
- 🌙 **Sleep Phase**: Exploring new abstractions, compressions, and generalizations through "dreaming"

Inspired by DreamCoder and neuroscience, DreamLog discovers general principles through compression—following the insight that **simpler explanations covering more cases are likely more true**.

## Key Innovation: Dreaming for Optimization

```python
from dreamlog.pythonic import dreamlog
from dreamlog.kb_dreamer import KnowledgeBaseDreamer

# Wake phase: Use knowledge
kb = dreamlog(llm_provider="openai")
kb.fact("parent", "john", "mary")
kb.fact("parent", "mary", "alice")

# Sleep phase: Dream to optimize
dreamer = KnowledgeBaseDreamer(kb.provider)
session = dreamer.dream(
    kb, 
    dream_cycles=3,           # Multiple REM cycles
    exploration_samples=10,    # Explore different optimizations
    verify=True               # Ensure behavior preservation
)

print(f"Compression achieved: {session.compression_ratio:.1%}")
print(f"Generalization score: {session.generalization_score:.2f}")
```

## Features

### 🧠 Self-Improving Knowledge Base
- Automatically discovers abstractions and patterns
- Compresses redundant rules into general principles
- Verifies that optimizations preserve behavior

### 🤖 LLM-Powered Learning
- Generates missing knowledge when needed
- Context-aware generation based on existing facts
- Multiple provider support (OpenAI, Anthropic, Ollama)

### 🎯 Modern Python Integration
```python
kb = dreamlog()
kb.fact("likes", "alice", "bob") \
  .fact("likes", "bob", "alice") \
  .rule("friends", ["X", "Y"]) \
  .when("likes", ["X", "Y"]) \
  .and_("likes", ["Y", "X"])

for result in kb.query("friends", "X", "Y"):
    print(f"{result.bindings['X']} and {result.bindings['Y']} are friends")
```

### 🔄 Wake-Sleep Architecture
- **Exploitation**: Fast query answering during wake
- **Exploration**: Creative reorganization during sleep
- **Verification**: Ensures improvements don't break existing behavior

## Installation

```bash
pip install dreamlog
```

## Quick Start

```python
from dreamlog.pythonic import dreamlog

# Create a knowledge base
kb = dreamlog()

# Add facts using S-expressions
kb.parse("""
(parent john mary)
(parent mary alice)
(parent bob charlie)
""")

# Add rules
kb.parse("""
(grandparent X Z) :- (parent X Y), (parent Y Z)
""")

# Query
for result in kb.query("grandparent", "X", "alice"):
    print(f"{result.bindings['X']} is Alice's grandparent")

# Enable LLM for undefined predicates
kb_with_llm = dreamlog(llm_provider="openai")
# Now queries for undefined predicates will generate knowledge automatically
```

## The Philosophy

DreamLog embodies the principle that **intelligence emerges from the interplay of exploration and exploitation**:

1. **Consolidation**: Strengthen important patterns
2. **Abstraction**: Find general principles  
3. **Compression**: Achieve more with less
4. **Creativity**: Explore novel reorganizations

This isn't just logic programming with LLMs bolted on—it's a fundamentally new paradigm where the system's knowledge representation **evolves through use**.

## Documentation

See the [full documentation](docs/) for:
- [Getting Started Guide](docs/getting-started/installation.md)
- [Tutorial](docs/getting-started/tutorial.md)
- [S-Expression Syntax](docs/guide/syntax.md)
- [Wake-Sleep Cycles](docs/guide/dreaming.md)
- [API Reference](docs/api/)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use DreamLog in your research, please cite:

```bibtex
@software{dreamlog2025,
  title = {DreamLog: Logic Programming with Wake-Sleep Cycles},
  author = {queelius},
  year = {2025},
  url = {https://github.com/queelius/dreamlog}
}
```

---

*Built by dreamers who believe reasoning systems should sleep, perchance to dream*