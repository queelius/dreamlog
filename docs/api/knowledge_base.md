# Knowledge Base Module

The `dreamlog.knowledge` module provides classes for storing and managing facts and rules.

## Classes

### `KnowledgeBase`

Main storage for facts and rules with efficient indexing.

```python
class KnowledgeBase:
    def __init__(self)
    def add_fact(self, fact: Fact) -> None
    def add_rule(self, rule: Rule) -> None
    def get_facts(self, functor: Optional[str] = None) -> List[Fact]
    def get_rules(self, functor: Optional[str] = None) -> List[Rule]
    def remove_fact(self, fact: Fact) -> bool
    def remove_rule(self, rule: Rule) -> bool
    def clear(self) -> None
    @property
    def facts(self) -> List[Fact]
    @property
    def rules(self) -> List[Rule]
    @property
    def stats(self) -> Dict[str, Any]
```

#### Methods

##### `add_fact(fact: Fact) -> None`

Add a fact to the knowledge base.

```python
from dreamlog.knowledge import KnowledgeBase, Fact
from dreamlog.terms import compound, atom

kb = KnowledgeBase()
fact = Fact(compound("parent", atom("john"), atom("mary")))
kb.add_fact(fact)
```

##### `add_rule(rule: Rule) -> None`

Add a rule to the knowledge base.

```python
from dreamlog.knowledge import Rule
from dreamlog.terms import compound, var

rule = Rule(
    head=compound("grandparent", var("X"), var("Z")),
    body=[
        compound("parent", var("X"), var("Y")),
        compound("parent", var("Y"), var("Z"))
    ]
)
kb.add_rule(rule)
```

##### `get_facts(functor: Optional[str] = None) -> List[Fact]`

Get facts, optionally filtered by functor.

```python
# Get all facts
all_facts = kb.get_facts()

# Get facts with specific functor
parent_facts = kb.get_facts("parent")
```

##### `get_rules(functor: Optional[str] = None) -> List[Rule]`

Get rules, optionally filtered by head functor.

```python
# Get all rules
all_rules = kb.get_rules()

# Get rules with specific head functor
grandparent_rules = kb.get_rules("grandparent")
```

##### `stats` Property

Get statistics about the knowledge base.

```python
stats = kb.stats
# {
#     'num_facts': 10,
#     'num_rules': 5,
#     'total_items': 15,
#     'functors': {'parent', 'grandparent', 'sibling'},
#     'fact_functors': {'parent'},
#     'rule_functors': {'grandparent', 'sibling'}
# }
```

### `Fact`

Represents a ground fact (term without variables).

```python
class Fact:
    def __init__(self, term: Term)
    
    @property
    def term(self) -> Term
    
    @property
    def functor(self) -> str
    
    @property
    def arity(self) -> int
    
    @classmethod
    def from_prefix(cls, data: Any) -> 'Fact'
    
    def to_prefix(self) -> List[Any]
```

#### Example

```python
from dreamlog.knowledge import Fact
from dreamlog.terms import compound, atom

# Create from term
fact = Fact(compound("parent", atom("john"), atom("mary")))

# Create from prefix notation
fact2 = Fact.from_prefix(["fact", ["parent", "john", "mary"]])

# Access properties
print(fact.functor)  # "parent"
print(fact.arity)    # 2

# Convert to prefix
prefix = fact.to_prefix()  # ["fact", ["parent", "john", "mary"]]
```

### `Rule`

Represents a logic rule with head and body.

```python
class Rule:
    def __init__(self, head: Term, body: List[Term])
    
    @property
    def head(self) -> Term
    
    @property
    def body(self) -> List[Term]
    
    @property
    def functor(self) -> str
    
    @property
    def arity(self) -> int
    
    @classmethod
    def from_prefix(cls, data: Any) -> 'Rule'
    
    def to_prefix(self) -> List[Any]
```

#### Example

```python
from dreamlog.knowledge import Rule
from dreamlog.terms import compound, var

# Create rule: (grandparent X Z) :- (parent X Y), (parent Y Z)
rule = Rule(
    head=compound("grandparent", var("X"), var("Z")),
    body=[
        compound("parent", var("X"), var("Y")),
        compound("parent", var("Y"), var("Z"))
    ]
)

# Create from prefix notation
rule2 = Rule.from_prefix([
    "rule",
    ["grandparent", "X", "Z"],
    [["parent", "X", "Y"], ["parent", "Y", "Z"]]
])

# Access properties
print(rule.functor)  # "grandparent"
print(rule.arity)    # 2
print(len(rule.body))  # 2
```

## Indexing

The knowledge base uses functor-based indexing for efficient retrieval:

```python
kb = KnowledgeBase()

# Add many facts
for i in range(1000):
    kb.add_fact(Fact(compound("fact", atom(f"item_{i}"))))

# Fast retrieval by functor
facts = kb.get_facts("fact")  # O(1) lookup + O(n) for n matching facts
```

## Persistence

### Saving to File

```python
from dreamlog.knowledge import save_knowledge_base

# Save to JSON file
save_knowledge_base(kb, "knowledge.json")

# Save to S-expression file
save_knowledge_base(kb, "knowledge.sexp", format="sexp")
```

### Loading from File

```python
from dreamlog.knowledge import load_knowledge_base

# Load from JSON file
kb = load_knowledge_base("knowledge.json")

# Load from S-expression file
kb = load_knowledge_base("knowledge.sexp", format="sexp")
```

## Knowledge Base Operations

### Merging Knowledge Bases

```python
kb1 = KnowledgeBase()
kb2 = KnowledgeBase()

# Add facts to both
kb1.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
kb2.add_fact(Fact(compound("parent", atom("mary"), atom("alice"))))

# Merge kb2 into kb1
for fact in kb2.facts:
    kb1.add_fact(fact)
for rule in kb2.rules:
    kb1.add_rule(rule)
```

### Filtering

```python
# Get facts matching a pattern
def get_facts_about(kb, entity):
    results = []
    for fact in kb.facts:
        if entity in str(fact.term):
            results.append(fact)
    return results

johns_facts = get_facts_about(kb, "john")
```

### Validation

```python
def validate_knowledge_base(kb):
    """Check for common issues"""
    issues = []
    
    # Check for duplicate facts
    seen = set()
    for fact in kb.facts:
        if fact.term in seen:
            issues.append(f"Duplicate fact: {fact.term}")
        seen.add(fact.term)
    
    # Check for ground terms in facts
    for fact in kb.facts:
        if not fact.term.get_vars().isempty():
            issues.append(f"Fact contains variables: {fact.term}")
    
    return issues
```

## Memory Management

### Clearing the Knowledge Base

```python
# Remove all facts and rules
kb.clear()

# Selective clearing
kb.clear_facts()  # Remove only facts
kb.clear_rules()  # Remove only rules
```

### Removing Individual Items

```python
# Remove a specific fact
fact = Fact(compound("parent", atom("john"), atom("mary")))
removed = kb.remove_fact(fact)  # Returns True if removed

# Remove a specific rule
rule = Rule(...)
removed = kb.remove_rule(rule)  # Returns True if removed
```

## Thread Safety

The default `KnowledgeBase` is not thread-safe. For concurrent access:

```python
import threading

class ThreadSafeKnowledgeBase(KnowledgeBase):
    def __init__(self):
        super().__init__()
        self._lock = threading.RLock()
    
    def add_fact(self, fact):
        with self._lock:
            super().add_fact(fact)
    
    def add_rule(self, rule):
        with self._lock:
            super().add_rule(rule)
    
    def get_facts(self, functor=None):
        with self._lock:
            return super().get_facts(functor)
```

## Best Practices

1. **Use functors for indexing**: Design your facts with meaningful functors for efficient retrieval
2. **Avoid variable facts**: Facts should be ground terms (no variables)
3. **Batch operations**: Add multiple facts/rules at once when possible
4. **Regular persistence**: Save knowledge base periodically if modifications are made
5. **Validate input**: Check facts and rules before adding to avoid inconsistencies