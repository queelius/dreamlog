# Quick Start

Get up and running with DreamLog in 5 minutes!

## 1. Interactive REPL

The fastest way to try DreamLog:

```bash
# Start the REPL
python -m dreamlog.repl

# With LLM support
python -m dreamlog.repl --llm
```

### REPL Commands

```
dreamlog> (parent john mary)
✓ Added fact: (parent john mary)

dreamlog> (parent mary alice)
✓ Added fact: (parent mary alice)

dreamlog> (grandparent X Z) :- (parent X Y), (parent Y Z)
✓ Added rule: (grandparent X Z) :- (parent X Y), (parent Y Z)

dreamlog> (grandparent john Z)?
Found 1 solution(s):
  1. Z=alice

dreamlog> :help
[Shows all commands]

dreamlog> :save family.dreamlog
✓ Saved to family.dreamlog

dreamlog> :exit
Goodbye!
```

## 2. Python Integration

### Basic Usage

```python
from dreamlog.pythonic import dreamlog

# Create a DreamLog instance
jl = dreamlog()

# Add facts
jl.fact("parent", "john", "mary")
jl.fact("parent", "mary", "alice")
jl.fact("parent", "tom", "bob")

# Add a rule
jl.rule("grandparent", ["X", "Z"]) \
  .when("parent", ["X", "Y"]) \
  .and_("parent", ["Y", "Z"]) \
  .build()

# Query
for result in jl.query("grandparent", "john", "Z"):
    print(f"John is grandparent of {result['Z']}")
# Output: John is grandparent of alice
```

### With LLM Integration

```python
from dreamlog.pythonic import dreamlog

# Create with LLM support (auto-detects from environment)
jl = dreamlog(llm_provider="openai")

# Add minimal knowledge
jl.fact("person", "socrates")

# Query for undefined knowledge
# LLM will generate: "All persons are mortal" rule
for result in jl.query("mortal", "socrates"):
    print("Socrates is mortal!")
```

## 3. S-Expression Syntax

### Facts

```lisp
; Simple facts
(parent john mary)
(age john 42)
(likes alice programming)

; Facts with multiple arguments
(teaches professor algebra students)
(located paris france europe)
```

### Rules

```lisp
; Basic rule
(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)

; Recursive rule
(ancestor X Y) :- (parent X Y)
(ancestor X Z) :- (parent X Y), (ancestor Y Z)

; Complex rule with multiple conditions
(can_graduate Student) :- 
    (enrolled Student Program),
    (completed_credits Student Credits),
    (required_credits Program Required),
    (greater_or_equal Credits Required)
```

### Queries

```lisp
; Ground query (all constants)
(parent john mary)?

; Query with variables
(parent john X)?
(parent X mary)?
(parent X Y)?

; Complex query
(grandparent john X)?
```

## 4. Complete Example

Let's build a simple family tree system:

```python
from dreamlog.pythonic import dreamlog

# Initialize with mock LLM for testing
jl = dreamlog(llm_provider="mock", knowledge_domain="family")

# Define family relationships
jl.facts(
    ("parent", "john", "mary"),
    ("parent", "john", "tom"),
    ("parent", "mary", "alice"),
    ("parent", "mary", "bob"),
    ("parent", "tom", "charlie"),
    ("male", "john"),
    ("male", "tom"),
    ("male", "bob"),
    ("male", "charlie"),
    ("female", "mary"),
    ("female", "alice")
)

# Define rules
jl.parse("(grandparent X Z) :- (parent X Y), (parent Y Z)")
jl.parse("(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)")
jl.parse("(brother X Y) :- (sibling X Y), (male X)")
jl.parse("(sister X Y) :- (sibling X Y), (female X)")

# Queries
print("John's grandchildren:")
for r in jl.query("grandparent", "john", "X"):
    print(f"  - {r['X']}")

print("\nAlice's siblings:")
for r in jl.query("sibling", "alice", "X"):
    print(f"  - {r['X']}")

print("\nBrothers:")
for r in jl.query("brother", "X", "Y"):
    print(f"  {r['X']} is brother of {r['Y']}")

# Save the knowledge base
jl.save("family_tree.dreamlog")

# Statistics
print(f"\nKnowledge base: {jl.stats}")
```

Output:
```
John's grandchildren:
  - alice
  - bob
  - charlie

Alice's siblings:
  - bob

Brothers:
  bob is brother of alice

Knowledge base: {'num_facts': 11, 'num_rules': 4, 'functors': [...], 'total_items': 15}
```

## 5. REST API Server

Start the API server:

```bash
python integrations/api/dreamlog_api_server.py --port 8000
```

Query via HTTP:

```bash
# Add a fact
curl -X POST http://localhost:8000/facts \
  -H "Content-Type: application/json" \
  -d '{"fact": "(parent john mary)"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "(parent john X)"}'
```

## 6. Jupyter Integration

In a Jupyter notebook:

```python
%load_ext dreamlog.jupyter.dreamlog_magic
%dreamlog_init --llm

%dreamlog_fact (parent john mary)
%dreamlog_fact (parent mary alice)

%dreamlog_query (parent X Y)
```

## Next Steps

- [Tutorial](tutorial.md) - Deep dive into DreamLog
- [Python API](../api/pythonic.md) - Full API documentation
- [LLM Integration](../guide/llm.md) - Configure AI-powered reasoning
- [Examples](../examples/family.md) - More complex examples