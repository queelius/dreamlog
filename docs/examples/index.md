# DreamLog Examples

This section contains practical examples demonstrating various DreamLog features and use cases.

## Basic Examples

### [Family Relations](family_relations.md)
Classic family tree reasoning with parent, grandparent, and sibling relationships.

### [Academic Database](academic_database.md)
Student enrollment, course prerequisites, and grade calculations.

### [Graph Algorithms](graph_algorithms.md)
Path finding, connectivity, and graph traversal using logic programming.

## Advanced Examples

### [Wake-Sleep Optimization](wake_sleep_optimization.md)
Demonstrating knowledge base compression and abstraction through dream cycles.

### [LLM Knowledge Generation](llm_generation.md)
Automatic fact and rule generation using language models.

### [Real-time Reasoning](realtime_reasoning.md)
Building a reactive system with DreamLog and WebSockets.

## Quick Examples

### Simple Facts and Queries

```python
from dreamlog import dreamlog

# Create knowledge base
kb = dreamlog()

# Add facts
kb.parse("""
(parent john mary)
(parent john bob)
(parent mary alice)
(parent bob charlie)
""")

# Query
results = kb.query("parent", "john", "X")
for r in results:
    print(f"john is parent of {r['X']}")
# Output:
# john is parent of mary
# john is parent of bob
```

### Rules and Inference

```python
# Add rules
kb.parse("""
(grandparent X Z) :- (parent X Y), (parent Y Z)
(ancestor X Y) :- (parent X Y)
(ancestor X Z) :- (parent X Y), (ancestor Y Z)
""")

# Query with inference
results = kb.query("grandparent", "john", "X")
for r in results:
    print(f"john is grandparent of {r['X']}")
# Output:
# john is grandparent of alice
# john is grandparent of charlie
```

### LLM Integration

```python
from dreamlog import dreamlog
from dreamlog.llm_providers import OpenAIProvider

# Create KB with LLM support
kb = dreamlog(llm_provider=OpenAIProvider())

# Query undefined predicate - triggers LLM
results = kb.query("healthy", "alice")
# LLM generates relevant facts and rules

# The generated knowledge is now available
results = kb.query("healthy", "X")
```

### Dream Cycles

```python
from dreamlog.kb_dreamer import KnowledgeBaseDreamer

# Create dreamer
dreamer = KnowledgeBaseDreamer(llm_provider)

# Run optimization
session = dreamer.dream(
    kb,
    dream_cycles=3,
    focus="compression",
    verify=True
)

print(f"Compression ratio: {session.compression_ratio:.1%}")
print(f"Insights found: {len(session.insights)}")

# Apply optimizations if verified
if session.verification.preserved:
    optimized_kb = dreamer.apply_insights(kb, session.insights)
```

## Domain-Specific Examples

### Medical Diagnosis

```python
kb.parse("""
(symptom patient1 fever)
(symptom patient1 cough)
(symptom patient2 headache)

(diagnosis X flu) :- 
    (symptom X fever),
    (symptom X cough)

(diagnosis X migraine) :-
    (symptom X headache),
    (symptom X nausea)
""")

# Diagnose patient
results = kb.query("diagnosis", "patient1", "X")
# Result: flu
```

### Configuration Management

```python
kb.parse("""
(service web nginx)
(service db postgres)
(depends web db)

(requires X memory 2gb) :- (service X nginx)
(requires X memory 4gb) :- (service X postgres)

(can_start X) :- 
    (service X Y),
    (depends X Z) -> (running Z),
    (available_memory M),
    (requires X memory Required),
    (greater M Required)
""")

# Check if service can start
results = kb.query("can_start", "web")
```

### Natural Language Processing

```python
kb.parse("""
(word the determiner)
(word cat noun)
(word sat verb)
(word on preposition)
(word mat noun)

(phrase X noun_phrase) :-
    (sequence X [Det, Noun]),
    (word Det determiner),
    (word Noun noun)

(phrase X verb_phrase) :-
    (sequence X [Verb, Prep, NP]),
    (word Verb verb),
    (word Prep preposition),
    (phrase NP noun_phrase)
""")

# Parse sentence structure
results = kb.query("phrase", "[the, cat]", "X")
# Result: noun_phrase
```

## Interactive Examples

### REPL Session

```bash
$ python -m dreamlog.repl

DreamLog> (parent john mary)
Fact added: (parent john mary)

DreamLog> (parent mary alice)
Fact added: (parent mary alice)

DreamLog> (grandparent X Z) :- (parent X Y), (parent Y Z)
Rule added: (grandparent X Z) :- ...

DreamLog> ?- (grandparent john X)
X = alice

DreamLog> dream
Running dream cycle...
Found 3 insights:
  1. Compression: Merged similar rules (2.0x)
  2. Abstraction: Found general pattern (1.5x)
Verification: ✓ Behavior preserved

DreamLog> save family.dl
Knowledge base saved to family.dl
```

### Jupyter Notebook

```python
%%dreamlog
(parent john mary)
(parent mary alice)
(grandparent X Z) :- (parent X Y), (parent Y Z)

query: (grandparent john X)
```

Output:
```
Results for (grandparent john X):
  X = alice
```

## Performance Examples

### Optimizing Large Knowledge Bases

```python
import time

# Load large KB
kb = load_knowledge_base("large_dataset.dl")
print(f"Initial size: {len(kb.facts)} facts, {len(kb.rules)} rules")

# Measure query performance
start = time.time()
results = list(kb.query("complex_predicate", "X", "Y"))
initial_time = time.time() - start
print(f"Initial query time: {initial_time:.3f}s")

# Optimize with dreaming
dreamer = KnowledgeBaseDreamer(llm_provider)
session = dreamer.dream(kb, cycles=5, focus="all")

# Apply optimizations
optimized_kb = dreamer.apply_insights(kb, session.insights)
print(f"Optimized size: {len(optimized_kb.facts)} facts, "
      f"{len(optimized_kb.rules)} rules")

# Measure improved performance
start = time.time()
results = list(optimized_kb.query("complex_predicate", "X", "Y"))
optimized_time = time.time() - start
print(f"Optimized query time: {optimized_time:.3f}s")
print(f"Speedup: {initial_time/optimized_time:.2f}x")
```

## Integration Examples

### Web Application

```python
from flask import Flask, request, jsonify
from dreamlog import dreamlog

app = Flask(__name__)
kb = dreamlog()

@app.route('/query', methods=['POST'])
def query():
    q = request.json['query']
    results = list(kb.query_from_string(q))
    return jsonify(results)

@app.route('/fact', methods=['POST'])
def add_fact():
    fact = request.json['fact']
    kb.parse(fact)
    return jsonify({"status": "added"})

if __name__ == '__main__':
    app.run(port=5000)
```

### Discord Bot

```python
import discord
from dreamlog import dreamlog

client = discord.Client()
kb = dreamlog()

@client.event
async def on_message(message):
    if message.content.startswith('!query'):
        query = message.content[7:]
        results = list(kb.query_from_string(query))
        
        if results:
            response = '\n'.join(str(r) for r in results)
        else:
            response = "No results found"
        
        await message.channel.send(response)

client.run('YOUR_BOT_TOKEN')
```

## Testing Examples

See individual example files for complete, runnable code with tests and documentation.