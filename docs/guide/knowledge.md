# Managing Knowledge Bases

Learn how to effectively organize, maintain, and query DreamLog knowledge bases.

## Knowledge Base Structure

### Core Components

A DreamLog knowledge base consists of:

1. **Facts** - Ground truths about the world
2. **Rules** - Conditional relationships and inference patterns  
3. **Indexes** - Efficient lookups by functor
4. **LLM Hook** - Optional AI-powered knowledge generation

```python
from dreamlog.pythonic import dreamlog

kb = dreamlog()
print(kb.stats)
# {'num_facts': 0, 'num_rules': 0, 'functors': [], 'total_items': 0}
```

### Internal Organization

```python
# Facts are indexed by functor for efficient retrieval
kb.fact("parent", "john", "mary")
kb.fact("parent", "mary", "alice")
kb.fact("age", "john", 45)

# Internally organized as:
# fact_index = {
#   "parent": [Fact(parent john mary), Fact(parent mary alice)],
#   "age": [Fact(age john 45)]
# }
```

## Adding Knowledge

### Adding Facts

```python
# Single fact
kb.fact("student", "alice", "cs")

# Multiple facts at once
kb.facts(
    ("student", "bob", "math"),
    ("student", "charlie", "physics"),
    ("professor", "smith", "cs")
)

# From S-expressions
kb.parse("(enrolled alice cs101)")
kb.parse("(grade alice cs101 95)")
```

### Adding Rules

```python
# Using fluent API
kb.rule("grandparent", ["X", "Z"]) \
  .when("parent", ["X", "Y"]) \
  .and_("parent", ["Y", "Z"]) \
  .build()

# From S-expressions
kb.parse("""
(ancestor X Y) :- (parent X Y)
(ancestor X Z) :- (parent X Y), (ancestor Y Z)
""")

# Multiple rules
rules = """
(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)
(uncle X Y) :- (sibling X Z), (parent Z Y), (male X)
(aunt X Y) :- (sibling X Z), (parent Z Y), (female X)
"""
kb.parse(rules)
```

### Batch Loading

```python
# From file
kb.load("knowledge_base.dreamlog")

# From Python data structures
facts_data = [
    ["student", "alice", "cs"],
    ["student", "bob", "math"],
    ["grade", "alice", "cs101", 95]
]

for fact in facts_data:
    kb.fact(*fact)

# From JSON
import json
with open("facts.json") as f:
    data = json.load(f)
    for item in data["facts"]:
        kb.fact(*item)
```

## Querying Knowledge

### Basic Queries

```python
# Ground query (checking existence)
exists = kb.query_exists("parent", "john", "mary")
print(f"John is Mary's parent: {exists}")

# Variable queries
for result in kb.query("parent", "john", "X"):
    print(f"John is parent of {result['X']}")

# Multiple variables
for result in kb.query("parent", "X", "Y"):
    print(f"{result['X']} is parent of {result['Y']}")
```

### Advanced Queries

```python
# Complex queries with multiple goals
results = kb.query_complex([
    ("parent", "X", "Y"),
    ("parent", "Y", "Z"),
    ("age", "X", "Age"),
    ("greater", "Age", 40)
])

for r in results:
    print(f"{r['X']} (age {r['Age']}) is grandparent of {r['Z']}")

# Get first N results
for i, result in enumerate(kb.query("student", "X", "_")):
    if i >= 5:
        break
    print(f"Student {i+1}: {result['X']}")

# Collect all results
all_students = list(kb.query("student", "X", "_"))
print(f"Total students: {len(all_students)}")
```

### Query with Explanations

```python
# Enable tracing for explanations
kb.set_trace(True)

for result in kb.query("grandparent", "john", "X"):
    print(f"Result: {result}")
    print(f"Explanation: {kb.get_last_trace()}")

kb.set_trace(False)
```

## Knowledge Organization

### Namespacing with Functors

```python
# Use prefixes for organization
kb.fact("person:name", "alice", "Alice Smith")
kb.fact("person:age", "alice", 25)
kb.fact("person:email", "alice", "alice@example.com")

kb.fact("course:name", "cs101", "Intro to Programming")
kb.fact("course:credits", "cs101", 3)
kb.fact("course:instructor", "cs101", "smith")
```

### Hierarchical Organization

```python
# Department -> Course -> Section structure
kb.fact("department", "cs", "Computer Science")
kb.fact("course", "cs101", "cs", "Intro Programming")
kb.fact("section", "cs101-01", "cs101", "morning")
kb.fact("section", "cs101-02", "cs101", "afternoon")

# Query hierarchically
kb.parse("""
(courses_in_dept Dept Course) :- 
    (course Course Dept _)

(sections_of_course Course Section) :-
    (section Section Course _)
""")
```

### Temporal Facts

```python
# Add timestamps to facts
from datetime import datetime

def add_temporal_fact(kb, predicate, *args, timestamp=None):
    ts = timestamp or datetime.now().isoformat()
    kb.fact(f"{predicate}_at", *args, ts)
    kb.fact(predicate, *args)  # Current fact

# Usage
add_temporal_fact(kb, "enrolled", "alice", "cs101")
add_temporal_fact(kb, "grade", "alice", "cs101", 95, 
                  timestamp="2024-05-15")

# Query historical data
for r in kb.query("enrolled_at", "alice", "Course", "Time"):
    print(f"Enrolled in {r['Course']} at {r['Time']}")
```

## Modifying Knowledge

### Updating Facts

```python
# DreamLog doesn't have direct update, so we retract and assert
def update_fact(kb, old_fact, new_fact):
    # Remove old fact
    kb.retract(*old_fact)
    # Add new fact
    kb.fact(*new_fact)

# Example
update_fact(kb, 
    ("age", "alice", 25),
    ("age", "alice", 26))
```

### Retracting Knowledge

```python
# Remove specific fact
kb.retract("enrolled", "alice", "cs101")

# Remove all facts matching pattern
kb.retract_all("enrolled", "alice", "_")

# Clear all facts for a functor
kb.clear_functor("temp_data")
```

### Rule Management

```python
# Add versioned rules
kb.parse("""
(discount_v1 Student Amount) :- 
    (student Student _),
    (equals Amount 10)
""")

# Replace with new version
kb.remove_rules("discount_v1")
kb.parse("""
(discount_v2 Student Amount) :- 
    (student Student _),
    (honors Student),
    (equals Amount 20)
""")

kb.parse("""
(discount_v2 Student Amount) :- 
    (student Student _),
    (not (honors Student)),
    (equals Amount 10)
""")
```

## Persistence

### Saving Knowledge

```python
# Save to file
kb.save("my_knowledge.dreamlog")

# Save with metadata
metadata = {
    "version": "1.0",
    "created": datetime.now().isoformat(),
    "domain": "academic"
}
kb.save_with_metadata("kb_with_meta.dreamlog", metadata)

# Export as S-expressions
with open("kb.sexp", "w") as f:
    f.write(kb.to_sexp())

# Export as JSON
import json
with open("kb.json", "w") as f:
    json.dump(kb.to_json(), f, indent=2)
```

### Loading Knowledge

```python
# Load from file
kb = dreamlog()
kb.load("my_knowledge.dreamlog")

# Merge multiple knowledge bases
kb1 = dreamlog()
kb1.load("domain1.dreamlog")

kb2 = dreamlog()
kb2.load("domain2.dreamlog")

# Merge kb2 into kb1
kb1.merge(kb2)

# Load with validation
def validate_and_load(kb, filepath):
    temp_kb = dreamlog()
    temp_kb.load(filepath)
    
    # Validate
    if temp_kb.stats['num_facts'] == 0:
        raise ValueError("Empty knowledge base")
    
    # Check for required functors
    required = ["student", "course", "enrolled"]
    functors = temp_kb.stats['functors']
    for req in required:
        if req not in functors:
            raise ValueError(f"Missing required functor: {req}")
    
    kb.merge(temp_kb)
    return True
```

## Knowledge Base Patterns

### Repository Pattern

```python
class StudentRepository:
    def __init__(self, kb):
        self.kb = kb
    
    def add_student(self, id, name, major):
        self.kb.fact("student", id)
        self.kb.fact("student_name", id, name)
        self.kb.fact("student_major", id, major)
        return id
    
    def get_student(self, id):
        student = {"id": id}
        
        for r in self.kb.query("student_name", id, "Name"):
            student["name"] = r["Name"]
        
        for r in self.kb.query("student_major", id, "Major"):
            student["major"] = r["Major"]
        
        return student if "name" in student else None
    
    def find_by_major(self, major):
        students = []
        for r in self.kb.query("student_major", "Id", major):
            students.append(self.get_student(r["Id"]))
        return students

# Usage
repo = StudentRepository(kb)
repo.add_student("s001", "Alice Smith", "CS")
repo.add_student("s002", "Bob Jones", "Math")

cs_students = repo.find_by_major("CS")
```

### Domain Separation

```python
class DomainKB:
    def __init__(self):
        self.domains = {}
    
    def get_domain(self, name):
        if name not in self.domains:
            self.domains[name] = dreamlog()
        return self.domains[name]
    
    def query_across_domains(self, *domains, query):
        results = []
        for domain in domains:
            if domain in self.domains:
                results.extend(self.domains[domain].query(*query))
        return results

# Usage
dkb = DomainKB()

# Academic domain
academic = dkb.get_domain("academic")
academic.fact("student", "alice", "cs")
academic.fact("grade", "alice", "A")

# Financial domain  
financial = dkb.get_domain("financial")
financial.fact("tuition_paid", "alice", True)
financial.fact("balance", "alice", 0)

# Cross-domain query
eligible = []
for r in dkb.query_across_domains("academic", "financial",
                                  query=("student", "X", "_")):
    student = r["X"]
    # Check both domains
    if academic.query_exists("grade", student, "A") and \
       financial.query_exists("tuition_paid", student, True):
        eligible.append(student)
```

### Caching Pattern

```python
class CachedKB:
    def __init__(self, kb):
        self.kb = kb
        self.cache = {}
    
    def query_cached(self, *args):
        key = str(args)
        if key not in self.cache:
            self.cache[key] = list(self.kb.query(*args))
        return self.cache[key]
    
    def invalidate(self, functor=None):
        if functor:
            # Invalidate queries involving this functor
            self.cache = {k: v for k, v in self.cache.items() 
                         if functor not in k}
        else:
            self.cache.clear()
    
    def fact(self, *args):
        self.kb.fact(*args)
        self.invalidate(args[0])  # Invalidate related cache

# Usage
cached_kb = CachedKB(kb)
results1 = cached_kb.query_cached("expensive_query", "X", "Y")
results2 = cached_kb.query_cached("expensive_query", "X", "Y")  # From cache
```

## Performance Optimization

### Indexing Strategies

```python
# Create custom indexes for frequent queries
class IndexedKB:
    def __init__(self, kb):
        self.kb = kb
        self.indexes = {}
    
    def create_index(self, name, functor, position):
        """Create index on specific argument position"""
        index = {}
        for result in self.kb.query(functor, *["_"] * 3):
            key = result[f"arg{position}"]
            if key not in index:
                index[key] = []
            index[key].append(result)
        self.indexes[name] = index
    
    def query_indexed(self, index_name, key):
        return self.indexes.get(index_name, {}).get(key, [])

# Usage
ikb = IndexedKB(kb)
ikb.create_index("by_major", "student", 2)
cs_students = ikb.query_indexed("by_major", "cs")
```

### Query Optimization

```python
# Order goals from most to least selective
# Bad: Generate all pairs then filter
kb.parse("""
(efficient_query X Y) :-
    (person X),
    (person Y),
    (age X AgeX),
    (age Y AgeY),
    (greater AgeX 50),
    (less AgeY 30)
""")

# Good: Filter early
kb.parse("""
(efficient_query X Y) :-
    (age X AgeX),
    (greater AgeX 50),
    (age Y AgeY),
    (less AgeY 30),
    (person X),
    (person Y)
""")
```

## Knowledge Validation

### Consistency Checking

```python
def check_consistency(kb):
    """Check for logical inconsistencies"""
    issues = []
    
    # Check for conflicting facts
    for r in kb.query("age", "X", "Age1"):
        person = r["X"]
        ages = list(kb.query("age", person, "Age"))
        if len(ages) > 1:
            issues.append(f"Multiple ages for {person}: {ages}")
    
    # Check for impossible relationships
    for r in kb.query("parent", "X", "Y"):
        if kb.query_exists("parent", r["Y"], r["X"]):
            issues.append(f"Circular parentage: {r['X']} <-> {r['Y']}")
    
    return issues

# Usage
issues = check_consistency(kb)
if issues:
    print("Consistency issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Schema Validation

```python
class SchemaValidator:
    def __init__(self):
        self.schemas = {}
    
    def define_schema(self, functor, arity, types=None):
        self.schemas[functor] = {
            "arity": arity,
            "types": types or []
        }
    
    def validate_fact(self, functor, *args):
        if functor not in self.schemas:
            return True  # No schema defined
        
        schema = self.schemas[functor]
        
        # Check arity
        if len(args) != schema["arity"]:
            return False, f"Wrong arity: expected {schema['arity']}, got {len(args)}"
        
        # Check types if defined
        if schema["types"]:
            for i, (arg, expected_type) in enumerate(zip(args, schema["types"])):
                if expected_type and not isinstance(arg, expected_type):
                    return False, f"Arg {i}: expected {expected_type}, got {type(arg)}"
        
        return True, "Valid"

# Usage
validator = SchemaValidator()
validator.define_schema("age", 2, [str, int])
validator.define_schema("enrolled", 3, [str, str, str])

# Validate before adding
is_valid, msg = validator.validate_fact("age", "alice", "twenty-five")
if not is_valid:
    print(f"Invalid fact: {msg}")
```

## Next Steps

- [Query Evaluation](queries.md) - Advanced query techniques
- [LLM Integration](llm.md) - AI-powered knowledge generation
- [Python API](../api/pythonic.md) - Programmatic knowledge management
- [Examples](../examples/academic.md) - Real-world knowledge bases