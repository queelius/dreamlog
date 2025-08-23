# Query Evaluation Guide

Understanding how DreamLog evaluates queries using SLD resolution and backtracking.

## Query Basics

### Simple Queries

```python
from dreamlog.pythonic import dreamlog

kb = dreamlog()
kb.fact("parent", "john", "mary")
kb.fact("parent", "mary", "alice")

# Yes/no query
exists = kb.query_exists("parent", "john", "mary")
print(f"John is Mary's parent: {exists}")  # True

# Find bindings
for result in kb.query("parent", "john", "X"):
    print(f"John is parent of: {result['X']}")  # mary
```

### Variable Patterns

```python
# Single variable
kb.query("parent", "john", "X")     # John's children
kb.query("parent", "X", "mary")     # Mary's parents
kb.query("parent", "X", "X")        # Self-parents (none)

# Multiple variables  
kb.query("parent", "X", "Y")        # All parent-child pairs

# Anonymous variables
kb.query("parent", "_", "mary")     # Does Mary have any parent?
kb.query("parent", "john", "_")     # Does John have any children?
```

## SLD Resolution

### How Resolution Works

DreamLog uses SLD (Selective Linear Definite) resolution:

1. **Goal Selection**: Start with query goal
2. **Rule Matching**: Find rules with matching heads
3. **Unification**: Bind variables consistently
4. **Substitution**: Replace goal with rule body
5. **Recursion**: Resolve new goals
6. **Backtracking**: Try alternatives on failure

```python
kb.parse("""
(grandparent X Z) :- (parent X Y), (parent Y Z)
""")

# Query: (grandparent john X)?
# 1. Match rule: (grandparent john Z) :- (parent john Y), (parent Y Z)
# 2. Resolve: (parent john Y) → Y = mary
# 3. Resolve: (parent mary Z) → Z = alice
# 4. Solution: X = alice
```

### Resolution Trace

```python
# Enable tracing to see resolution steps
kb.set_trace(True)

for result in kb.query("grandparent", "john", "X"):
    print(f"Result: {result}")

# Output shows:
# CALL: (grandparent john X)
# CALL: (parent john Y)
# EXIT: (parent john mary) with Y=mary
# CALL: (parent mary Z)
# EXIT: (parent mary alice) with Z=alice
# EXIT: (grandparent john alice) with X=alice

kb.set_trace(False)
```

## Backtracking

### Multiple Solutions

```python
kb.facts(
    ("parent", "john", "mary"),
    ("parent", "john", "bob"),
    ("parent", "mary", "alice"),
    ("parent", "bob", "charlie")
)

# Backtracking finds all grandchildren
for result in kb.query("grandparent", "john", "X"):
    print(f"Grandchild: {result['X']}")
# Output:
# Grandchild: alice
# Grandchild: charlie
```

### Choice Points

```python
kb.parse("""
; Multiple rules create choice points
(can_fly X) :- (bird X)
(can_fly X) :- (plane X)
(can_fly X) :- (superhero X)

(bird robin)
(bird sparrow)
(plane boeing747)
(superhero superman)
""")

# Each rule is tried via backtracking
for result in kb.query("can_fly", "X"):
    print(f"Can fly: {result['X']}")
# Tries all three rules, finding all solutions
```

### Controlling Backtracking

```python
# Order rules from specific to general
kb.parse("""
; Specific cases first
(classify X mammal) :- (has_fur X), (gives_milk X)
(classify X bird) :- (has_feathers X), (lays_eggs X)
(classify X reptile) :- (has_scales X), (cold_blooded X)
(classify X unknown) :- (animal X)  ; Catch-all
""")

# First matching rule wins for each X
```

## Complex Queries

### Conjunctive Queries (AND)

```python
# Multiple goals with shared variables
kb.parse("""
(eligible_student X) :-
    (student X Major),
    (gpa X GPA),
    (greater GPA 3.0),
    (credits X Credits),
    (greater Credits 60)
""")

# All conditions must be satisfied
for result in kb.query("eligible_student", "X"):
    print(f"Eligible: {result['X']}")
```

### Disjunctive Queries (OR)

```python
# Multiple rules provide disjunction
kb.parse("""
(discount_eligible X) :- (student X)
(discount_eligible X) :- (senior X)
(discount_eligible X) :- (veteran X)
""")

# Any rule can match
eligible = set()
for result in kb.query("discount_eligible", "X"):
    eligible.add(result['X'])
```

### Nested Queries

```python
kb.parse("""
(related X Y) :- (parent X Y)
(related X Y) :- (parent Y X)
(related X Y) :- (sibling X Y)
(related X Y) :- 
    (parent X Z),
    (related Z Y),
    (different X Y)
""")

# Recursive resolution with depth
```

## Query Patterns

### Existence Checking

```python
def exists(kb, *query):
    """Check if any solution exists"""
    try:
        next(kb.query(*query))
        return True
    except StopIteration:
        return False

# Usage
if exists(kb, "student", "alice", "_"):
    print("Alice is a student")
```

### Finding All Solutions

```python
def find_all(kb, *query):
    """Collect all solutions"""
    return list(kb.query(*query))

# Get all students
all_students = find_all(kb, "student", "X", "_")
print(f"Found {len(all_students)} students")

# Extract specific variable
names = [r['X'] for r in all_students]
```

### First Solution

```python
def find_first(kb, *query):
    """Get first solution or None"""
    try:
        return next(kb.query(*query))
    except StopIteration:
        return None

# Get any parent of mary
parent = find_first(kb, "parent", "X", "mary")
if parent:
    print(f"Mary's parent: {parent['X']}")
```

### Counting Solutions

```python
def count_solutions(kb, *query):
    """Count solutions without materializing all"""
    count = 0
    for _ in kb.query(*query):
        count += 1
    return count

# Count students
num_students = count_solutions(kb, "student", "_", "_")
print(f"Total students: {num_students}")
```

## Advanced Query Techniques

### Aggregation

```python
# Aggregate using Python
def sum_credits(kb, student):
    total = 0
    for r in kb.query("completed_course", student, "Course"):
        for c in kb.query("course_credits", r['Course'], "Credits"):
            total += c['Credits']
    return total

# Or define in rules
kb.parse("""
(total_credits Student Total) :-
    (findall Credits 
        (and (completed Student Course)
             (credits Course Credits))
        CreditsList),
    (sum_list CreditsList Total)
""")
```

### Negation as Failure

```python
kb.parse("""
(not_enrolled Student) :-
    (student Student _),
    (not (enrolled Student _))

(available_course Course) :-
    (course Course _ _),
    (not (full Course))
""")

# Find students not enrolled
for r in kb.query("not_enrolled", "X"):
    print(f"Not enrolled: {r['X']}")
```

### Guards and Constraints

```python
kb.parse("""
(valid_enrollment Student Course) :-
    (student Student Major),
    (course Course Dept _),
    (or (equals Major Dept)
        (elective Course)),
    (not (completed Student Course)),
    (has_prerequisites Student Course)
""")

# Complex constraints in queries
```

### Meta-Queries

```python
# Query about the knowledge base itself
kb.parse("""
(defined_predicate Functor) :-
    (or (fact Functor _ _)
        (rule Functor _ _))

(rule_count Functor Count) :-
    (findall 1 (rule Functor _ _) Ones),
    (length Ones Count)
""")

# Introspection
for r in kb.query("defined_predicate", "X"):
    print(f"Defined: {r['X']}")
```

## Query Optimization

### Goal Ordering

```python
# Inefficient: Generate all, then filter
kb.parse("""
(slow_query X Y) :-
    (person X),           ; Generate all people
    (person Y),           ; Generate all people
    (age X AgeX),
    (age Y AgeY),
    (greater AgeX 50),    ; Filter late
    (less AgeY 30)
""")

# Efficient: Filter early
kb.parse("""
(fast_query X Y) :-
    (age X AgeX),
    (greater AgeX 50),    ; Filter early
    (age Y AgeY),
    (less AgeY 30),       ; Filter early
    (person X),
    (person Y)
""")
```

### Indexing

```python
# DreamLog automatically indexes by functor
# Additional optimization strategies:

class IndexedQuery:
    def __init__(self, kb):
        self.kb = kb
        self.indexes = {}
    
    def build_index(self, functor, arg_position):
        """Build index on specific argument"""
        index = {}
        for result in self.kb.query(functor, *(['_'] * 3)):
            key = result.get(f'arg{arg_position}')
            if key:
                if key not in index:
                    index[key] = []
                index[key].append(result)
        self.indexes[(functor, arg_position)] = index
    
    def query_indexed(self, functor, arg_position, value):
        """Use index for fast lookup"""
        key = (functor, arg_position)
        if key in self.indexes:
            return self.indexes[key].get(value, [])
        return list(self.kb.query(functor, *(['_'] * 3)))
```

### Memoization

```python
from functools import lru_cache

class MemoizedKB:
    def __init__(self, kb):
        self.kb = kb
    
    @lru_cache(maxsize=128)
    def query_cached(self, query_str):
        """Cache query results"""
        # Parse query string
        parts = query_str.split()
        return tuple(self.kb.query(*parts))
    
    def invalidate_cache(self):
        """Clear cache when KB changes"""
        self.query_cached.cache_clear()
```

## Query Debugging

### Trace Mode

```python
# Detailed tracing
kb.set_trace(True, verbose=True)

for r in kb.query("complex_rule", "X"):
    print(f"Solution: {r}")

kb.set_trace(False)
```

### Step-by-Step Execution

```python
class DebugQuery:
    def __init__(self, kb):
        self.kb = kb
        self.steps = []
    
    def query_debug(self, *args):
        """Track each resolution step"""
        self.steps = []
        
        # Hook into resolution
        original_resolve = self.kb._resolve
        
        def tracked_resolve(*args, **kwargs):
            self.steps.append({
                'goal': args[0] if args else None,
                'depth': len(self.steps)
            })
            return original_resolve(*args, **kwargs)
        
        self.kb._resolve = tracked_resolve
        results = list(self.kb.query(*args))
        self.kb._resolve = original_resolve
        
        return results, self.steps
    
    def print_steps(self):
        """Display resolution steps"""
        for i, step in enumerate(self.steps):
            indent = "  " * step['depth']
            print(f"{i}: {indent}{step['goal']}")
```

### Performance Profiling

```python
import time
from collections import defaultdict

class QueryProfiler:
    def __init__(self, kb):
        self.kb = kb
        self.timings = defaultdict(list)
    
    def profile_query(self, *args):
        """Profile query execution"""
        start = time.perf_counter()
        results = list(self.kb.query(*args))
        elapsed = time.perf_counter() - start
        
        query_key = f"{args[0]}/{len(args)-1}"
        self.timings[query_key].append(elapsed)
        
        return results, elapsed
    
    def report(self):
        """Generate performance report"""
        print("Query Performance Report:")
        print("-" * 40)
        for query, times in self.timings.items():
            avg_time = sum(times) / len(times)
            print(f"{query}: {avg_time:.4f}s (n={len(times)})")
```

## Error Handling

### Query Errors

```python
from dreamlog.exceptions import QueryError, UnificationError

def safe_query(kb, *args):
    """Query with error handling"""
    try:
        results = list(kb.query(*args))
        return {'success': True, 'results': results}
    except UnificationError as e:
        return {'success': False, 'error': f"Unification failed: {e}"}
    except QueryError as e:
        return {'success': False, 'error': f"Query error: {e}"}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected: {e}"}

# Usage
result = safe_query(kb, "invalid_query", "X")
if not result['success']:
    print(f"Query failed: {result['error']}")
```

### Timeout Handling

```python
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    """Timeout context for queries"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Query exceeded {seconds}s")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Usage
try:
    with timeout(5):
        results = list(kb.query("potentially_slow_query", "X"))
except TimeoutError:
    print("Query timed out")
```

## Integration with Python

### Generator Pattern

```python
# Queries return generators for efficiency
def process_results(kb, *query):
    """Process results one at a time"""
    for result in kb.query(*query):
        # Process immediately without storing all
        process_single_result(result)
        
        # Can break early if needed
        if should_stop(result):
            break

def process_single_result(result):
    """Handle individual result"""
    print(f"Processing: {result}")

def should_stop(result):
    """Determine if we should stop early"""
    return result.get('score', 0) > 100
```

### Async Queries

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncKB:
    def __init__(self, kb):
        self.kb = kb
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def query_async(self, *args):
        """Async query execution"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            lambda: list(self.kb.query(*args))
        )
        return results
    
    async def parallel_queries(self, queries):
        """Run multiple queries in parallel"""
        tasks = [self.query_async(*q) for q in queries]
        return await asyncio.gather(*tasks)

# Usage
async def main():
    akb = AsyncKB(kb)
    results = await akb.parallel_queries([
        ("student", "X", "cs"),
        ("professor", "Y", "math"),
        ("course", "Z", "_")
    ])
    return results
```

## Next Steps

- [LLM Integration](llm.md) - AI-powered query enhancement
- [Python API](../api/pythonic.md) - Full API reference
- [Examples](../examples/queries.md) - Query examples
- [Performance](../guide/performance.md) - Optimization guide