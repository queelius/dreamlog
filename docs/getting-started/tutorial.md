# Tutorial: Building a Smart Knowledge System

Welcome to the DreamLog tutorial! We'll build a complete knowledge system step by step, learning all the core concepts along the way.

## What We'll Build

We'll create an academic advising system that can:
- Track student enrollments and grades
- Determine graduation eligibility
- Recommend courses based on prerequisites
- Use LLMs to generate missing knowledge

## Part 1: Basic Facts and Queries

### Setting Up

```python
from dreamlog.pythonic import dreamlog

# Create a new DreamLog instance
advisor = dreamlog()
```

### Adding Facts

Facts are basic truths in our knowledge base:

```python
# Student enrollments
advisor.fact("student", "alice", "cs")
advisor.fact("student", "bob", "math")
advisor.fact("student", "charlie", "physics")

# Course offerings
advisor.fact("course", "cs101", "intro_programming")
advisor.fact("course", "cs201", "data_structures")
advisor.fact("course", "math101", "calculus_1")

# Enrollments
advisor.fact("enrolled", "alice", "cs101")
advisor.fact("enrolled", "alice", "math101")
advisor.fact("enrolled", "bob", "math101")

print(f"Added {advisor.stats['num_facts']} facts")
```

### Basic Queries

Query the knowledge base:

```python
# Who is enrolled in math101?
print("Students in math101:")
for result in advisor.query("enrolled", "X", "math101"):
    print(f"  - {result['X']}")

# What courses is Alice taking?
print("\nAlice's courses:")
for result in advisor.query("enrolled", "alice", "X"):
    print(f"  - {result['X']}")
```

## Part 2: Rules and Relationships

### Defining Rules

Rules express conditional relationships:

```python
# Prerequisites rule
advisor.rule("can_take", ["Student", "Course"]) \
    .when("completed", ["Student", "Prereq"]) \
    .and_("prerequisite", ["Course", "Prereq"]) \
    .build()

# Classmates rule
advisor.rule("classmates", ["X", "Y"]) \
    .when("enrolled", ["X", "Course"]) \
    .and_("enrolled", ["Y", "Course"]) \
    .and_("different", ["X", "Y"]) \
    .build()

# Add prerequisites
advisor.fact("prerequisite", "cs201", "cs101")
advisor.fact("prerequisite", "cs301", "cs201")

# Add completions
advisor.fact("completed", "alice", "cs101")
```

### Using S-Expression Syntax

You can also use S-expressions directly:

```lisp
# Parse S-expression rules
advisor.parse('''
(honors_student X) :- 
    (student X Major),
    (gpa X GPA),
    (greater GPA 3.5)
''')

# Add GPA facts
advisor.parse("(gpa alice 3.8)")
advisor.parse("(gpa bob 3.2)")
advisor.parse("(gpa charlie 3.9)")
```

### Query with Rules

```python
# Who can take cs201?
for result in advisor.query("can_take", "X", "cs201"):
    print(f"{result['X']} can take cs201")

# Find honors students
for result in advisor.query("honors_student", "X"):
    print(f"{result['X']} is an honors student")
```

## Part 3: Complex Logic

### Recursive Rules

```python
# Course dependency chains
advisor.parse('''
(depends_on Direct Indirect) :- (prerequisite Direct Indirect)
(depends_on Course Indirect) :- 
    (prerequisite Course Direct),
    (depends_on Direct Indirect)
''')

# Query full dependency chain
for result in advisor.query("depends_on", "cs301", "X"):
    print(f"cs301 depends on {result['X']}")
```

### Negation as Failure

```python
# Students not enrolled in any course
advisor.parse('''
(not_enrolled Student) :-
    (student Student Major),
    (not (enrolled Student _))
''')

# Add a student with no enrollments
advisor.fact("student", "diana", "cs")

for result in advisor.query("not_enrolled", "X"):
    print(f"{result['X']} is not enrolled in any courses")
```

## Part 4: LLM Integration

### Enable LLM Support

```python
# Create a new instance with LLM
smart_advisor = dreamlog(llm_provider="openai")  # or "mock" for testing

# Copy existing knowledge
smart_advisor.load_from(advisor)
```

### Automatic Knowledge Generation

```python
# Query for undefined concept
# The LLM will generate rules for "good_standing"
for result in smart_advisor.query("good_standing", "alice"):
    print(f"Alice is in good standing: {result}")

# Check what was generated
print("\nGenerated rules:")
for rule in smart_advisor.get_rules("good_standing"):
    print(f"  {rule}")
```

### Domain-Specific Knowledge

```python
# Set knowledge domain for better LLM responses
smart_advisor.set_llm_domain("academic_advising")

# Query for complex undefined relationships
for result in smart_advisor.query("graduation_ready", "alice"):
    print(f"Alice graduation status: {result}")

# The LLM generates appropriate rules based on domain
```

## Part 5: Working with Data

### Batch Operations

```python
# Load many facts at once
students = [
    ("student", "eve", "cs"),
    ("student", "frank", "math"),
    ("student", "grace", "physics")
]

courses = [
    ("course", "cs102", "intro_algorithms"),
    ("course", "math201", "calculus_2"),
    ("course", "phys101", "mechanics")
]

advisor.facts(*students)
advisor.facts(*courses)
```

### Save and Load

```python
# Save knowledge base
advisor.save("academic_advisor.dreamlog")

# Load into new instance
new_advisor = dreamlog()
new_advisor.load("academic_advisor.dreamlog")

print(f"Loaded {new_advisor.stats['total_items']} items")
```

### Export Formats

```python
# Export as S-expressions
sexpr = advisor.to_sexp()
print(sexpr[:500])  # First 500 chars

# Export as JSON (prefix notation)
import json
json_kb = advisor.to_json()
print(json.dumps(json_kb["facts"][:3], indent=2))
```

## Part 6: Advanced Patterns

### Guards and Constraints

```python
# Rules with numeric constraints
advisor.parse('''
(can_graduate Student) :-
    (student Student Major),
    (total_credits Student Credits),
    (required_credits Major Required),
    (greater_equal Credits Required),
    (gpa Student GPA),
    (greater_equal GPA 2.0)
''')

# Add credit facts
advisor.fact("total_credits", "alice", 120)
advisor.fact("required_credits", "cs", 120)
```

### Meta-predicates

```python
# Query about the knowledge base itself
advisor.parse('''
(has_rule Functor) :- (rule Functor _ _)
(has_fact Functor) :- (fact Functor _ _)
''')

# Find all defined predicates
for result in advisor.query("has_rule", "X"):
    print(f"Rule defined: {result['X']}")
```

### Custom Evaluators

```python
# Add Python function as built-in
def current_semester():
    return "spring2024"

advisor.add_builtin("current_semester", current_semester)

# Use in rules
advisor.parse('''
(active_enrollment Student Course) :-
    (enrolled Student Course Semester),
    (current_semester Semester)
''')
```

## Part 7: Interactive Development

### REPL Usage

```bash
# Start REPL
python -m dreamlog.repl

# In REPL:
dreamlog> (student alice cs)
dreamlog> (enrolled alice cs101)
dreamlog> (enrolled alice X)?
  X = cs101
dreamlog> :save session.dreamlog
dreamlog> :help
```

### Debugging Queries

```python
# Enable trace mode
advisor.set_trace(True)

# Query with tracing
for result in advisor.query("can_graduate", "alice"):
    print(result)
# Shows step-by-step resolution

advisor.set_trace(False)
```

### Performance Analysis

```python
import time

# Time complex query
start = time.time()
results = list(advisor.query("depends_on", "X", "Y"))
elapsed = time.time() - start

print(f"Found {len(results)} dependencies in {elapsed:.3f}s")
print(f"Knowledge base size: {advisor.stats}")
```

## Part 8: Integration Examples

### With Pandas DataFrames

```python
import pandas as pd

# Load student data
students_df = pd.DataFrame({
    'name': ['alice', 'bob', 'charlie'],
    'major': ['cs', 'math', 'physics'],
    'gpa': [3.8, 3.2, 3.9]
})

# Add to knowledge base
for _, row in students_df.iterrows():
    advisor.fact("student", row['name'], row['major'])
    advisor.fact("gpa", row['name'], row['gpa'])

# Query and convert back to DataFrame
results = []
for r in advisor.query("honors_student", "X"):
    results.append({'student': r['X']})
    
honors_df = pd.DataFrame(results)
print(honors_df)
```

### With REST API

```python
# Start API server (in terminal)
# python integrations/api/dreamlog_api_server.py

import requests

# Add facts via API
requests.post('http://localhost:8000/facts', 
    json={'fact': '(student diana bio)'})

# Query via API
response = requests.post('http://localhost:8000/query',
    json={'query': '(student X Major)'})
    
print(response.json()['results'])
```

### In Jupyter Notebooks

```python
# In Jupyter cell
%load_ext dreamlog.jupyter.dreamlog_magic
%dreamlog_init

# Now use magic commands
%dreamlog_fact (student eve cs)
%dreamlog_rule (mentor X Y) :- (professor X), (student Y _)
%dreamlog_query (student X cs)
```

## Part 9: Best Practices

### 1. Structure Your Knowledge

```python
# Organize facts by category
class AcademicKB:
    def __init__(self):
        self.jl = dreamlog()
        self._init_schema()
    
    def _init_schema(self):
        # Core entities
        self.jl.parse('''
        (entity student)
        (entity course)
        (entity professor)
        ''')
        
    def add_student(self, name, major, year):
        self.jl.fact("student", name, major)
        self.jl.fact("year", name, year)
        return self
        
    def add_enrollment(self, student, course, semester):
        self.jl.fact("enrolled", student, course, semester)
        return self
```

### 2. Error Handling

```python
from dreamlog.exceptions import UnificationError, QueryError

try:
    # Attempt query
    results = list(advisor.query("invalid_predicate", "X"))
except QueryError as e:
    print(f"Query failed: {e}")
    # Handle gracefully
```

### 3. Testing Logic

```python
def test_graduation_rules():
    test_kb = dreamlog()
    
    # Setup test data
    test_kb.fact("student", "test_student", "cs")
    test_kb.fact("total_credits", "test_student", 120)
    test_kb.fact("gpa", "test_student", 3.5)
    
    # Add rule under test
    test_kb.parse('''
    (can_graduate X) :- 
        (student X _),
        (total_credits X C),
        (greater_equal C 120),
        (gpa X G),
        (greater_equal G 2.0)
    ''')
    
    # Assert expected results
    results = list(test_kb.query("can_graduate", "test_student"))
    assert len(results) == 1
    print("✓ Graduation rule test passed")

test_graduation_rules()
```

## Part 10: Complete Example

Let's put it all together:

```python
from dreamlog.pythonic import dreamlog

class UniversityAdvisor:
    """Complete academic advising system"""
    
    def __init__(self, use_llm=False):
        provider = "openai" if use_llm else None
        self.kb = dreamlog(llm_provider=provider)
        self._init_rules()
    
    def _init_rules(self):
        """Initialize core rules"""
        rules = '''
        ; Graduation eligibility
        (can_graduate Student) :-
            (student Student Major),
            (completed_credits Student Credits),
            (required_credits Major Required),
            (greater_equal Credits Required),
            (gpa Student GPA),
            (greater_equal GPA 2.0),
            (completed_requirements Student Major)
        
        ; Dean's list
        (deans_list Student Semester) :-
            (semester_gpa Student Semester GPA),
            (greater_equal GPA 3.5),
            (full_time Student Semester)
        
        ; Academic warning
        (academic_warning Student) :-
            (gpa Student GPA),
            (less GPA 2.0)
        
        ; Advisor assignment
        (advisor_for Student Professor) :-
            (student Student Major),
            (professor Professor Major),
            (advises Professor Student)
        '''
        self.kb.parse(rules)
    
    def add_student(self, name, major, year, gpa):
        self.kb.fact("student", name, major)
        self.kb.fact("year", name, year)
        self.kb.fact("gpa", name, gpa)
        return self
    
    def add_course(self, code, name, credits, professor=None):
        self.kb.fact("course", code, name)
        self.kb.fact("credits", code, credits)
        if professor:
            self.kb.fact("teaches", professor, code)
        return self
    
    def enroll(self, student, course, semester="current"):
        self.kb.fact("enrolled", student, course, semester)
        return self
    
    def complete_course(self, student, course, grade):
        self.kb.fact("completed", student, course)
        self.kb.fact("grade", student, course, grade)
        return self
    
    def check_graduation(self, student):
        """Check if student can graduate"""
        results = list(self.kb.query("can_graduate", student))
        return len(results) > 0
    
    def get_advisees(self, professor):
        """Get all students advised by professor"""
        return [r['Student'] 
                for r in self.kb.query("advisor_for", "Student", professor)]
    
    def recommend_courses(self, student):
        """Recommend courses for student (uses LLM if enabled)"""
        # This will trigger LLM if "recommend" isn't defined
        return list(self.kb.query("recommend", student, "Course"))
    
    def report(self, student):
        """Generate student report"""
        report = f"=== Report for {student} ===\n"
        
        # Basic info
        for r in self.kb.query("student", student, "Major"):
            report += f"Major: {r['Major']}\n"
        
        for r in self.kb.query("gpa", student, "GPA"):
            report += f"GPA: {r['GPA']}\n"
        
        # Enrollment
        report += "\nCurrent Enrollments:\n"
        for r in self.kb.query("enrolled", student, "Course", "current"):
            report += f"  - {r['Course']}\n"
        
        # Status checks
        if self.check_graduation(student):
            report += "\n✓ Eligible for graduation\n"
        
        if list(self.kb.query("deans_list", student, "current")):
            report += "✓ On Dean's List\n"
        
        if list(self.kb.query("academic_warning", student)):
            report += "⚠ Academic Warning\n"
        
        return report

# Usage example
uni = UniversityAdvisor(use_llm=False)

# Add data
uni.add_student("alice", "cs", 4, 3.8) \
   .add_student("bob", "math", 3, 2.1) \
   .add_student("charlie", "physics", 2, 3.9)

uni.add_course("cs101", "Intro to Programming", 3, "dr_smith") \
   .add_course("cs201", "Data Structures", 3, "dr_jones") \
   .add_course("math101", "Calculus I", 4, "dr_brown")

uni.enroll("alice", "cs201") \
   .enroll("bob", "math101") \
   .enroll("charlie", "cs101")

# Generate reports
print(uni.report("alice"))
print(uni.report("bob"))

# Check specific queries
if uni.check_graduation("alice"):
    print("Alice can graduate!")

# Save state
uni.kb.save("university.dreamlog")
print(f"\nSaved {uni.kb.stats['total_items']} items to university.dreamlog")
```

## Summary

You've learned how to:
- Create facts and rules in DreamLog
- Query knowledge bases with variables
- Use S-expression syntax
- Integrate LLMs for automatic knowledge generation
- Build complex logic systems
- Integrate with Python applications
- Follow best practices for maintainable code

## Next Steps

- Explore [S-Expression Syntax](../guide/syntax.md) in depth
- Learn about [LLM Integration](../guide/llm.md) options
- Check out [API Reference](../api/pythonic.md) for all methods
- See [Examples](../examples/family.md) for more use cases

Happy logic programming with DreamLog!