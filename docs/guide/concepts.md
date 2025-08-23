# Core Concepts

Understanding DreamLog's fundamental concepts will help you leverage its full power.

## Logic Programming Basics

### Facts

Facts are statements that are unconditionally true in your knowledge base:

```lisp
(parent john mary)      ; John is Mary's parent
(age alice 25)          ; Alice is 25 years old
(capital france paris)  ; Paris is the capital of France
```

In Python:
```python
jl.fact("parent", "john", "mary")
jl.fact("age", "alice", 25)
jl.fact("capital", "france", "paris")
```

### Rules

Rules define relationships that are conditionally true:

```lisp
; X is Y's grandparent if X is parent of Z and Z is parent of Y
(grandparent X Y) :- (parent X Z), (parent Z Y)

; X and Y are siblings if they share a parent
(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)
```

In Python:
```python
jl.rule("grandparent", ["X", "Y"]) \
  .when("parent", ["X", "Z"]) \
  .and_("parent", ["Z", "Y"]) \
  .build()
```

### Variables

Variables start with uppercase letters and can match any value:

- `X`, `Y`, `Z` - Common variables
- `Person`, `Student` - Descriptive variables
- `_` - Anonymous variable (different each use)

### Queries

Queries search for solutions that satisfy given conditions:

```lisp
(parent john X)?        ; Who are John's children?
(parent X mary)?        ; Who is Mary's parent?
(grandparent X alice)?  ; Who are Alice's grandparents?
```

## Unification

Unification is the process of making two terms identical by finding substitutions for variables.

### How It Works

```lisp
; Terms to unify:
(parent john X)
(parent john mary)

; Result: X = mary
```

### Unification Rules

1. **Identical terms unify with empty substitution**
   ```lisp
   (parent john mary) ≡ (parent john mary)  ; ✓
   ```

2. **Different constants don't unify**
   ```lisp
   john ≠ mary  ; ✗
   ```

3. **Variable unifies with any term**
   ```lisp
   X ≡ mary     ; ✓ {X = mary}
   X ≡ (f a b)  ; ✓ {X = (f a b)}
   ```

4. **Occurs check prevents infinite structures**
   ```lisp
   X ≡ (f X)    ; ✗ Would create infinite term
   ```

## Query Resolution

DreamLog uses SLD resolution (Selective Linear Definite clause resolution) with backtracking.

### Resolution Process

1. **Match query against facts**
   ```lisp
   Query: (parent john X)?
   Fact:  (parent john mary)
   Result: X = mary
   ```

2. **Match query against rule heads**
   ```lisp
   Query: (grandparent john X)?
   Rule:  (grandparent A B) :- (parent A C), (parent C B)
   
   ; Unify query with head: A=john, B=X
   ; New goals: (parent john C), (parent C X)
   ```

3. **Backtrack on failure**
   ```lisp
   ; If one path fails, try alternatives
   ; DreamLog explores all possibilities
   ```

## The LLM Hook

When DreamLog encounters an unknown term, it can invoke an LLM to generate relevant knowledge.

### How It Works

1. **Query fails to match any facts/rules**
   ```lisp
   Query: (uncle bob alice)?
   ; No uncle facts or rules exist
   ```

2. **LLM Hook triggered**
   ```python
   # DreamLog asks LLM: "Generate facts/rules for uncle(bob, alice)"
   ```

3. **LLM generates knowledge**
   ```json
   [
     ["rule", ["uncle", "X", "Y"], 
      [["sibling", "X", "Z"], ["parent", "Z", "Y"], ["male", "X"]]]
   ]
   ```

4. **Knowledge added and query retried**
   ```lisp
   ; New rule added to KB
   ; Query resolution continues
   ```

### Benefits

- **Zero-shot learning** - No predefined rules needed
- **Domain adaptation** - LLM provides domain-specific knowledge
- **Incremental learning** - KB grows as needed

## Knowledge Base

The knowledge base stores all facts and rules with efficient indexing.

### Structure

```python
class KnowledgeBase:
    facts: List[Fact]       # All facts
    rules: List[Rule]       # All rules
    fact_index: Dict        # Functor -> Facts mapping
    rule_index: Dict        # Functor -> Rules mapping
```

### Indexing

Facts and rules are indexed by functor for efficient retrieval:

```python
# Adding (parent john mary)
kb.fact_index["parent"] = [
    Fact(parent john mary),
    Fact(parent mary alice),
    ...
]
```

## Terms and Data Types

### Atoms

Constants in the system:
```lisp
john        ; Simple atom
"John Doe"  ; Quoted atom
42          ; Number atom
true        ; Boolean atom
```

### Compounds

Structured terms with functor and arguments:
```lisp
(parent john mary)           ; Binary relation
(student alice cs101 2024)   ; Ternary relation
(list 1 2 3 4 5)            ; List-like structure
```

### Lists (Simulated)

While DreamLog doesn't have native lists, you can simulate them:
```lisp
; Empty list
nil

; List [1, 2, 3]
(cons 1 (cons 2 (cons 3 nil)))

; List operations
(head (cons H T) H)  ; Get head
(tail (cons H T) T)  ; Get tail
```

## Evaluation Strategy

DreamLog uses depth-first search with backtracking:

1. **Depth-first** - Explores one solution path completely
2. **Backtracking** - Returns to choice points on failure
3. **Left-to-right** - Evaluates goals in order
4. **Lazy evaluation** - Yields solutions one at a time

Example trace:
```lisp
Query: (grandparent john X)?
1. Match rule: (grandparent A B) :- (parent A C), (parent C B)
2. Unify: A=john, B=X
3. New goals: (parent john C), (parent C X)
4. Solve (parent john C):
   - Match fact: (parent john mary), C=mary
5. Solve (parent mary X):
   - Match fact: (parent mary alice), X=alice
6. Solution: X=alice
7. Backtrack for more solutions...
```

## Best Practices

### 1. Use Descriptive Names

```lisp
; Good
(enrolled Student Course)
(teaches Professor Subject)

; Less clear
(r1 X Y)
(p A B C)
```

### 2. Order Matters in Rules

```lisp
; Efficient: Filter first
(valid_student X) :- (enrolled X _), (active X)

; Less efficient: Generate all first
(valid_student X) :- (person X), (enrolled X _)
```

### 3. Avoid Infinite Loops

```lisp
; Dangerous: No base case
(ancestor X Y) :- (ancestor X Z), (parent Z Y)

; Safe: Base case first
(ancestor X Y) :- (parent X Y)
(ancestor X Z) :- (parent X Y), (ancestor Y Z)
```

### 4. Use Cut Points Wisely

While DreamLog doesn't have Prolog's cut (!), structure rules to minimize backtracking:

```lisp
; Structure rules from specific to general
(classify X mammal) :- (has_fur X)
(classify X bird) :- (has_feathers X)
(classify X unknown) :- (animal X)
```

## Next Steps

- [S-Expression Syntax](syntax.md) - Detailed syntax guide
- [Knowledge Bases](knowledge.md) - Managing facts and rules
- [LLM Integration](llm.md) - AI-powered reasoning