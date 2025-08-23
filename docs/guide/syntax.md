# S-Expression Syntax Guide

DreamLog uses S-expressions (symbolic expressions) as its primary syntax. This clean, uniform syntax makes parsing simple and reasoning transparent.

## Basic Syntax

### Atoms

Atoms are the basic building blocks:

```lisp
; Simple atoms (alphanumeric, underscores)
alice
bob
cs101
course_2024

; Numbers
42
3.14
-17

; Quoted strings (for spaces/special chars)
"John Doe"
"New York"
"user@example.com"

; Special atoms
nil      ; Empty/null value
true     ; Boolean true
false    ; Boolean false
```

### Compound Terms

Compound terms have a functor and arguments:

```lisp
; Basic structure: (functor arg1 arg2 ...)
(parent john mary)
(age alice 25)
(enrolled student cs101 spring2024)

; Nested compounds
(location (city paris) (country france))
(teaches (professor smith) (course cs101))
```

### Variables

Variables start with uppercase letters or underscore:

```lisp
X         ; Simple variable
Person    ; Descriptive variable
_         ; Anonymous variable (different each use)
_result   ; Named underscore variable
X1        ; Variable with number
```

## Facts

Facts are assertions that are always true:

```lisp
; Simple facts
(parent john mary)
(age bob 30)
(capital france paris)

; Multi-argument facts
(grade alice cs101 95)
(flight aa100 boston chicago 1430)

; Facts with nested structure
(owns john (car toyota camry 2022))
```

### Fact Patterns

Common patterns for organizing facts:

```lisp
; Entity-Attribute-Value
(age alice 25)
(height bob 180)
(color car1 red)

; Relationships
(parent john mary)
(friend alice bob)
(manages carol david)

; Classifications
(type sparrow bird)
(instance fido dog)
(category cs101 required)

; Events
(enrolled alice cs101 fall2023)
(purchased john item42 2024-01-15)
```

## Rules

Rules define conditional relationships:

### Basic Rule Structure

```lisp
; Format: (head) :- (body)
(grandparent X Z) :- (parent X Y), (parent Y Z)

; Read as: "X is grandparent of Z IF X is parent of Y AND Y is parent of Z"
```

### Multiple Conditions

```lisp
; Use comma for AND
(sibling X Y) :- 
    (parent Z X), 
    (parent Z Y), 
    (different X Y)

; Multiple rules for same head (implicit OR)
(can_fly X) :- (bird X)
(can_fly X) :- (airplane X)
(can_fly X) :- (has_wings X), (lightweight X)
```

### Complex Rules

```lisp
; Nested conditions
(eligible_for_honors Student) :-
    (student Student Major),
    (gpa Student GPA),
    (greater GPA 3.5),
    (completed_credits Student Credits),
    (greater Credits 60)

; Recursive rules
(ancestor X Y) :- (parent X Y)
(ancestor X Z) :- 
    (parent X Y), 
    (ancestor Y Z)

; Rules with computations
(can_graduate Student) :-
    (total_credits Student Total),
    (required_credits Major Required),
    (student Student Major),
    (greater_equal Total Required)
```

## Queries

Queries search for solutions:

### Query Syntax

```lisp
; Add ? at the end for queries
(parent john mary)?      ; Yes/no query
(parent john X)?         ; Find X where john is parent of X
(parent X Y)?           ; Find all parent relationships

; Complex queries
(grandparent john X)?   ; Who are John's grandchildren?
(ancestor X alice)?     ; Who are Alice's ancestors?
```

### Query Variables

```lisp
; Single variable
(age alice X)?          ; What is Alice's age?

; Multiple variables
(parent X Y)?           ; Find all parent-child pairs

; Mixed ground/variable
(grade alice Course Grade)?  ; Alice's grades in all courses

; Anonymous variables
(parent _ mary)?        ; Does Mary have any parent?
(enrolled alice _ _)?   ; Is Alice enrolled in anything?
```

## Lists and Data Structures

While DreamLog doesn't have native lists, you can simulate them:

### List Representation

```lisp
; Empty list
nil

; List construction: (cons head tail)
(cons 1 nil)                    ; [1]
(cons 1 (cons 2 nil))           ; [1, 2]
(cons 1 (cons 2 (cons 3 nil))) ; [1, 2, 3]

; List operations as rules
(head (cons H _) H)            ; Get first element
(tail (cons _ T) T)            ; Get rest of list

(member X (cons X _))           ; X is member of list
(member X (cons _ T)) :- (member X T)

(append nil L L)
(append (cons H T1) L2 (cons H T3)) :- (append T1 L2 T3)
```

### Tree Structures

```lisp
; Binary tree: (node value left right)
(node 5 
    (node 3 nil nil)
    (node 7 nil nil))

; Tree traversal
(in_tree X (node X _ _))
(in_tree X (node _ L _)) :- (in_tree X L)
(in_tree X (node _ _ R)) :- (in_tree X R)
```

### Records/Structs

```lisp
; Person record: (person name age occupation)
(person "John Doe" 30 engineer)

; Nested structures
(employee 
    (person "Alice Smith" 25 developer)
    (department engineering)
    (salary 75000))

; Access patterns
(person_name (person Name _ _) Name)
(person_age (person _ Age _) Age)
```

## Special Constructs

### Negation as Failure

```lisp
; Using 'not' for negation
(bachelor X) :- 
    (male X), 
    (not (married X))

; Safe negation (X must be bound)
(unemployed X) :- 
    (person X),
    (not (has_job X))
```

### Built-in Predicates

```lisp
; Comparison
(greater X Y)         ; X > Y
(less X Y)           ; X < Y
(greater_equal X Y)  ; X >= Y
(less_equal X Y)     ; X <= Y
(equal X Y)          ; X = Y
(different X Y)      ; X ≠ Y

; Arithmetic (in guards)
(sum X Y Z)          ; X + Y = Z
(difference X Y Z)   ; X - Y = Z
(product X Y Z)      ; X * Y = Z
(quotient X Y Z)     ; X / Y = Z

; Type checking
(number X)
(atom X)
(compound X)
(var X)
```

### Cut and Control

While DreamLog doesn't have Prolog's cut (!), you can structure rules for similar effect:

```lisp
; Order rules from specific to general
(classify X reptile) :- (has_scales X), (cold_blooded X)
(classify X mammal) :- (has_fur X), (warm_blooded X)
(classify X bird) :- (has_feathers X), (warm_blooded X)
(classify X unknown) :- (animal X)  ; Catch-all

; Use guards to prevent backtracking
(max X Y X) :- (greater_equal X Y)
(max X Y Y) :- (less X Y)
```

## Comments and Documentation

```lisp
; Single-line comment
(parent john mary)  ; John is Mary's parent

; Multi-line comment style
; This rule determines if someone can graduate
; based on credits and GPA requirements
(can_graduate Student) :-
    (credits Student C),
    (greater_equal C 120),
    (gpa Student G),
    (greater_equal G 2.0)

; Documentation facts
(doc can_graduate "Determines graduation eligibility")
(param can_graduate 1 "Student ID or name")
```

## JSON Array Format

DreamLog also accepts JSON array notation:

```json
["parent", "john", "mary"]
["age", "alice", 25]

["rule", ["grandparent", "X", "Z"],
  [["parent", "X", "Y"], ["parent", "Y", "Z"]]]
```

This is equivalent to S-expressions and useful for programmatic generation.

## Best Practices

### 1. Naming Conventions

```lisp
; Use descriptive names
(enrolled_in Student Course)     ; Clear relationship
(has_prerequisite Course Prereq) ; Self-documenting

; Avoid abbreviations
; Bad: (enr S C)
; Good: (enrolled Student Course)
```

### 2. Rule Organization

```lisp
; Group related rules together
; === Student Rules ===
(full_time Student) :- ...
(part_time Student) :- ...
(enrolled Student Course) :- ...

; === Course Rules ===
(prerequisite Course Required) :- ...
(corequisite Course Required) :- ...
```

### 3. Variable Naming

```lisp
; Use meaningful variable names
(teaches Professor Course) :-
    (faculty Professor Department),
    (course Course Department)

; Not: (teaches P C) :- (faculty P D), (course C D)
```

### 4. Fact Organization

```lisp
; Group facts by type
; === People ===
(person alice)
(person bob)

; === Relationships ===
(parent john mary)
(parent mary alice)

; === Attributes ===
(age alice 25)
(age bob 30)
```

### 5. Query Patterns

```lisp
; Existential queries (checking existence)
(enrolled alice _)?

; Universal queries (finding all)
(enrolled X cs101)?

; Joined queries (complex relationships)
(parent X Y), (parent Y Z)?  ; Grandparent relationships
```

## Common Patterns

### Transitive Relations

```lisp
; Direct and transitive closure
(connected X Y) :- (direct_link X Y)
(connected X Z) :- (direct_link X Y), (connected Y Z)
```

### Symmetric Relations

```lisp
(friend X Y) :- (friend_of X Y)
(friend X Y) :- (friend_of Y X)
```

### Equivalence Classes

```lisp
(same_group X Y) :- (in_group X G), (in_group Y G)
```

### Aggregation

```lisp
; Count (using list accumulation)
(count_children Parent Count) :-
    (findall Child (parent Parent Child) Children),
    (length Children Count)
```

## Advanced Features

### Meta-Programming

```lisp
; Rules about rules
(has_rule Functor Arity) :- 
    (rule Head Body),
    (functor Head Functor Arity)

; Dynamic rule generation
(make_symmetric Pred) :-
    (assert (Pred X Y)),
    (assert (Pred Y X))
```

### Constraints

```lisp
; Constraint satisfaction
(valid_schedule Student) :-
    (enrolled Student Courses),
    (no_conflicts Courses),
    (within_credit_limit Courses),
    (prerequisites_met Student Courses)
```

## Next Steps

- [Knowledge Bases](knowledge.md) - Managing facts and rules
- [Queries](queries.md) - Advanced query techniques
- [Python API](../api/pythonic.md) - Using S-expressions from Python
- [Examples](../examples/family.md) - Real-world usage patterns
