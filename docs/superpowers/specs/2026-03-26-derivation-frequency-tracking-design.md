# Derivation Frequency Tracking

**Date**: 2026-03-26
**Status**: Design
**Scope**: Track clause usage during wake phase, add dead clause pruning, frequency-weighted compression prioritization

## Context

The sleep cycle (Operations A-E) analyzes the KB statically. It does not know
which clauses are actually used during query resolution. A rule might exist in
the KB but never fire. A fact might be derivable from a rule but still be queried
directly 100 times. This dynamic information should inform compression decisions.

This is the first step toward DreamCoder-faithful derivation tracking. DreamCoder
uses solution traces to decide which sub-expressions to extract as library
primitives. We start with the simplest useful dynamic signal: per-clause usage
frequency.

## Part 1: Usage Counters on KnowledgeBase

### New fields and methods on `KnowledgeBase`

```python
# Internal storage
self._usage_counts: Dict[int, int] = {}  # hash(clause) -> count

# Methods
def record_usage(self, clause: Union[Fact, Rule]) -> None:
    """Increment usage counter for a clause."""
    key = hash(clause)
    self._usage_counts[key] = self._usage_counts.get(key, 0) + 1

def get_usage(self, clause: Union[Fact, Rule]) -> int:
    """Get usage count for a clause (0 if never used)."""
    return self._usage_counts.get(hash(clause), 0)

def reset_usage(self) -> None:
    """Clear all usage counters."""
    self._usage_counts.clear()

def total_queries_tracked(self) -> int:
    """Total usage events recorded (sum of all counters)."""
    return sum(self._usage_counts.values())
```

Usage counts are preserved across `copy()` and `restore_from()`. They are NOT
cleared when facts/rules are added or removed (stale entries are harmless;
`get_usage` returns 0 for clauses not in the dict).

### Hash-based keying

Clauses are keyed by `hash(clause)`. Since `Fact` and `Rule` are frozen
dataclasses, their hashes are stable and equality-consistent. Hash collisions
are theoretically possible but practically negligible for KB sizes we handle.

## Part 2: Recording Usage in the Evaluator

### Where to increment

In `PrologEvaluator._solve_goals`:

**Fact match (line ~145-157):** After a fact unifies with the current goal and
before recursing into remaining goals:
```python
for fact in self.kb.get_matching_facts(current_goal.term):
    ...
    bindings = unify(current_goal.term, renamed_fact.term, current_goal.bindings)
    if bindings is not None:
        self.kb.record_usage(fact)  # <-- NEW
        solutions_found = True
        ...
```

**Rule match (line ~160-185):** After a rule head unifies and before recursing
into the body:
```python
for rule in self.kb.get_matching_rules(current_goal.term):
    ...
    bindings = unify(current_goal.term, renamed_rule.head, current_goal.bindings)
    if bindings is not None:
        self.kb.record_usage(rule)  # <-- NEW
        solutions_found = True
        ...
```

### Counting semantics

We count every successful match attempt, including those that eventually
backtrack. This is "attempt frequency" rather than "solution frequency." A rule
that matches frequently but often backtracks is still heavily used by the
resolution process. This is simpler than tracking only solution-contributing
matches (which would require propagating success signals back up the tree).

### Performance

One dict lookup and increment per successful match. Negligible overhead relative
to unification.

## Part 3: Operation F (Dead Clause Pruning)

### Purpose

Remove facts and rules that have never been used across a sufficient number of
queries. Dead clauses consume space and noise up the sleep cycle's analysis
without contributing to any derivation.

### Algorithm

1. Check precondition: `kb.total_queries_tracked() >= min_query_threshold`
   (default 10). If not met, skip (not enough data to judge).
2. For each fact in the KB: if `kb.get_usage(fact) == 0`, mark as dead.
3. For each rule in the KB: if `kb.get_usage(rule) == 0`, mark as dead.
4. Skip generated predicates (`_invented_`, `_extracted_`, `exception_`).
5. Verify: ensure removing dead clauses doesn't break anything (same
   verification suite as other operations).
6. Apply: remove verified dead clauses.

### MDL

Dead clause pruning always reduces clause count (pure removal). MDL delta =
-|dead_clauses|.

### Operation ordering

F runs after E (body pattern extraction) and before final verification. The
full order is: A -> B -> C -> D -> E -> F.

Rationale: Operations A-E may create new clauses (`_invented_`, `_extracted_`,
`exception_`) that have 0 usage (they didn't exist during the wake phase). F
must skip these (they're new, not dead). The `_invented_`/`_extracted_`/
`exception_` prefix filter handles this.

### Interaction with `copy()` and `restore_from()`

When `dream()` takes a KB snapshot, the snapshot includes usage counts. If the
dream is rolled back, usage counts are restored to the pre-dream state.

## Part 4: Frequency-Weighted Prioritization

### How frequencies inform compression

When Operations C, D, or E find multiple compression candidates, they currently
pick the one with the best MDL savings. With frequency data, we add a tiebreaker:

```python
score = mdl_savings * (1 + log2(total_frequency_of_affected_clauses + 1))
```

Higher-frequency patterns get a logarithmic bonus. The `+1` prevents log(0).
The log dampens the effect so a 1000x-used pattern isn't 1000x more likely to
be compressed.

This is a scoring change within existing operations, not a new operation. Each
operation that evaluates candidates (C's subgroup loop, D's skeleton group loop,
E's best-pattern selection) incorporates the frequency score when choosing among
qualifying candidates.

### Implementation

Add a helper method to `KnowledgeBaseDreamer`:

```python
def _frequency_score(self, kb: KnowledgeBase,
                     clauses: List[Union[Fact, Rule]]) -> float:
    """Compute frequency-weighted score for a set of clauses."""
    import math
    total = sum(kb.get_usage(c) for c in clauses)
    return 1 + math.log2(total + 1)
```

Multiply this into the candidate selection logic where candidates are compared.

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/knowledge.py` | **Edit** | Add `_usage_counts`, `record_usage`, `get_usage`, `reset_usage`, `total_queries_tracked`; update `copy()` and `restore_from()` |
| `dreamlog/evaluator.py` | **Edit** | Add `record_usage` calls on fact and rule match |
| `dreamlog/kb_dreamer.py` | **Edit** | Add Operation F, `_frequency_score`, frequency-weighted candidate selection |
| `tests/test_sleep_cycle.py` | **Edit** | Add frequency tracking and Operation F tests |

## Test Strategy

### Usage counter tests
- `record_usage` increments count
- `get_usage` returns 0 for unused clause
- `get_usage` returns correct count after multiple queries
- `reset_usage` clears all counts
- `total_queries_tracked` returns sum
- `copy()` preserves usage counts
- `restore_from()` restores usage counts
- Evaluator records usage on fact match
- Evaluator records usage on rule match
- Usage accumulates across multiple queries

### Operation F tests
- Dead facts removed after sufficient queries
- Dead rules removed after sufficient queries
- Used clauses preserved
- Threshold: not enough queries -> skip pruning
- Generated predicates (`_invented_`, `_extracted_`, `exception_`) not pruned even if 0 usage
- Verification: removing dead clause doesn't break derivations

### Frequency prioritization tests
- Higher-frequency candidate preferred over lower-frequency with same MDL
- Frequency score is logarithmic (not linear)
