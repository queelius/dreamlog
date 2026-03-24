# Sleep Cycle First Pass: Symbolic Compression via Anti-Unification

**Date**: 2026-03-24
**Status**: Design
**Scope**: First working implementation of DreamLog's sleep/dream phase

## Context

DreamLog implements a wake-sleep architecture inspired by DreamCoder. During wake
phases, the system answers queries via SLD resolution, optionally generating new
rules via LLM when undefined predicates are encountered. During sleep phases, the
system should compress and optimize the knowledge base, discovering generalizations
and removing redundancy.

Currently the sleep phase is scaffolding: `KnowledgeBaseDreamer` detects patterns
but every `_apply_*` method is a no-op (`pass`). This design makes the sleep cycle
real.

## Design Principles

**Compression is learning.** Following Solomonoff induction and DreamCoder's core
thesis, the shortest description of the knowledge that preserves its deductive
closure is the best generalization. The MDL (Minimum Description Length) criterion
is the reward signal: compress only when the result is shorter than the original.

**Symbolic first.** All operations in this first pass are deterministic symbolic
transformations. No LLM calls. This makes the sleep cycle testable, reproducible,
and principled. LLM-assisted compression is a future layer on top.

**Safety through verification.** Every transformation is checked against a
verification suite that ensures deductive closure is preserved. Transformations
that break existing derivations are rolled back.

## New Module: `dreamlog/anti_unification.py`

### Algorithm

Anti-unification (Plotkin 1970, Reynolds 1970) computes the least general
generalization (lgg) of two terms. It is the dual of unification: where
unification finds the most general common *instance*, anti-unification finds
the most specific common *generalization*.

```
anti_unify(t1, t2, seen_pairs) -> (generalized, sub1, sub2)

1. If t1 == t2: return t1 unchanged (shared structure preserved)
2. If (t1, t2) in seen_pairs: return the variable already assigned (consistency)
3. If both Compound with same functor and arity:
   - Anti-unify each argument pair recursively
   - Return Compound(functor, [anti-unified args])
4. Otherwise:
   - Introduce fresh variable V
   - Record (t1, t2) -> V in seen_pairs
   - Record V -> t1 in sub1, V -> t2 in sub2
   - Return V
```

The `seen_pairs` dictionary ensures consistency: when the same pair of differing
subterms appears in multiple argument positions, the same variable is reused.

```
anti_unify(f(a, a), f(b, b)) -> f(X, X)     # same pair (a,b) reuses X
anti_unify(f(a, b), f(b, a)) -> f(X, Y)     # different pairs, different vars
```

The substitution mappings allow recovery of originals:
`generalized.substitute(sub1) == t1` (and similarly for sub2).

### Multi-term anti-unification

`anti_unify_many([t1, ..., tn])` folds `anti_unify` pairwise over the list.
The lgg of first-order terms is unique up to variable renaming, so fold order
does not affect correctness.

### Data structures

```python
@dataclass
class AntiUnificationResult:
    generalized: Term                        # the lgg
    substitutions: List[Dict[str, Term]]     # one mapping per input term
    variables_introduced: int                # positions where inputs differed
    shared_structure: float                  # ratio: shared nodes / total nodes
```

`shared_structure` measures how much the inputs had in common:
- Near 1.0: inputs are nearly identical (few variables introduced)
- Near 0.0: inputs share almost no structure (mostly variables)
- This score directly informs whether generalization is useful

### Public API

```python
def anti_unify(term1: Term, term2: Term) -> AntiUnificationResult:
    """Compute the least general generalization of two terms."""

def anti_unify_many(terms: List[Term]) -> AntiUnificationResult:
    """Compute the lgg of multiple terms via pairwise folding."""
```

## Evaluator Enhancement: Negation as Failure

### Specification

Add `not/1` to `PrologEvaluator`. When the current goal is `not(G)`:

1. Attempt to prove G using normal SLD resolution
2. If G has any solution: `not(G)` fails (yield nothing)
3. If G has no solutions: `not(G)` succeeds (yield current bindings unchanged)

### Soundness constraint

G must be ground (or sufficiently instantiated) when `not` is evaluated. NAF on
goals with unbound variables is unsound ("floundering"). The evaluator raises
`FlounderingError` if `not` is called on a non-ground goal.

```python
class FlounderingError(Exception):
    """Raised when not/1 is applied to a non-ground goal."""
```

### Implementation location

In `PrologEvaluator`, within the goal-solving method. When the goal term has
functor `not` and arity 1, dispatch to the NAF handler instead of normal
resolution. Approximately 15-20 lines of code.

### Implications

NAF makes the system non-monotonic: adding a fact can invalidate previously
derivable conclusions. This is the correct behavior for real-world data (learning
that tweety is a penguin should change whether tweety flies). The verification
framework accounts for this.

## Sleep Cycle Operations

The rewritten `KnowledgeBaseDreamer` performs three operations in sequence. Each
is independently useful and independently testable.

### Operation A: Subsumption Elimination

**Purpose**: Remove clauses that are logically entailed by more general clauses
already in the KB.

**Algorithm for rules**: Rule A subsumes rule B if:
1. A's head subsumes B's head under some substitution theta
2. Every body goal in A (under theta) subsumes a corresponding body goal in B
3. A != B (don't self-subsume)

**Algorithm for facts vs rules**: Fact F is subsumed by rule R if:
1. R has an empty body (it's a universal rule)
2. R's head subsumes F's term

**Implementation**: For each functor/arity group, check all pairs. Use the
existing `subsumes()` function from `unification.py`. O(n^2) per functor group,
but functor-based indexing keeps groups small.

**Safety**: Verified by construction. If A subsumes B, everything derivable via B
is derivable via A. No runtime check needed.

**Example**:
```prolog
% Before
ancestor(X, Y) :- parent(X, Y).
ancestor(john, Y) :- parent(john, Y).

% After (second rule removed, subsumed by first)
ancestor(X, Y) :- parent(X, Y).
```

### Operation B: Redundant Fact Pruning

**Purpose**: Remove facts that are derivable from the remaining KB (rules + other
facts).

**Algorithm**:
1. For each fact F in the KB:
   a. Temporarily exclude F
   b. Query the remaining KB for F's term
   c. If the query succeeds, mark F as redundant
2. Collect all independently-redundant facts
3. Remove them in a single batch
4. Post-removal verification: re-check that all removed facts are still derivable

The batch approach avoids cascading errors where removing fact A makes fact B
no longer derivable.

**Implementation**: Uses the existing `PrologEvaluator` for derivability checks.
Need a way to query "is this term derivable?" which is just running a query and
checking if any solution exists.

**Safety**: High. The derivability check is the proof of safety. The post-removal
verification is a safety net.

**Example**:
```prolog
% Before
parent(john, mary).
ancestor(X, Y) :- parent(X, Y).
ancestor(john, mary).               % redundant: derivable from rule + fact

% After
parent(john, mary).
ancestor(X, Y) :- parent(X, Y).
```

### Operation C: Fact Generalization with Exceptions

**Purpose**: Replace groups of facts with a general rule plus explicit exceptions,
when doing so compresses the KB.

**Algorithm**:

**Step 1: Group and anti-unify**

Group facts by functor/arity. For each group with >= N facts (configurable,
default 3):
- Run `anti_unify_many` on all facts in the group
- If `shared_structure` < threshold (default 0.1), skip (trivially general)
- The result identifies which argument positions are constant (shared across all
  facts) and which vary

**Step 2: Find a guard predicate**

The anti-unification result has variable positions where the original facts
differ. We need to *bound* those variables to prevent the generalized rule from
over-generating.

Search the KB for an existing unary predicate P such that:
- The set {x : P(x) is a fact in KB} is a superset of the values that appear in
  the varying positions
- P is not the same functor we're generalizing

If no suitable guard exists, skip this group (we cannot safely generalize without
a bound on the variables).

**Step 3: Compute exceptions and MDL score**

Exceptions are values where the guard predicate holds but the original fact does
not exist:
```
exceptions = {x : P(x) in KB} - {values from original facts}
```

MDL criterion:
```
cost_before = len(original_facts)
cost_after  = 1 (rule) + len(exceptions) (exception facts)
compress if cost_after < cost_before
```

**Step 4: Apply transformation**

Replace the original facts with:
```prolog
f(constant_args..., X, more_constants...) :- guard(X), not(exception_f_guard(X)).
exception_f_guard(e1).
exception_f_guard(e2).
...
```

The exception predicate name is generated deterministically from the original
functor and the guard predicate to avoid collisions.

**Step 5: Handle override values**

If an exception entity has a *different* value for the same predicate (not just
absence), keep the original fact as an override:
```prolog
% dave likes vanilla, not chocolate (override, not just exception)
likes(dave, vanilla).
```

**Example walkthrough**:
```prolog
% Before (7 clauses)
person(alice). person(bob). person(carol). person(dave).
likes(alice, chocolate). likes(bob, chocolate). likes(carol, chocolate).

% Sleep cycle:
% 1. Anti-unify likes group: likes(X, chocolate), shared_structure=0.5
% 2. Variable X takes {alice, bob, carol}
% 3. Guard: person/1 has {alice, bob, carol, dave} (superset)
% 4. Exceptions: {dave}
% 5. MDL: before=3 likes facts, after=1 rule + 1 exception = 2. 2 < 3. Apply.

% After (6 clauses)
person(alice). person(bob). person(carol). person(dave).
likes(X, chocolate) :- person(X), not(exception_likes_chocolate(X)).
exception_likes_chocolate(dave).
```

## MDL Scoring

Description length of a clause:
- **First pass**: Each clause (fact or rule) costs 1. Total DL = clause count.
- **Future refinement**: Weight by complexity (number of terms, nesting depth,
  number of variables).

```python
@dataclass
class CompressionCandidate:
    operation: str                              # "subsumption", "pruning", "generalization"
    original_clauses: List[Union[Fact, Rule]]   # removed
    new_clauses: List[Union[Fact, Rule]]        # added

    @property
    def mdl_delta(self) -> int:
        """Negative = compression (good). Positive = expansion (bad)."""
        return len(self.new_clauses) - len(self.original_clauses)

    @property
    def is_worth_it(self) -> bool:
        return self.mdl_delta < 0
```

For operations A and B, `new_clauses` is always empty (pure removal), so
`mdl_delta` is always negative. Operation C is the only one that needs the
explicit check.

## Verification Framework

### Verification suite construction

Before the sleep cycle begins, build a `VerificationSuite`:

**Positive queries** (must remain derivable after compression):
- Every fact in the KB, as a query
- For each rule, one instantiated query derived from the rule + existing facts

**Negative queries** (must remain non-derivable after compression):
- For each functor f/n in the KB, generate terms with argument values that do NOT
  appear in any fact or rule head for f/n
- These catch over-generation (e.g., a rule that's too permissive)

```python
@dataclass
class VerificationSuite:
    positive_queries: List[Term]    # must succeed
    negative_queries: List[Term]    # must fail

    def verify(self, kb: KnowledgeBase, evaluator_factory) -> VerificationResult:
        evaluator = evaluator_factory(kb)
        failures = []

        for q in self.positive_queries:
            if not has_solution(evaluator, q):
                failures.append(("false_negative", q))

        for q in self.negative_queries:
            if has_solution(evaluator, q):
                failures.append(("false_positive", q))

        return VerificationResult(
            passed=len(failures) == 0,
            failures=failures,
            positive_count=len(self.positive_queries),
            negative_count=len(self.negative_queries)
        )
```

### Per-operation verification

- **Operations A and B**: Verified by construction (subsumption/derivability
  proofs). No runtime suite check needed, but the final suite check serves as
  safety net.
- **Operation C**: Each generalization candidate is verified individually before
  committing. If verification fails, that candidate is skipped (not the entire
  sleep cycle).

### Rollback

The sleep cycle operates on a copy of the KB. Only if the final verification
passes does the compressed KB replace the original. This guarantees atomicity:
either the full sleep cycle succeeds, or the KB is unchanged.

## The Dream Loop

### Orchestration

```python
class KnowledgeBaseDreamer:
    """Symbolic sleep-phase compression via anti-unification and MDL."""

    def __init__(self, min_group_size: int = 3,
                 shared_structure_threshold: float = 0.1):
        self.min_group_size = min_group_size
        self.shared_structure_threshold = shared_structure_threshold

    def dream(self, kb: KnowledgeBase, verify: bool = True) -> DreamSession:
        # 1. Snapshot for rollback
        original_kb = deep_copy(kb)
        original_size = len(kb)

        # 2. Build verification suite
        suite = build_verification_suite(kb) if verify else None

        # 3. Operations (in order)
        ops_applied = []
        ops_applied.extend(self._eliminate_subsumed(kb))         # Op A
        ops_applied.extend(self._prune_redundant_facts(kb))      # Op B
        ops_applied.extend(self._generalize_facts(kb))           # Op C

        # 4. Final verification
        if verify and suite:
            result = suite.verify(kb, make_evaluator)
            if not result.passed:
                # Rollback
                restore(kb, original_kb)
                return DreamSession(
                    compressed=False,
                    operations=[],
                    compression_ratio=1.0,
                    verification=result
                )

        # 5. Return results
        return DreamSession(
            compressed=True,
            operations=ops_applied,
            compression_ratio=len(kb) / original_size,
            verification=result
        )
```

### DreamSession (revised)

```python
@dataclass
class DreamSession:
    compressed: bool                           # did the KB change?
    operations: List[CompressionCandidate]     # what was applied
    compression_ratio: float                   # new_size / old_size (< 1.0 = good)
    verification: Optional[VerificationResult] # pass/fail + details

    @property
    def clauses_removed(self) -> int:
        return sum(-op.mdl_delta for op in self.operations)
```

### Removed: LLMProvider dependency

The dreamer no longer takes an `LLMProvider` in its constructor. This is a pure
symbolic module. A future `LLMAssistedDreamer` can extend it to use LLMs for
creative leaps that symbolic methods cannot make.

## Interface Changes

### KnowledgeBase additions needed

- `copy() -> KnowledgeBase`: Deep copy for rollback
- `remove_fact_by_value(fact: Fact)`: Remove a specific fact (not by index)
- `remove_rule_by_value(rule: Rule)`: Remove a specific rule (not by index)
- `replace_facts(old: List[Fact], new: List[Union[Fact, Rule]])`: Atomic
  replacement for Operation C

### Evaluator additions needed

- `not/1` handler (NAF)
- `has_solution(term: Term) -> bool`: Convenience method, True if any solution
  exists (avoids materializing all solutions)

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/anti_unification.py` | **New** | Anti-unification algorithm |
| `dreamlog/kb_dreamer.py` | **Rewrite** | Sleep cycle with three operations |
| `dreamlog/evaluator.py` | **Edit** | Add `not/1` and `has_solution` |
| `dreamlog/knowledge.py` | **Edit** | Add `copy`, value-based removal, `replace_facts` |
| `tests/test_anti_unification.py` | **New** | Anti-unification unit tests |
| `tests/test_sleep_cycle.py` | **New** | Integration tests for the dream loop |
| `tests/test_naf.py` | **New** | Negation as failure tests |

## Test Strategy

### Anti-unification tests
- Identical terms return unchanged
- Different atoms introduce variable
- Same-pair consistency (f(a,a) vs f(b,b) -> f(X,X))
- Different-pair distinctness (f(a,b) vs f(b,a) -> f(X,Y))
- Compound with shared functor, differing args
- Mismatched functors or arities
- Multi-term anti-unification
- Substitution recovery (generalized.substitute(sub_i) == original_i)
- shared_structure scoring

### NAF tests
- not(known_fact) fails
- not(unknown_fact) succeeds
- not(derivable_from_rule) fails
- Floundering error on non-ground goal
- NAF in rule body (exception clause pattern)

### Sleep cycle integration tests
- Subsumption elimination: specific rule removed when general exists
- Redundant fact pruning: derivable fact removed
- Fact generalization: group of facts compressed to rule + exceptions
- MDL gating: transformation rejected when cost_after >= cost_before
- Verification: over-generating rule caught by negative queries
- Rollback: failed verification restores original KB
- Idempotence: running dream twice produces same result
- Empty KB: no crash
- KB with only facts, no rules: operations still apply where possible
- KB with only rules: subsumption elimination still works

## Deferred to Future Work

- `call/N` and predicate invention (cross-functor abstraction)
- Rule body pattern extraction (common sub-goals across rules)
- Derivation tree tracking and compression (DreamCoder-faithful)
- LLM-assisted compression (creative leaps beyond symbolic reach)
- Dream journal and experience replay
- Weighted MDL scoring (clause complexity, not just count)
- Multi-argument guard predicates (current design uses unary guards only)
