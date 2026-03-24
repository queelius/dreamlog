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

### Variable naming convention

Fresh variables introduced by anti-unification are named `_G0`, `_G1`, `_G2`, etc.
The `_G` prefix avoids collisions with user variables (which are uppercase-initial
by Prolog convention, e.g., `X`, `Y`, `Parent`). The counter resets per
anti-unification call.

### Multi-term anti-unification

`anti_unify_many([t1, ..., tn])` folds `anti_unify` pairwise over the list. The
lgg of first-order terms is unique up to variable renaming, so the final result is
correct regardless of fold order. However, intermediate results may have different
variable structure depending on fold order.

Note on substitution recovery: when folding, the substitution for each original
term must be composed through the fold chain. `anti_unify_many` computes this by
maintaining a list of composed substitutions at each fold step.

### Node count and shared_structure metric

A term's **node count** is:
- `Atom` or `Variable`: 1
- `Compound(f, args)`: 1 + sum(node_count(arg) for arg in args)

**Shared nodes** are positions where anti-unification returned the original
sub-term unchanged (identical in both inputs). **Non-shared nodes** are positions
where a fresh variable was introduced.

```
shared_structure = shared_node_count / total_node_count_of_generalized_term
```

Where `total_node_count_of_generalized_term` counts each node in the result,
and introduced variables count as non-shared (1 node each).

### Data structures

```python
@dataclass
class AntiUnificationResult:
    generalized: Term                        # the lgg
    substitutions: List[Dict[str, Term]]     # one mapping per input term
    variables_introduced: int                # positions where inputs differed
    shared_structure: float                  # ratio: shared nodes / total nodes
```

### Public API

```python
def anti_unify(term1: Term, term2: Term) -> AntiUnificationResult:
    """Compute the least general generalization of two terms."""

def anti_unify_many(terms: List[Term]) -> AntiUnificationResult:
    """Compute the lgg of multiple terms via pairwise folding."""

def node_count(term: Term) -> int:
    """Count nodes in a term tree."""
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

### Unknown hook suppression

When evaluating the inner goal G of `not(G)`, the unknown hook (LLM integration)
must be **disabled**. Without this, a failing `not(G)` check would trigger the LLM
hook to generate rules for G, defeating the purpose of the negation. NAF evaluation
creates a temporary evaluator (or disables the hook on the current one) for the
inner proof attempt.

### Implementation location

In `PrologEvaluator`, within the goal-solving method. When the goal term has
functor `not` and arity 1, dispatch to the NAF handler instead of normal
resolution. Approximately 20-25 lines of code.

### has_solution convenience method

```python
def has_solution(self, term: Term) -> bool:
    """Check if a term is derivable without materializing all solutions."""
    goal = Goal(term, {})
    return next(iter(self._solve_goals([goal], {}, 0)), None) is not None
```

The existing `ask_yes_no` method materializes all solutions via `list()`. The new
`has_solution` uses a generator-based approach that stops at the first solution.
This is critical for Operation B performance, which calls this once per fact.

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

**Scope restriction**: In this first pass, Operation A is restricted to:
1. **Fact vs. fact**: Fact F1 subsumes fact F2 if `subsumes(F1.term, F2.term)`.
2. **Fact vs. bodyless rule**: A bodyless rule `R: H.` (with no body) subsumes
   a fact F if `subsumes(R.head, F.term)`.
3. **Rule vs. rule (same body length only)**: Rule A subsumes rule B if:
   - A's head subsumes B's head under substitution theta
   - A and B have the same number of body goals
   - Each body goal in A (under theta) subsumes the body goal at the same
     position in B

Full theta-subsumption (which requires checking all permutations of body goal
mappings and is NP-complete in the general case) is deferred. Restricting to
same-length, same-position matching keeps this O(n * m) where n is the number of
body goals and m is the number of rules per functor group.

**Implementation**: For each functor/arity group, check all pairs. Use the
existing `subsumes()` function from `unification.py` for term-level checks. A
new `clause_subsumes(clause_a, clause_b) -> bool` utility function wraps the
term-level checks with the body-matching logic described above.

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
   a. Temporarily exclude F from the KB
   b. Query the remaining KB for F's term using `has_solution`
   c. If the query succeeds, mark F as redundant
2. Collect all independently-redundant facts
3. Remove them in a single batch
4. Post-removal verification: for each removed fact, re-check that it is still
   derivable from the post-removal KB

**Recovery on post-removal failure**: If any removed fact is no longer derivable
after the batch removal, fall back to one-at-a-time mode: restore all removed
facts, then iterate through them removing each one individually and re-checking
derivability after each removal. Only remove facts that remain derivable after
their own removal AND after all previously removed facts are also gone. This
handles cases where two facts provide mutual derivability through rule chains.

**Implementation**: Uses `PrologEvaluator.has_solution` (generator-based, stops at
first solution) for efficient derivability checks.

**Safety**: High. The derivability check is the proof of safety. The post-removal
verification and fallback provide a safety net.

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

**Precondition**: Operation C only applies to groups where anti-unification
introduces **exactly one variable** (one varying argument position). Groups where
anti-unification produces multiple variables are skipped. Multi-argument
generalization requires multi-arity guard predicates, which is deferred to future
work.

**Algorithm**:

**Step 1: Group and anti-unify**

Group facts by functor/arity. For each group with >= N facts (configurable,
default 3):
- Run `anti_unify_many` on all facts in the group
- If `shared_structure` < threshold (default 0.1), skip (trivially general)
- If `variables_introduced` != 1, skip (multi-variable case deferred)
- The result identifies one argument position that varies and all others that are
  constant across the group

**Step 2: Find a guard predicate**

The anti-unification result has one variable position. Collect the set of values V
that appear in that position across the original facts.

Search the KB for an existing unary predicate P such that:
- The set `{x : P(x) is a fact in KB}` is a superset of V
- P is not the same functor we're generalizing
- P is not an exception predicate (does not start with `exception_`)

If multiple candidate guards exist, select the one with the smallest extension
(fewest extra values beyond V), since this minimizes exceptions.

If no suitable guard exists, skip this group.

**Step 3: Compute exceptions and MDL score**

Exceptions are values where the guard predicate holds but the original fact does
not exist:
```
guard_values = {x : P(x) in KB}
fact_values  = {values from original facts in the varying position}
exceptions   = guard_values - fact_values
```

MDL criterion:
```
cost_before = len(original_facts)
cost_after  = 1 (the rule) + len(exceptions) (exception facts)
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

The exception predicate name is generated deterministically:
`exception_{original_functor}_{guard_predicate}` (e.g., `exception_likes_person`).
This avoids collisions with existing KB predicates.

**Step 5: Verify this individual candidate**

Before committing, verify the candidate against the verification suite (which was
built from the original KB state, before Operations A and B). Run both positive
queries (facts we replaced must still be derivable) and negative queries (nothing
new should become derivable). If verification fails, skip this candidate and move
to the next functor group.

**Step 6: Handle override values**

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
% 1. Anti-unify likes group: likes(X, chocolate)
%    shared_structure = 0.5 (1 shared node 'chocolate', 1 variable)
%    variables_introduced = 1 (OK, proceed)
% 2. Variable X takes {alice, bob, carol}
% 3. Guard search: person/1 has {alice, bob, carol, dave} (superset, smallest)
% 4. Exceptions: {dave}
% 5. MDL: before=3 likes facts, after=1 rule + 1 exception = 2. 2 < 3. Apply.
% 6. Verify: likes(alice, chocolate) still derivable? Yes. likes(dave, chocolate)
%    still non-derivable? Yes (blocked by exception). Passed.

% After (6 clauses)
person(alice). person(bob). person(carol). person(dave).
likes(X, chocolate) :- person(X), not(exception_likes_person(X)).
exception_likes_person(dave).
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

Before the sleep cycle begins (from the **original** KB state), build a
`VerificationSuite`:

**Positive queries** (must remain derivable after compression):
- Every fact in the KB, represented as its ground term
- For each rule, one instantiated query derived from the rule + existing facts
  (if possible; skip rules that can't be instantiated from current facts)

**Negative queries** (must remain non-derivable after compression):
- Restricted to **fact-defined functors only** (skip predicates defined solely by
  rules, since variable-headed rules make negative query construction ambiguous)
- For each fact-defined functor f/n, generate negative terms by taking an existing
  fact and substituting one argument position with a value from the KB's atom pool
  that does not appear in that position in any f/n fact. Keep all other positions
  identical to the existing fact.
- Generate at most `2 * len(positive_queries)` negative queries total to bound the
  verification cost.

**Example**: KB has `parent(john, mary)` and `parent(bob, alice)`. The atom pool
includes `{john, mary, bob, alice, carol}`. A negative query might be
`parent(carol, mary)` (carol does not appear in position 1 of any parent fact).

```python
@dataclass
class VerificationSuite:
    positive_queries: List[Term]    # must succeed
    negative_queries: List[Term]    # must fail

    def verify(self, kb: KnowledgeBase, evaluator_factory) -> VerificationResult:
        evaluator = evaluator_factory(kb)
        failures = []

        for q in self.positive_queries:
            if not evaluator.has_solution(q):
                failures.append(("false_negative", q))

        for q in self.negative_queries:
            if evaluator.has_solution(q):
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
  committing. The individual verification runs against the KB state **after**
  Operations A and B have already been applied. If verification fails, that
  candidate is skipped (not the entire sleep cycle).

### Rollback

The sleep cycle operates on a copy of the KB (via `kb.copy()`). Only if the final
verification passes does the compressed KB replace the original. If the final
verification fails, `kb.restore_from(snapshot)` reverts to the original state.
This guarantees atomicity: either the full sleep cycle succeeds, or the KB is
unchanged.

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
        # 0. Guard: empty KB
        original_size = len(kb)
        if original_size == 0:
            return DreamSession(
                compressed=False, operations=[],
                compression_ratio=1.0, verification=None
            )

        # 1. Snapshot for rollback
        snapshot = kb.copy()

        # 2. Build verification suite from original state
        suite = build_verification_suite(kb) if verify else None

        # 3. Operations (in order)
        result = None
        ops_applied = []
        ops_applied.extend(self._eliminate_subsumed(kb))         # Op A
        ops_applied.extend(self._prune_redundant_facts(kb))      # Op B
        ops_applied.extend(self._generalize_facts(kb, suite))    # Op C

        # 4. Final verification
        if verify and suite:
            result = suite.verify(kb, make_evaluator)
            if not result.passed:
                kb.restore_from(snapshot)
                return DreamSession(
                    compressed=False, operations=[],
                    compression_ratio=1.0, verification=result
                )

        # 5. Return results
        new_size = len(kb)
        return DreamSession(
            compressed=new_size < original_size,
            operations=ops_applied,
            compression_ratio=new_size / original_size,
            verification=result
        )
```

### DreamSession (revised)

```python
@dataclass
class DreamSession:
    compressed: bool                           # did the KB shrink?
    operations: List[CompressionCandidate]     # what was applied
    compression_ratio: float                   # new_size / old_size (< 1.0 = good)
    verification: Optional[VerificationResult] # pass/fail + details, None if skipped

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

- `copy() -> KnowledgeBase`: Deep copy for rollback (copies all facts, rules, and
  indices)
- `restore_from(other: KnowledgeBase)`: Replace this KB's contents with another's
  (for rollback)
- `remove_fact_by_value(fact: Fact)`: Remove a specific fact by equality (not by
  index)
- `remove_rule_by_value(rule: Rule)`: Remove a specific rule by equality
- `replace_facts(old: List[Fact], new: List[Union[Fact, Rule]])`: Atomic
  replacement for Operation C (removes old facts, adds new facts and/or rules)

### Evaluator additions needed

- `not/1` handler (NAF), with unknown hook suppression for the inner proof
- `has_solution(term: Term) -> bool`: Generator-based, stops at first solution
- `FlounderingError` exception class

### New utility needed

- `clause_subsumes(clause_a, clause_b) -> bool`: Clause-level subsumption built
  on top of the existing `subsumes()` function, implementing the restricted
  same-body-length matching described in Operation A.

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/anti_unification.py` | **New** | Anti-unification algorithm |
| `dreamlog/kb_dreamer.py` | **Rewrite** | Sleep cycle with three operations |
| `dreamlog/evaluator.py` | **Edit** | Add `not/1`, `has_solution`, `FlounderingError` |
| `dreamlog/knowledge.py` | **Edit** | Add `copy`, `restore_from`, value-based removal, `replace_facts` |
| `dreamlog/unification.py` | **Edit** | Add `clause_subsumes` utility |
| `tests/test_anti_unification.py` | **New** | Anti-unification unit tests |
| `tests/test_sleep_cycle.py` | **New** | Integration tests for the dream loop |
| `tests/test_naf.py` | **New** | Negation as failure tests |

## Test Strategy

### Anti-unification tests
- Identical terms return unchanged
- Different atoms introduce variable with `_G` prefix
- Same-pair consistency: `f(a, a)` vs `f(b, b)` -> `f(_G0, _G0)`
- Different-pair distinctness: `f(a, b)` vs `f(b, a)` -> `f(_G0, _G1)`
- Nested compound terms: `f(g(a), h(b))` vs `f(g(c), h(d))` -> `f(g(_G0), h(_G1))`
- Mismatched functors: `f(a)` vs `g(a)` -> `_G0`
- Mismatched arities: `f(a, b)` vs `f(a)` -> `_G0`
- Multi-term anti-unification (fold correctness)
- Substitution recovery: `generalized.substitute(sub_i) == original_i` for each i
- `shared_structure` scoring: known inputs produce expected scores
- `node_count` for atoms, variables, nested compounds

### NAF tests
- `not(known_fact)` fails
- `not(unknown_fact)` succeeds
- `not(derivable_from_rule)` fails
- `not(not(G))` succeeds iff G succeeds (double negation)
- `FlounderingError` on non-ground goal
- NAF in rule body (exception clause pattern)
- Unknown hook is NOT triggered during NAF inner proof

### Sleep cycle integration tests
- **Operation A**: Specific rule removed when general rule exists (same body length)
- **Operation A**: Fact subsumed by bodyless rule is removed
- **Operation A**: Rules with different body lengths are NOT compared
- **Operation B**: Derivable fact removed
- **Operation B**: Mutually-dependent facts handled correctly (fallback to one-at-a-time)
- **Operation B**: Non-derivable fact kept
- **Operation C**: Group of facts compressed to rule + exceptions
- **Operation C**: MDL rejects when cost_after >= cost_before (e.g., 2 facts)
- **Operation C**: Multi-variable anti-unification result is skipped
- **Operation C**: Guard predicate search selects smallest extension
- **Operation C**: No suitable guard found -> skip
- **Operation C**: Exception predicates excluded from future generalization
- **Verification**: Over-generating rule caught by negative queries
- **Rollback**: Failed final verification restores original KB
- **Idempotence**: Running dream twice produces same result
- **Empty KB**: Returns immediately, no crash
- **KB with only facts, no rules**: Operations B and C still apply where possible
- **KB with only rules**: Operation A (subsumption elimination) still works

## Design Decisions and Constraints

### Exception predicate hygiene

Exception predicates (e.g., `exception_likes_person`) are generated by the sleep
cycle and should be excluded from future generalization passes. The naming
convention (`exception_` prefix) serves as the exclusion marker. Guard predicate
search also skips predicates with this prefix.

### Operation sequencing matters

Operations run in A -> B -> C order. This is intentional:
- A (subsumption) simplifies the rule set, which may reveal new redundant facts
- B (pruning) removes derivable facts, which reduces the input to C
- C (generalization) works on the remaining facts

The verification suite is built from the **original** KB. Individual Op C
verification runs against the KB state after A and B.

### What this first pass cannot do

- Generalize across multiple varying argument positions (multi-variable)
- Discover cross-functor relationships (e.g., father = parent + male)
- Invent new predicates beyond exception guards
- Compress rule bodies (common sub-goals)
- Handle rule subsumption with body reordering (NP-complete in general)

These are explicitly deferred, not overlooked.

## Deferred to Future Work

- `call/N` and predicate invention (cross-functor abstraction)
- Multi-argument guard predicates and multi-variable generalization
- Rule body pattern extraction (common sub-goals across rules)
- Full theta-subsumption with body reordering (NP-complete, needs heuristics)
- Derivation tree tracking and compression (DreamCoder-faithful)
- LLM-assisted compression (creative leaps beyond symbolic reach)
- Dream journal and experience replay
- Weighted MDL scoring (clause complexity, not just count)
