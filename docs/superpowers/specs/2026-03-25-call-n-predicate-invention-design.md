# call/N and Predicate Invention via Identical Rule-Set Detection

**Date**: 2026-03-25
**Status**: Design
**Scope**: Add `call/N` to the evaluator and Operation D (cross-functor predicate invention) to the sleep cycle

## Context

The sleep cycle currently compresses facts within a single predicate (Operation C).
It cannot detect that two predicates (e.g., `ancestor` and `reachable`) are
structurally identical rule sets differing only in functor names. This is the
logic programming equivalent of DreamCoder's library extraction: discovering that
multiple programs share the same structure and factoring it out as a reusable
primitive.

To express the extracted abstraction, the system needs `call/N`, which
parameterizes a predicate name at runtime.

## Part 1: `call/N` in the Evaluator

### Specification

When the evaluator encounters a goal with functor `call` and arity >= 2:

1. Extract the first argument F (must be a ground Atom). If F is an unbound
   Variable, raise an `InstantiationError`. If F is a Compound (not Atom), also
   raise `InstantiationError`.
2. Extract the remaining arguments `[X1, ..., Xn]`.
3. Construct `Compound(F.value, [X1, ..., Xn])`.
4. Evaluate this constructed term as a normal goal (facts, rules, recursion).

The minimum arity is 2 (`call(F, X)` = call F with one arg). `call/1` (`call(F)`
with zero additional args) is not supported in this pass since the semantics of
0-arity predicate dispatch are ambiguous (`Atom` vs `Compound("f", [])` have
different equality in the existing term system). If encountered, raise
`InstantiationError` with a descriptive message.

### Examples

```prolog
call(parent, john, mary)   % becomes parent(john, mary)
call(edge, a, b)           % becomes edge(a, b)
call(likes, X, chocolate)  % becomes likes(X, chocolate) -- X can be unbound
```

### Error handling

```python
class InstantiationError(Exception):
    """Raised when call/N receives a non-ground or invalid functor."""
```

### Implementation location

In `PrologEvaluator._solve_goals`, as an intercept before normal resolution (same
pattern as the NAF handler). Approximately 15-20 lines.

The `call/N` handler should respect the unknown hook: if the constructed goal has
no matching facts or rules, the hook fires normally (unlike NAF which suppresses
the hook). This means the `call/N` handler does NOT create a separate evaluator
instance; it constructs the compound goal and then falls through to normal
resolution by replacing the current goal with the constructed term.

## Part 2: Rule-Set Skeleton Extraction

### New module: `dreamlog/skeleton.py`

A **skeleton** captures the structure of a rule set while abstracting away functor
names. Two predicates with the same skeleton are structurally identical.

### Skeleton construction

Given a predicate P defined by rules R1, ..., Rk:

**Step 1: Identify functor roles.** In each rule, classify each body functor:
- `SELF`: same functor as the head (recursive calls)
- `PARAM_i`: non-recursive body functors, assigned positional indices in order of
  first appearance across the rule set

All non-SELF body functors are treated as potentially parameterizable.

**Body goals containing `not/1` wrappers are opaque.** Skeleton extraction treats
`not(inner)` as a single goal with functor `not` and arity 1. It does not examine
the inner term. Rules containing `not/1` body goals will only match other rules
that also have `not/1` in the same position. In practice, the `param_count == 1`
restriction excludes Operation C-generated rules (which have both a guard functor
and a `not(exception_...)` body goal = 2 varying functors).

**Step 2: Normalize variables.** Rename all variables to `_V0`, `_V1`, `_V2`, ...
in order of first appearance (left-to-right through head args, then body goal args
in order). This makes variable naming deterministic across rule sets.

**Step 3: Sort rules within the set.** To make comparison order-independent, sort
rules by (body length, then by the normalized body content). This handles cases
where rules are defined in different orders.

### Variable connectivity representation

The variable connectivity captures which argument positions share variables across
goals. Each position is identified as `(goal_index, arg_index)` where
`goal_index = -1` for the head.

For each normalized variable `_Vi` (ordered by i), record a sorted tuple of all
positions where it appears:

```python
# f(X, Z) :- g(X, Y), f(Y, Z).
# _V0 = X: ((-1, 0), (0, 0))
# _V1 = Z: ((-1, 1), (1, 1))
# _V2 = Y: ((0, 1), (1, 0))
variable_map = (
    ((-1, 0), (0, 0)),         # _V0 positions
    ((-1, 1), (1, 1)),         # _V1 positions
    ((0, 1), (1, 0)),          # _V2 positions
)
```

Concrete type: `tuple[tuple[tuple[int, int], ...], ...]`. Singleton variables
(appearing once) are included since they still constrain the skeleton structure.

### Skeleton data structure

```python
@dataclass(frozen=True)
class RuleSkeleton:
    """Skeleton of a single rule."""
    head_arity: int
    body: tuple  # tuple of (functor_role: str, arity: int) pairs
    variable_map: tuple  # see variable connectivity above

@dataclass(frozen=True)
class RuleSetSkeleton:
    """Skeleton of a complete rule set (all rules for one predicate)."""
    rules: tuple  # tuple of RuleSkeleton, sorted
    param_count: int  # number of distinct PARAM_i functors
```

Both are frozen dataclasses and support `__hash__` and `__eq__` via the default
frozen behavior (structural comparison of all fields).

### Public API

```python
def extract_skeleton(
    predicate: str, rules: List[Rule]
) -> Tuple[RuleSetSkeleton, Dict[str, str]]:
    """Extract skeleton and functor mapping from a rule set.

    Args:
        predicate: The head functor name
        rules: All rules for this predicate

    Returns:
        skeleton: The structural skeleton (hashable, comparable)
        functor_map: Maps PARAM_0, PARAM_1, ... to actual functor names
    """
```

## Part 3: Operation D in the Sleep Cycle

### Extended verification suite

The existing `build_verification_suite` generates positive queries only from KB
facts. Operation D transforms rules, not facts. To verify Operation D correctly,
the verification suite must be extended:

**Before running Operation D**, for each predicate that has rules (not just facts):
1. Run a bounded set of ground queries against the pre-transformation KB using
   existing facts as seed values.
2. Record which queries succeed as additional positive queries.
3. Record which queries fail as additional negative queries.

This is done by `extend_verification_for_rules(suite, kb)`, called in the
`dream()` method after Operations A/B/C but before Operation D. It adds
rule-derived positive/negative queries to the existing suite.

The query generation strategy: for each rule-defined predicate P with arity n,
take the Cartesian product of atom values that appear in the KB's facts (bounded
to a reasonable sample), construct ground queries, and test them against the
current KB.

### Algorithm

**Step 1: Group predicates by skeleton.**

For each predicate P in the KB (identified by unique head functor), collect its
rules and extract the skeleton. Group predicates by skeleton.

Skip predicates with:
- Zero rules (fact-only predicates, Operation D only considers rules)
- Functors starting with `_invented_` or `exception_` (generated predicates)

Predicates that have both facts and rules: Operation D considers only the rules.
Facts for that predicate are left untouched.

**Step 2: For each group with >= 2 predicates, evaluate MDL.**

For a group of N predicates, each with K rules:
```
cost_before = N * K
cost_after = K (invented predicate rules) + N (wrapper rules)
compress if K + N < N * K
```

Equivalently: `(N - 1) * (K - 1) > 1`:
- K=1: never compress (single-rule patterns produce no gain)
- K=2: need N >= 3
- K >= 3: need N >= 2

Only groups where the skeleton has `param_count == 1` are considered in this pass.

**Step 3: Build the invented predicate.**

Given a group with skeleton S and N members:

1. Pick one member as the template.
2. Add a new first argument `R` (the functor parameter) to the head of each rule.
3. Replace each occurrence of the PARAM_0 functor in body goals with
   `call(R, original_args)`.
4. Self-recursive calls become calls to the invented predicate with `R` threaded
   through.

Naming: `_invented_N` where N is determined by scanning the KB for existing
`_invented_*` predicates and using `max + 1`. This is stateless and idempotent.

**Step 4: Build wrapper rules.**

For each member predicate P with functor mapping {PARAM_0: actual_functor}:
```prolog
P(X1, ..., Xn) :- _invented_k(actual_functor, X1, ..., Xn).
```

**Step 5: Verify and apply.**

Verify against the extended verification suite. If it passes, remove the original
rules for all member predicates and add the invented predicate rules + wrapper
rules. Facts for member predicates are preserved.

### Scope restriction

This first pass handles rule sets where exactly **one** non-recursive body functor
varies across instances (`param_count == 1`). Rule sets with multiple varying
body functors (including those generated by Operation C with guard + exception)
are skipped.

### Example walkthrough

```prolog
% Before (6 rules)
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
reachable(X, Y) :- edge(X, Y).
reachable(X, Z) :- edge(X, Y), reachable(Y, Z).
connected(X, Y) :- link(X, Y).
connected(X, Z) :- link(X, Y), connected(Y, Z).

% Skeletons (all identical):
% SELF(_V0,_V1) :- PARAM_0(_V0,_V1).
% SELF(_V0,_V1) :- PARAM_0(_V0,_V2), SELF(_V2,_V1).
%
% Group of 3, K=2 rules each. MDL: 6 -> 2+3 = 5. Compress.

% After (5 rules)
_invented_0(R, X, Y) :- call(R, X, Y).
_invented_0(R, X, Z) :- call(R, X, Y), _invented_0(R, Y, Z).
ancestor(X, Y) :- _invented_0(parent, X, Y).
reachable(X, Y) :- _invented_0(edge, X, Y).
connected(X, Y) :- _invented_0(link, X, Y).
```

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/evaluator.py` | **Edit** | Add `call/N` handler and `InstantiationError` |
| `dreamlog/skeleton.py` | **New** | Skeleton extraction, normalization, comparison |
| `dreamlog/kb_dreamer.py` | **Edit** | Add Operation D, extend verification for rules |
| `tests/test_call.py` | **New** | `call/N` evaluator tests |
| `tests/test_skeleton.py` | **New** | Skeleton extraction and comparison tests |
| `tests/test_sleep_cycle.py` | **Edit** | Add Operation D integration tests |

## Test Strategy

### `call/N` tests
- `call(parent, john, mary)` resolves like `parent(john, mary)`
- `call(F, X, Y)` with F bound via unification works
- `call` with partially-bound args: `call(parent, X, mary)` with X free
- `call` with unbound first arg raises `InstantiationError`
- `call` with Compound first arg raises `InstantiationError`
- `call/1` (zero additional args) raises `InstantiationError`
- `call/N` inside rule body works (the predicate invention pattern)
- `call/N` triggers unknown hook when predicate is undefined
- `call/N` nested inside `not/1`: `not(call(pred, X))` works correctly
- `call/N` recursion depth: invented predicate with 10+ hops still resolves

### Skeleton tests
- Single non-recursive rule produces correct skeleton
- Recursive rule: SELF detected in body
- Variable normalization is deterministic
- Two structurally identical rule sets produce the same skeleton
- Different arities produce different skeletons
- Different body lengths produce different skeletons
- Different variable connectivity produces different skeletons
- Rule ordering does not affect skeleton (sorted)
- Same PARAM functor appearing in multiple body goals is handled correctly
- Rules with `not/1` body goals: `not` is treated as opaque (not decomposed)

### Operation D integration tests
- Three transitive-closure predicates compressed to invented + wrappers
- Two predicates with K=2: MDL rejects (K+N=4, N*K=4, not less)
- Two predicates with K=3: MDL accepts (K+N=5, N*K=6)
- Generated `_invented_` predicates excluded from future detection
- Predicates with different skeletons not grouped
- Single-rule predicates skipped (K=1 never compresses)
- Verification: derived queries (e.g., ancestor(john, sue) with 2 hops) still work
- Wrapper rules correctly dispatch via `call/N`
- Idempotent: running dream twice gives same result
- Predicate with both facts and rules: facts preserved, rules transformed
- KB with existing `_invented_` predicates from previous dream: counter increments

## Design Decisions

### Invented predicate naming

`_invented_0`, `_invented_1`, etc. Counter determined by max-scan of existing
`_invented_N` predicates in the KB (stateless, idempotent). The underscore prefix
marks them as system-generated. LLM-assisted naming is deferred.

### Operation ordering

Operation D runs after A, B, and C. This is intentional:
- A removes subsumed rules (simplifies what D sees)
- B removes redundant facts (reduces noise)
- C compresses fact groups (may reveal rule-only patterns)
- D detects cross-functor rule patterns (builds on simplified KB)

### `not/1` in skeleton extraction

Body goals wrapped in `not/1` are treated as opaque single-functor goals. The
skeleton records `("not", 1)` without examining the inner term. This is correct
for the `param_count == 1` restriction, which excludes Operation C-generated rules
(those have both a guard and a `not(exception_...)` = 2 varying body functors).

### What this does NOT do

- Multiple varying body functors (multiple PARAM_i)
- Sub-goal extraction from rule bodies (step #3 in the roadmap)
- Named invented predicates (needs LLM)
- Detecting near-identical rule sets (e.g., same structure but different arities)
- `call/1` (0-arity dispatch)

## Deferred to Future Work

- Multiple-parameter predicate invention
- Rule body pattern extraction (common sub-goal sequences)
- LLM-assisted naming of invented predicates
- Derivation tree tracking and compression
- `call/1` support (requires resolving Atom vs Compound equality for 0-arity terms)
