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

When the evaluator encounters a goal with functor `call` and arity >= 1:

1. Extract the first argument F (must be a ground Atom). If F is an unbound
   Variable, raise an `InstantiationError` (we cannot enumerate all predicates).
2. Extract the remaining arguments `[X1, ..., Xn]`.
3. Construct `Compound(F.value, [X1, ..., Xn])`.
4. Evaluate this constructed term as a normal goal (facts, rules, recursion).

### Examples

```prolog
call(parent, john, mary)   % becomes parent(john, mary)
call(edge, a, b)           % becomes edge(a, b)
call(likes, X, chocolate)  % becomes likes(X, chocolate)
```

### Error handling

```python
class InstantiationError(Exception):
    """Raised when call/N receives a non-ground functor."""
```

If the first argument is a Variable after substitution, raise `InstantiationError`.
If the first argument is a Compound (not an Atom), also raise `InstantiationError`.

### Implementation location

In `PrologEvaluator._solve_goals`, as an intercept before normal resolution (same
pattern as the NAF handler). Approximately 15 lines.

The `call/N` handler should respect the unknown hook: if the constructed goal has
no matching facts or rules, the hook fires normally (unlike NAF which suppresses
the hook).

## Part 2: Rule-Set Skeleton Extraction

### New module: `dreamlog/skeleton.py`

A **skeleton** captures the structure of a rule set while abstracting away functor
names. Two predicates with the same skeleton are structurally identical.

### Skeleton construction

Given a predicate P defined by rules R1, ..., Rk:

**Step 1: Identify functor roles.** In each rule, classify each functor:
- `SELF`: the head functor (appears in the head and possibly in recursive body
  calls)
- `PARAM_i`: non-recursive body functors, assigned positional indices in order of
  first appearance across the rule set
- `FIXED`: body functors that are the same in this position across all instances
  (discovered during comparison, not during extraction)

For skeleton extraction, we treat all non-SELF body functors as potentially
parameterizable and assign them `PARAM_0`, `PARAM_1`, etc.

**Step 2: Normalize variables.** Rename all variables to `_V0`, `_V1`, `_V2`, ...
in order of first appearance (left-to-right through head, then body goals). This
makes variable naming consistent across rule sets.

**Step 3: Sort rules within the set.** To make comparison order-independent, sort
rules by (body length, then lexicographic order of the normalized body). This
handles cases where rules are defined in different orders.

### Skeleton data structure

```python
@dataclass(frozen=True)
class RuleSkeleton:
    """Skeleton of a single rule."""
    head_arity: int
    body: tuple  # tuple of (functor_role, arity) pairs
    variable_map: tuple  # normalized variable connectivity

@dataclass(frozen=True)
class RuleSetSkeleton:
    """Skeleton of a complete rule set (all rules for one predicate)."""
    rules: tuple[RuleSkeleton, ...]  # sorted, hashable
    param_count: int  # number of PARAM_i functors

    def __hash__(self):
        return hash(self.rules)

    def __eq__(self, other):
        return isinstance(other, RuleSetSkeleton) and self.rules == other.rules
```

### Public API

```python
def extract_skeleton(predicate: str, rules: List[Rule]) -> Tuple[RuleSetSkeleton, Dict[str, str]]:
    """Extract skeleton and functor mapping from a rule set.

    Returns:
        skeleton: The structural skeleton (hashable, comparable)
        functor_map: Maps PARAM_0, PARAM_1, ... to actual functor names
    """

def skeletons_match(s1: RuleSetSkeleton, s2: RuleSetSkeleton) -> bool:
    """Check if two skeletons are structurally identical."""
```

### Variable connectivity representation

The variable connectivity must capture which argument positions share variables
across goals. For example:

```prolog
f(X, Z) :- g(X, Y), f(Y, Z).
```

Variables: X appears in head[0] and body[0][0]. Y appears in body[0][1] and
body[1][0]. Z appears in head[1] and body[1][1].

This is represented as a tuple of (first_occurrence, all_occurrences) per variable,
where occurrences are encoded as (goal_index, arg_index) positions. Goal index -1
is the head. This fully captures the "wiring" between positions.

## Part 3: Operation D in the Sleep Cycle

### Algorithm

**Step 1: Group predicates by skeleton.**

For each predicate P in the KB (identified by unique head functor), collect its
rules and extract the skeleton. Group predicates by skeleton.

Skip predicates with:
- Zero rules (fact-only predicates)
- Functors starting with `_invented_` or `exception_` (generated predicates)

**Step 2: For each group with >= 2 predicates, evaluate MDL.**

For a group of N predicates, each with K rules and P varying parameters:
```
cost_before = N * K
cost_after = K (invented predicate rules) + N (wrapper rules)
compress if K + N < N * K
```

Simplification: `K + N < N * K` is equivalent to `(N - 1) * (K - 1) > 1`, which
means:
- K=1: never compress (single-rule patterns produce no gain)
- K=2: need N >= 3
- K >= 3: need N >= 2

**Step 3: Build the invented predicate.**

Given a group with skeleton S and N members:

1. Pick one member as the template (the first one). Use its rules.
2. For each `PARAM_i`, add a new argument to the head of each rule.
3. Replace each occurrence of the PARAM_i functor in body goals with
   `call(ParamVar, original_args)`.
4. Self-recursive calls (SELF) become calls to the invented predicate with the
   param passed through.

Naming: `_invented_0`, `_invented_1`, ... (counter shared across dream cycles,
or based on max existing `_invented_N` in KB).

**Step 4: Build wrapper rules.**

For each member predicate P with functor mapping {PARAM_0: actual_functor}:
```prolog
P(X1, ..., Xn) :- _invented_k(actual_functor, X1, ..., Xn).
```

**Step 5: Verify and apply.**

Verify against the verification suite. If it passes, remove the original rules
for all member predicates and add the invented predicate rules + wrapper rules.

### Scope restriction

This first pass handles rule sets where exactly **one** non-recursive body functor
varies across instances (`param_count == 1`). Rule sets with multiple varying
body functors are skipped.

### Example walkthrough

```prolog
% Before (4 rules)
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
reachable(X, Y) :- edge(X, Y).
reachable(X, Z) :- edge(X, Y), reachable(Y, Z).
connected(X, Y) :- link(X, Y).
connected(X, Z) :- link(X, Y), connected(Y, Z).

% Skeletons:
% ancestor: SELF(_V0,_V1) :- PARAM_0(_V0,_V1). SELF(_V0,_V1) :- PARAM_0(_V0,_V2), SELF(_V2,_V1).
% reachable: same skeleton, PARAM_0=edge
% connected: same skeleton, PARAM_0=link
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
| `dreamlog/kb_dreamer.py` | **Edit** | Add Operation D |
| `tests/test_call.py` | **New** | `call/N` evaluator tests |
| `tests/test_skeleton.py` | **New** | Skeleton extraction and comparison tests |
| `tests/test_sleep_cycle.py` | **Edit** | Add Operation D integration tests |

## Test Strategy

### `call/N` tests
- `call(parent, john, mary)` resolves like `parent(john, mary)`
- `call(F, X, Y)` with F bound via unification works
- `call` with unbound first arg raises `InstantiationError`
- `call` with Compound first arg raises `InstantiationError`
- `call/N` inside rule body works (the predicate invention pattern)
- `call/N` triggers unknown hook when predicate is undefined
- `call` with zero additional args: `call(foo)` becomes `foo` (0-arity compound)

### Skeleton tests
- Single non-recursive rule produces correct skeleton
- Recursive rule: SELF detected in body
- Variable normalization is deterministic and order-independent
- Two structurally identical rule sets produce same skeleton
- Different arities produce different skeletons
- Different body lengths produce different skeletons
- Different variable connectivity produces different skeletons
- Rule ordering does not affect skeleton (sorted)

### Operation D integration tests
- Three transitive-closure predicates compressed to invented + wrappers
- Two predicates with K=2: MDL rejects (K+N = 4, N*K = 4, not less)
- Two predicates with K=3: MDL accepts (K+N = 5, N*K = 6, less)
- Generated `_invented_` predicates excluded from future detection
- Predicates with different skeletons not grouped
- Single-rule predicates skipped (K=1 never compresses)
- Verification: all original queries still work after invention
- Wrapper rules correctly dispatch to invented predicate via `call/N`
- Idempotent: running dream twice gives same result

## Design Decisions

### Invented predicate naming

`_invented_0`, `_invented_1`, etc. The underscore prefix marks them as system-
generated (like `exception_` predicates). LLM-assisted naming (e.g., choosing
"transitive_closure" instead of "_invented_0") is deferred.

### Operation ordering

Operation D runs after A, B, and C. This is intentional:
- A removes subsumed rules (simplifies what D sees)
- B removes redundant facts (reduces noise)
- C compresses fact groups (may reveal rule-only patterns)
- D detects cross-functor rule patterns (builds on simplified KB)

### What this does NOT do

- Multiple varying body functors (multiple PARAM_i)
- Sub-goal extraction from rule bodies (step #3 in the roadmap)
- Named invented predicates (needs LLM)
- Detecting near-identical rule sets (e.g., same structure but different arities)

## Deferred to Future Work

- Multiple-parameter predicate invention
- Rule body pattern extraction (common sub-goal sequences)
- LLM-assisted naming of invented predicates
- Derivation tree tracking and compression
