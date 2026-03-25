# Subgroup Discovery for Operation C

**Date**: 2026-03-25
**Status**: Design
**Scope**: Improve Operation C to find compressible subgroups within functor/arity groups

## Context

Operation C currently anti-unifies ALL facts in a functor/arity group and only
proceeds if `variables_introduced == 1`. This means if any fact in the group
differs in more than one argument position, the entire group is skipped.

Example: `likes(alice, chocolate)`, `likes(bob, chocolate)`, `likes(carol,
chocolate)`, `likes(eve, vanilla)` anti-unifies to `likes(X, Y)` with 2
variables, so the entire group is skipped, even though the 3 chocolate facts
are a valid compression target.

## Design

### Algorithm

Replace the current "anti-unify all, check for 1 variable" approach with
argument-position partitioning:

```
for each functor/arity group with >= min_group_size facts:
    for each argument position p in range(arity):
        constant_pattern = all args except position p
        sub-group facts by their constant_pattern
        for each sub-group with >= min_group_size facts:
            varying_values = {fact.args[p] for fact in subgroup}
            find guard predicate for varying_values
            compute exceptions, MDL check
            verify candidate against verification suite
            if passes: apply compression, move to next subgroup
```

### Constant Pattern Key

For a fact `f(a, b, c)` with varying position `p=1`, the constant pattern key
is `(a, c)`. For position `p=0`, the key is `(b, c)`. Facts with the same
constant pattern key belong to the same subgroup.

### Greedy Application

When a subgroup passes MDL + verification, apply it immediately and remove those
facts from the KB. Subsequent subgroups operate on the remaining facts. This is
greedy but correct (no fact is used in two compressions).

After applying a compression, rebuild the facts list for subsequent iterations
since the KB has changed.

### Anti-unification Role

Anti-unification is no longer needed for discovery (the partition determines the
generalization by construction). However, `anti_unify_many` is still called on
each subgroup for the `shared_structure` score used in threshold filtering.

### What Does Not Change

- Guard predicate search (`_find_guard`)
- Exception computation
- MDL scoring
- Verification framework
- The `dream()` loop and Operations A and B

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/kb_dreamer.py` | **Edit** | Rewrite `_generalize_facts` method |
| `tests/test_sleep_cycle.py` | **Edit** | Update/add Operation C tests |

## Test Strategy

### New tests
- Mixed-value group compressed: `likes(a, choc), likes(b, choc), likes(c, choc), likes(eve, vanilla)` with person guard. The 3 chocolate facts compress, vanilla fact kept as-is.
- Multiple subgroups in one functor: `config(app, debug, true), config(app, verbose, true), config(app, color, false)` where position 2 has two subgroups (true and false).
- Higher arity: `f(a, X, c)` pattern found in 3-arity facts.

### Existing tests that must still pass
- All existing TestOperationC tests (basic_generalization, mdl_rejects, multi_variable_skipped, no_guard, guard_selects_smallest, exception_excluded, idempotent)
- The multi_variable_skipped test should now SUCCEED in finding a subgroup (if the subgroup within the multi-variable group is large enough), OR remain skipped if no subgroup meets the threshold. The test may need updating.

## Edge Cases

- All facts in a group have unique constant patterns for every position: no subgroup reaches min_group_size. No compression. Correct.
- A fact belongs to multiple potential subgroups (different varying positions): the greedy approach means whichever position is tried first and passes MDL wins. Order is by argument position (0, 1, 2, ...).
- After applying a compression, the remaining facts may form new subgroups. We do NOT re-scan (one pass per dream cycle). Re-scanning happens on the next dream() call.
