# Rule Body Pattern Extraction (Operation E)

**Date**: 2026-03-25
**Status**: Design
**Scope**: Extract common contiguous sub-goal sequences from rule bodies

## Context

Operations A-D compress at the clause level: removing subsumed/redundant clauses,
generalizing facts, and inventing predicates from structurally identical rule sets.
None of them look *inside* rule bodies for shared sub-goal patterns.

Real knowledge bases often have rules that share body prefixes or sub-sequences.
For example, many family relationship rules start with `parent(X, Y), parent(Y, Z)`
(the "grandparent chain"). Extracting these shared sequences as named predicates
compresses the KB and creates reusable abstractions.

## Algorithm

### Step 1: Index sub-sequences

For each rule R with body goals [G1, G2, ..., Gn], enumerate all contiguous
sub-sequences of length >= 2:
- [G1, G2], [G2, G3], ..., [Gn-1, Gn]
- [G1, G2, G3], ..., [Gn-2, Gn-1, Gn]
- etc.

For each sub-sequence, compute a **structural key** that captures the functor
names, arities, and variable connectivity within the sub-sequence (ignoring
variable names). Two sub-sequences match if they have the same structural key.

### Step 2: Group and score

Group sub-sequences by structural key. For each group with N >= 2 occurrences
and sub-sequence length K, compute savings:

```
savings = (K - 1) * (N - 1) - 1   (measured in body goals)
```

Compress when savings > 0, equivalently `(K-1)*(N-1) > 1`:
- K=2: need N >= 3
- K=3: need N >= 2
- K >= 4: need N >= 2

### Step 3: Select best candidate

From all qualifying groups, select the one with the highest savings
(`K * N - K - N`). In case of ties, prefer longer sub-sequences (more
abstraction).

### Step 4: Compute interface variables

The extracted predicate must expose exactly the variables that connect the
sub-sequence to the rest of each rule (the "interface"). For a sub-sequence
S within rule R:

```
internal_vars = variables appearing ONLY within S (not in head or remaining body)
external_vars = variables appearing in S AND in (head or remaining body)
interface_vars = external_vars (ordered by first appearance in S)
```

The extracted predicate's arguments are the interface variables. Internal
variables become local to the extracted predicate.

**Example:** `great_grandparent(X, W) :- parent(X, Y), parent(Y, Z), parent(Z, W)`

Sub-sequence `parent(X, Y), parent(Y, Z)`:
- Variables in sub-sequence: {X, Y, Z}
- Variables in head + remaining body: {X, W, Z}
- Interface = {X, Y, Z} intersect {X, W, Z} = {X, Z}
- Y is internal (only appears within the sub-sequence)
- Extracted predicate: `_extracted_0(X, Z) :- parent(X, Y), parent(Y, Z).`

### Step 5: Build extracted predicate and rewrite rules

Create:
```prolog
_extracted_N(V1, ..., Vm) :- sub_sequence_goals.
```

For each rule containing the sub-sequence, replace the sub-sequence goals with
a single call to the extracted predicate:
```prolog
% Before
great_grandparent(X, W) :- parent(X, Y), parent(Y, Z), parent(Z, W).
% After
great_grandparent(X, W) :- _extracted_0(X, Z), parent(Z, W).
```

### Step 6: Verify and apply

Verify against the verification suite (extended for rule-derived queries). If
verification passes, apply the extraction. Then re-scan for new shared patterns
(one extraction may reveal new opportunities). Repeat until no more candidates
qualify.

### Greedy with re-scan

After each extraction, re-index sub-sequences from the modified rules. This
handles cascading extractions where shortening rules reveals new shared patterns.
Limit to a maximum of 10 extraction rounds per dream cycle to prevent runaway
loops.

## Structural key for sub-sequences

A sub-sequence's structural key captures:
1. For each goal: (functor_name, arity)
2. Variable connectivity within the sub-sequence: which argument positions share
   variables (same representation as skeleton variable_map, but scoped to the
   sub-sequence goals only)

Variable names are normalized to `_S0`, `_S1`, ... in order of first appearance
within the sub-sequence. Two sub-sequences match if they have the same functor
sequence and the same normalized variable connectivity.

**Important:** Unlike skeleton extraction, we do NOT abstract functor names. The
sub-sequence `parent(X,Y), parent(Y,Z)` does NOT match `edge(A,B), edge(B,C)`.
Functor names are part of the structural key. Cross-functor body patterns are
already handled by Operation D.

## MDL metric

Operation E uses body-goal-weighted MDL, not clause-count MDL:

```
cost_before = sum of body lengths of all affected rules
cost_after  = K (extracted predicate body length)
            + sum of (original_length - K + 1) for each affected rule
savings     = cost_before - cost_after = N*K - K - N = (K-1)*(N-1) - 1
```

This is separate from the clause-count MDL used by Operations A-D. Operation E
adds 1 new rule (the extracted predicate) but does not remove any rules. The
benefit is measured in reduced total body complexity, not clause count.

## Naming

Extracted predicates are named `_extracted_0`, `_extracted_1`, etc. Counter
determined by max-scan of existing `_extracted_N` predicates in the KB (same
pattern as `_invented_N`).

## Operation ordering

E runs after D. The full order is: A -> B -> C -> D -> E.

D finds whole rule-set structural patterns across predicates. E finds sub-goal
patterns within individual rule bodies. Running D first means E operates on
the post-invention KB, where wrapper rules are short (1 body goal each) and
invented predicate rules have the original structure. E may find patterns in
the invented predicate bodies or in rules that D didn't touch.

## Interaction with generated predicates

Skip rules whose head functor starts with `_extracted_`, `_invented_`, or
`exception_`. These are system-generated and should not be further decomposed
in this pass.

## File Plan

| File | Action | Description |
|------|--------|-------------|
| `dreamlog/kb_dreamer.py` | **Edit** | Add Operation E |
| `tests/test_sleep_cycle.py` | **Edit** | Add Operation E tests |

## Test Strategy

- Three rules sharing a 2-goal prefix: extracted (N=3, K=2, savings=1)
- Two rules sharing a 3-goal prefix: extracted (N=2, K=3, savings=1)
- Two rules sharing a 2-goal prefix: rejected (N=2, K=2, savings=0)
- Interface variables computed correctly (internal vars hidden, external exposed)
- Extracted predicate produces correct query results
- Sub-sequence at end of body (not just prefix) is detected
- Sub-sequence in middle of body is detected
- Re-scan finds cascading patterns after first extraction
- Generated predicates (`_extracted_`, `_invented_`) excluded from scanning
- Idempotent: running dream twice gives same result
- Full end-to-end: Operations A-E all run, KB is maximally compressed

## Example walkthrough

```prolog
% Before (3 rules, 8 total body goals)
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
great_grandparent(X, W) :- parent(X, Y), parent(Y, Z), parent(Z, W).
great_uncle(X, W) :- parent(X, Y), parent(Y, Z), brother(Z, W).

% Sub-sequence parent(X, Y), parent(Y, Z) found in all 3 rules
% K=2, N=3, savings = (2-1)*(3-1) - 1 = 1 body goal
% Interface vars: X (used in heads), Z (used in remaining goals)
% Y is internal

% After (4 rules, 7 total body goals)
_extracted_0(X, Z) :- parent(X, Y), parent(Y, Z).
grandparent(X, Z) :- _extracted_0(X, Z).
great_grandparent(X, W) :- _extracted_0(X, Z), parent(Z, W).
great_uncle(X, W) :- _extracted_0(X, Z), brother(Z, W).
```

## Deferred

- Cross-functor sub-sequence matching (handled by Operation D)
- Non-contiguous sub-goal patterns (e.g., goals 1 and 3 but not 2)
- Parameterized extraction with `call/N` (combining Operations D and E)
