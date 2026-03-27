# Sleep Cycle Findings

**Date**: 2026-03-27
**Benchmark version**: baseline.json (7 scenarios, 315 total clauses)

## Summary

The symbolic sleep cycle compresses a 315-clause benchmark suite to 268 clauses
(0.85 ratio) and 84 body goals to 68 (0.81 ratio) with 100% correctness across
all scenarios. Six operations fire: subsumption elimination (A), redundant fact
pruning (B), fact generalization with exceptions (C), predicate invention (D),
body pattern extraction (E), and dead clause pruning (F).

## What works well

**Operation C (fact generalization)** is the most reliable compressor for data-
heavy KBs. When a group of facts shares structure in all but one argument
position, and a guard predicate exists, the compression is clean and significant.
The subgroup discovery enhancement (partitioning by argument position) handles
mixed-value groups that would otherwise be skipped entirely.

**Operation D (predicate invention)** discovers genuine abstractions autonomously.
The transitive closure pattern (ancestor/reachable/connected) is detected and
extracted as `_invented_0(R, X, Y) :- call(R, X, Y)` with wrapper rules. This
is DreamCoder's library extraction operating on logic programs. The skeleton
fingerprinting correctly distinguishes structurally identical rule sets from
superficially similar ones (same structure but same base relation = nothing to
parameterize).

**Operation F (dead clause pruning)** is effective when the wake phase is broad
enough. The 50% predicate coverage threshold prevents false positives from narrow
query patterns. In the dead_clauses scenario, 9 of 16 clauses (56%) are correctly
identified and removed.

**Verification framework** catches every bad transformation. The positive/negative
query suite with atomic rollback means no operation can silently break the KB.
The extension for rule-derived queries (needed for Op D) works correctly.

## Limitations discovered

**Operation E (body pattern extraction) adds clauses.** Extracting a shared body
sub-sequence always creates +1 new rule. With clause-count MDL, this looks like
expansion even when structural complexity decreases (body_patterns: 11->12 clauses
but 11->9 body goals). A body-goal-weighted MDL would properly reward this, but
clause count remains the primary metric. Implication: Op E is most valuable for
longer shared sequences (K >= 3) or many sharing rules (N >= 4), where the body
goal savings outweigh the clause cost.

**Operation D requires 3+ instances for K=2 rule sets.** The MDL criterion
`K + N < N * K` means two predicates with 2 rules each (the most common case:
base + recursive) don't compress. You need 3+ structurally identical predicates.
This is mathematically correct but limits Op D's utility in smaller KBs. For
K=3+ rule sets, only 2 instances are needed.

**Operation D only parameterizes one body functor.** Rule sets where multiple
body functors differ (e.g., `f(X,Y) :- g(X,Z), h(Z,Y)` vs `f(X,Y) :- j(X,Z),
k(Z,Y)` with both g/j and h/k differing) are skipped. Multi-parameter invention
is deferred.

**Verification is the performance bottleneck.** In the stress scenario (185
clauses, 50 entities), verification accounts for ~85% of the 6.5s runtime. Each
verification query on a transitive closure predicate requires O(n) recursive
resolution steps. This is inherent to correctness guarantees on recursive KBs,
not a bug. Operations themselves (A through E) complete in <200ms.

**Operation F depends on wake-phase breadth.** If the wake phase only exercises a
subset of predicates, unused predicates are marked dead even if they are valuable.
The 50% predicate coverage threshold mitigates this but doesn't eliminate it. A
KB that is loaded but not yet queried broadly will see no Op F benefit.

**family_small (12 clauses) gets zero compression.** Too few facts per functor
(no group reaches min_group_size=3), only one rule (K=1, Op D threshold not met),
and no shared body patterns. This is expected: small KBs don't have enough
redundancy to compress.

## Cross-operation interactions

The cascading scenario demonstrates operations feeding each other:
- Op B removes a redundant `reach_e(a, b)` fact (derivable from rule + edge fact)
- Op C generalizes `active(_, true)` facts using `node` as guard
- Op D discovers `reach_e`, `reach_l`, `reach_p` share transitive closure structure
- Op E extracts `edge(X,Y), edge(Y,Z)` prefix from `two_hop`, `three_hop`,
  `hop_then_link`

Operations run in A->B->C->D->E->F order. Earlier operations simplify the KB
for later ones. Op D creates `_invented_` predicates that Op E correctly skips.
Op C creates `exception_` predicates that Op D correctly skips.

## Performance characteristics

| Scenario | Clauses | Time | Rate |
|---|---|---|---|
| family_small (12) | 12->12 | 32ms | 375 clauses/s |
| family_with_guards (43) | 43->39 | 298ms | 144 clauses/s |
| transitive_closures (22) | 22->20 | 121ms | 182 clauses/s |
| body_patterns (11) | 11->12 | 96ms | 115 clauses/s |
| dead_clauses (16) | 16->7 | 40ms | 400 clauses/s |
| cascading (26) | 26->23 | 203ms | 128 clauses/s |
| stress (185) | 185->155 | 7231ms | 26 clauses/s |

The rate drops sharply for the stress scenario because verification cost scales
with KB size and recursion depth, not linearly with clause count.

## Baseline for LLM-assisted features

The symbolic-only baseline:
- **Clauses**: 315 -> 268 (0.85 ratio)
- **Body goals**: 84 -> 68 (0.81 ratio)
- **Operations**: 22 total (9 dead_clause, 4 extraction, 4 invention, 3 generalization, 2 pruning)
- **Correctness**: 100%

LLM-assisted features should improve on this baseline in two ways:

1. **LLM-assisted naming**: No compression change, but `_invented_0` becomes
   `transitive_closure`, `_extracted_0` becomes `grandparent_chain`. Improves
   human interpretability without affecting the numbers.

2. **LLM-assisted compression**: Should find cross-functor relationships that
   symbolic methods miss. For example, given `father(X,Y)` and `parent(X,Y)` and
   `male(X)` facts, an LLM could propose `father(X,Y) :- parent(X,Y), male(X)`,
   which the symbolic sleep cycle cannot discover (it requires cross-functor
   reasoning that anti-unification within a single functor doesn't capture).
   This should improve the compression ratio, especially on the family_with_guards
   and stress scenarios. The benchmark baseline.json enables before/after
   comparison.

## Open questions

1. Should Op E use a body-goal-weighted MDL instead of clause count? The current
   metric penalizes extraction even when structural complexity decreases.

2. Should Op F use a more nuanced threshold than 50% predicate coverage? A
   time-decay model (older unused clauses are more likely dead) could be more
   principled.

3. Should the dream cycle run multiple passes? Currently each dream() call runs
   A-F once. Multiple passes could find cascading opportunities that single-pass
   misses (e.g., Op E creates a new predicate that Op D could then parameterize).

4. How should verification scale for large KBs? The current approach (sample
   ground queries from atom pool) is O(atoms^arity) per predicate. For KBs with
   1000+ entities and arity-3 predicates, this becomes prohibitive.
