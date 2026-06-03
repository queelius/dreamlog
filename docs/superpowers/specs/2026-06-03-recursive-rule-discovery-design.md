# Recursive Rule Discovery in DreamLog

**Date**: 2026-06-03
**Status**: Design
**Scope**: Discover transitive-closure recursive rules (e.g. `ancestor`) from an unannotated fact base, via a symbolic closure-detection operation and a recursion-aware Operation G prompt, verified with the bounded evaluator. Post-NeSy-2026 research spine.
**Branch**: `recursion-discovery` (off `v0.12.0` == `paper-v1.0`)
**Related**: current paper "Compression Enables Generalization" (paper-v1.0); EX25b/EX25c protocol; `docs/superpowers/specs/2026-03-29-derivation-trees-design.md`

## Context

DreamLog's sleep operations discover within-predicate generalizations (Op C) and cross-predicate rules (Op G), but they cannot discover recursive definitions. The canonical example is `ancestor`: the family experiments (EX25) recover `father`, `grandparent`, etc., but never `ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)`, because symbolic operations work on flat fact patterns and the Op G prompt does not invite self-reference. EX26 had to exclude `ancestor` from the Popper comparison for an unrelated reason (Popper's clingo backend ignores the Python timeout), so the recursion gap is currently disclosed as a limitation in both systems.

Recursion is the single largest untapped compression opportunity in the KB. In the family domain, the 204 derived facts include the entire `ancestor` closure. Discovering the two-clause recursive definition lets Op B prune every `ancestor` fact, a large MDL gain that no current operation can reach. The compression incentive already exists; what is missing is (1) a discovery mechanism and (2) safe verification over an unbounded closure. That gap is the research.

Timing is post-submission: touching recursion reaches into the verification suite, the EX01 two-iteration convergence claim, and the EX26 fairness argument, so the current paper ships as-is and this becomes the next spine.

## Research framing

### The claim

DreamLog discovers recursive rules (transitive-closure definitions such as `ancestor` and reachability) from an unannotated fact base, and the invented-vocabulary protocol separates genuine structural discovery from LLM recall of textbook recursion.

### The three-role insight

In the current paper the raw-LLM baseline scores 0% on the crafting domain because the missing rules are semantic and vocabulary-bound. Recursion splits the LLM's contribution into three roles that behave differently, which makes this a richer experiment rather than a re-run:

1. **As inference engine** (the raw-LLM baseline asks "does `flux_reaches(a,z)` hold?"): still fails on invented vocabulary, because it cannot compute a transitive closure over 100-plus facts in-context. The 0% baseline survives, so the protocol still isolates compression.
2. **As semantic proposer**: irrelevant here, because closure is structural, not semantic.
3. **As structural-schema proposer** (Op G proposes the rule): may succeed even on invented terms, because "X to Y, Y to Z implies X to Z" is a vocabulary-independent shape it can pattern-match from the fact distribution.

### Discriminating outcomes (all shippable)

1. Symbolic closure detection recovers recursion on invented vocabulary with no LLM: the strong "compression alone discovers recursion" result.
2. The LLM proposes correct recursion on invented vocabulary and the symbolic side verifies it: "structural priors transfer; symbolic verification makes them sound."
3. Both routes succeed: the cleanest result, two independent paths to recursion with MDL adjudicating which to keep.

## Scope (v1)

In scope:
- Transitive closure of a single binary base relation (`R == TC(B)` for binary `R`, `B`).
- Right-recursion only: `R(X,Z) :- B(X,Y), R(Y,Z)` plus base case `R(X,Y) :- B(X,Y)`.

Out of scope (future work, named explicitly so the spec stays focused):
- General, mutual, or nested recursion; accumulator patterns; list recursion (member/append-style).
- Left-recursion (`R(X,Z) :- R(X,Y), B(Y,Z)`), excluded as a termination safeguard.
- Closure over a derived or multi-relation base in one step.

## Experimental design (EX27)

Follows the EX25b template (four conditions, holdout recovery, mean over runs/seeds).

### Canonical recursive domain

The EX25 family tree. Base relation `parent`; target `ancestor == TC(parent)` (irreflexive). The LLM has seen this definition, so it is the confounded control.

### Invented-vocabulary closure domain

A synthetic graph over invented node names and an invented base relation `flux_links(a,b)`, with `flux_reaches(X,Z)` defined as the transitive closure of `flux_links`. Generated deterministically (parallel to the crafting generator in `ex25b_novel_generalization.py`): N invented nodes, a random sparse DAG of `flux_links` edges, and the materialized `flux_reaches` closure as derived facts. The LLM has never seen these tokens, so success here is the isolation result.

### Conditions

1. **no-dream**: baseline, no compression.
2. **symbolic-only**: dream with Operations A to F plus the new Operation I (closure detection), no LLM.
3. **full-pipeline**: dream with all operations including the recursion-aware Op G.
4. **raw-LLM**: prompt the LLM directly with the facts and held-out closure queries (yes/no), the inference-engine role.

### Metrics and holdout

Hold out a fraction of the closure facts (reuse the EX25c open-world holdout machinery). Measure:
- **Recovery**: fraction of held-out closure facts derivable from the compressed KB.
- **Precision**: no spurious reachability (no derived pair outside the true closure).
- **Discovered rules** and **compression ratio**, as in the current tables.

## Mechanism

### Operation I: symbolic closure detection (new, no LLM)

A new sleep operation in a dedicated module `dreamlog/recursive_discovery.py`, called by the dreamer as a flag-gated pipeline step that is off by default (preserving the current zero-drift behavior). Shape of the algorithm (illustrative, exact KB API confirmed during implementation):

```python
def detect_transitive_closures(kb, min_base_facts=4):
    """Find binary predicates R that are the irreflexive transitive
    closure of some binary base relation B already in the KB."""
    binary = kb.binary_predicates()          # functor/arity == 2
    found = []
    for R in binary:
        r_ext = kb.extension(R)              # set of (x, y) ground pairs
        for B in binary:
            if B == R:
                continue
            b_ext = kb.extension(B)
            if len(b_ext) < min_base_facts:
                continue
            if r_ext == transitive_closure(b_ext):   # irreflexive TC
                found.append((R, B))
    return found
```

For each `(R, B)` found, synthesize the right-recursive pair and offer it to the standard accept path:

```python
base = Rule(head=R(X, Y), body=[B(X, Y)])
rec  = Rule(head=R(X, Z), body=[B(X, Y), R(Y, Z)])
# accept iff verification passes and MDL improves:
#   2 clauses replace |r_ext| stored facts  ->  always a large gain
```

`transitive_closure(edges)` is standard (BFS or Warshall from each node). Cost is roughly O(V * E); fine for the experiment sizes, noted as a scaling concern for large KBs.

**Pipeline position**: Op I runs in the symbolic phase, after Op E and before Op G, so structural closure detection happens before the LLM step. In symbolic-only mode Op G is skipped as usual, so Op I is the only route to recursion. In full-pipeline mode both Op I and the recursion-aware Op G can fire, and MDL plus verification adjudicate: a correct closure found by Op I makes any later LLM proposal of the same rule redundant (Op A subsumption or the at-least-2-fact gate drops the duplicate). This ordering also means the symbolic-only and full-pipeline conditions differ only by Op G, keeping the ablation clean.

### Operation G: recursion-aware prompt

A new prompt template variant (in `llm_prompt_templates.py` / `prompt_template_system.py`) that explicitly invites recursive definitions, asking for a base case and a recursive case together, and showing one right-recursive example. Gated by a flag so the recursion prompt can be ablated on and off. The existing Op G cycle filter already permits self-recursion (it blocks only cross-functor cycles), and the response parser already handles multi-rule output, so no machinery change is required beyond the prompt and a small acceptance check that a recursive proposal arrives with its base case.

### Verification

Reuses the existing verification suite (`build_verification_suite`) and bounded evaluator (`max_total_calls`, scaled with KB size). Recursion-specific points:

- **Termination**: right-recursion only (enforced in both Op I synthesis and the Op G prompt guidance). The bounded evaluator caps total resolution calls so a pathological proposal cannot loop indefinitely; the cap must be high enough to derive a full closure but finite.
- **Soundness over the closure**: the synthetic negatives `S-` already substitute atoms at positions where they do not appear, which for a closure relation yields non-reachable near-miss pairs. A correct recursive rule derives none of them; an over-general one is caught. Confirm `S-` density is sufficient for the closure domains.
- **Open-world recovery**: in closed-world a correct rule derives nothing new (all closure facts present); in open-world/holdout mode (already built for EX25c) the rule derives exactly the held-out facts, which is the recovery metric. No new verification mode is needed.

### MDL and the Op B payoff

Clause-count MDL already favors recursion overwhelmingly: two clauses replace the N stored closure facts. This is the one place the paper's "clause-count is coarse" caveat does not bite, because the compression is large and unambiguous, so no MDL refinement is required for v1. Once the recursive rule lands, Op B's SLD-derivability check succeeds for every closure fact via the recursion and prunes them, which is both the payoff and the largest single compression in the KB.

## Implementation surface

Minimal and isolated, preserving the zero-drift default.

| Change | File | Notes |
|---|---|---|
| Op I closure detection | new `dreamlog/recursive_discovery.py` | isolated; dreamer calls it as a flag-gated step, off by default |
| recursion-aware Op G prompt | `llm_prompt_templates.py`, `prompt_template_system.py` | new template variant plus a toggle |
| evaluator hardening | `dreamlog/evaluator.py` | confirm and guard right-recursion plus depth; mostly tests, minimal code |
| dreamer wiring | `dreamlog/kb_dreamer.py` | call Op I behind a flag; extend the verification suite handling if needed |
| experiment | new `experiments/ex27_recursion.py` | four conditions, parallels `ex25b` |
| invented-closure domain | generator in the experiment or a small data module | deterministic, invented nodes plus a sparse DAG plus materialized closure |
| registry | `experiments/experiment_registry.yaml` | register EX27 with motivation, method, success criteria |
| tests | `tests/test_recursive_discovery.py` and additions to `tests/test_sleep_cycle.py` | unit, verification, regression |

## De-risking: the ancestor feasibility probe (go/no-go)

Before building Op I, the domain, or the prompt, the first task is a feasibility probe on `ancestor`:

1. Load the EX25 family KB.
2. Hand-write the right-recursive `ancestor` pair into the KB.
3. Confirm the bounded evaluator verifies it without looping (all `ancestor` closure facts remain derivable, no `S-` pair becomes derivable).
4. Confirm Op B then prunes the `ancestor` facts (SLD-derivability via the recursion).

If this fails even with right-recursion, the evaluator's recursion handling becomes the contribution and we replan before investing in Op I and the domain. This probe is the gate; it should take an afternoon, not a week.

## Risks and mitigations

1. **Termination / left-recursion (highest risk)**: restrict to right-recursion everywhere; the bounded evaluator is the backstop; the feasibility probe surfaces any evaluator weakness immediately.
2. **The LLM trivially proposes recursion on invented vocabulary**: this is a finding (structural priors transfer), not a failure. The symbolic Op I condition is the hedge: if symbolic-only also recovers the closure, the strong result holds regardless of the LLM.
3. **Closure detection cost at scale**: O(V * E) per candidate pair; fine for experiment sizes, flagged as a scaling note, not a v1 blocker.
4. **Over-general recursion admitted**: rely on `S-` near-miss negatives plus the closed-world false-positive check; confirm `S-` density on the closure domains during the probe.
5. **Op G recursion prompt destabilizes existing results**: the recursion prompt is a separate, toggled template; full-pipeline-without-recursion-prompt remains available as a control, and the symbolic ops are unchanged.

## Testing

- **Unit**: closure detection on known small graphs (chain, tree, DAG, disconnected, cycle-free vs cyclic base); negative cases where `R` is not a closure.
- **Verification**: a synthesized recursive rule preserves `S+`, rejects `S-`, and terminates under the bounded evaluator.
- **Integration**: EX27 runs all four conditions end to end and produces the comparison table.
- **Regression**: the existing 671 tests stay green; Op I off by default must not change any current behavior.

## Success criteria

The investigation is "done enough to decide if it is a paper" when:

1. The ancestor feasibility probe passes (verify plus prune under recursion).
2. Op I recovers `ancestor` on the family domain (symbolic-only condition is non-zero on recursion).
3. EX27 runs all four conditions on both the canonical and invented-closure domains and produces recovery, precision, and compression numbers.
4. The outcome is characterized against the three discriminating outcomes above, with the raw-LLM-as-engine baseline confirmed failing on invented vocabulary.
5. Existing tests stay green and new tests cover Op I and recursive verification.
