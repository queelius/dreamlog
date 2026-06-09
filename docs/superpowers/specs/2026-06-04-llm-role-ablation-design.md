# LLM-Role Ablation: Structure vs Semantics (EX28)

**Date**: 2026-06-04
**Status**: Design
**Scope**: A controlled ablation that isolates the LLM's marginal contribution by rule type, to replace the paper's weak "full == symbolic" observation with a measured claim: symbolic compression recovers structure on any vocabulary, while the LLM's value is recall of meaning that evaporates on invented predicates. Targets AAAI-27 (CORE A*).
**Related**: builds on EX27 (recursion / Operation I) and EX25b/EX25; this is the post-fold-in research-strengthening pass (direction B: "make the LLM's role real").

## Context

The current paper reports that on the recursion experiment the full pipeline equals the symbolic-only pipeline: Operation I discovers the recursive rule before Operation G is consulted, so the LLM adds nothing. As written this is an aside ("full == symbolic"), and an A* reviewer reads it as the neurosymbolic framing being hollow in the marquee result. EX28 turns this into a deliberate, measured characterization.

The thesis to test: DreamLog's two engines cover disjoint rule types for disjoint reasons. Symbolic MDL compression recovers **structure** (within-predicate generalizations via Operation C, recursive closures via Operation I), and structure is vocabulary-independent: the same operations fire on invented predicates as on real ones. The LLM's contribution is **recall of meaning** (cross-predicate semantic rules like `father = male parent`), which it cannot supply once the predicates are invented and there is no meaning to recall. The sharpest test is the cell where neither engine has an advantage: a cross-predicate rule over fully-invented predicates, recoverable only by genuine inference from the fact distribution, never by recall.

This experiment doubles as the rigorous per-mechanism ablation the paper currently lacks (EX08 ablates operations on a single mixed KB; EX28 ablates by rule type with a clean factorial).

## Research framing

### The claim
Across three rule types, the marginal contribution of the LLM is confined to cross-predicate semantic rules, and even there it depends on recall: on fully-invented predicates the LLM-only condition collapses while symbolic compression is unaffected. The LLM contributes meaning, not structure.

### The decisive cell
Cross-predicate semantic, fully-invented vocabulary. No symbolic operation synthesizes a new cross-predicate body from facts, and there is no semantic to recall, so the only signal is statistical co-occurrence in the fact distribution. If the LLM-only condition fails here (especially at modest model scale), the LLM's cross-predicate value is recall, not reasoning. If it succeeds, the LLM has genuine inference ability on invented relations, which is itself a publishable finding.

### Either outcome ships
The clean expected pattern is symbolic ~100% on the structure rows for both vocabulary columns, and LLM-only ~0% on every fully-invented cell. A non-zero LLM-only-invented cell (structural-prior transfer, or arbitrary-relation inference) is the nuanced alternative. Both are real characterizations.

## Domains: a clean 3x2 factorial

Three rule types crossed with {canonical, fully-invented}. The invented column invents the **predicate names**, not only the entity names, so the only recoverable signal is structural or statistical, never semantic recall. Reuse `flux_reaches`/`ancestor`/`family-father`; build three new domains.

| Rule type | Canonical | Fully-invented |
|---|---|---|
| within-predicate (Op C) | `can_fly(X) :- bird(X), not exc` (birds; penguin/ostrich exceptions) | `glonk(X) :- zorp(X), not exc` (invented) |
| recursive (Op I) | `ancestor = TC(parent)` (reuse) | `flux_reaches = TC(flux_links)` (reuse) |
| cross-predicate semantic (Op G) | `father(X,Y) :- parent(X,Y), male(X)` (reuse) | `wibble(X,Y) :- frob(X,Y), quax(X)` (invented) |

Note on the within-predicate row: Operation C generalizes a single predicate's fact group using a unary guard predicate plus exceptions (its existing `master_artisan(X) :- artisan(X), not exc` pattern), so `can_fly(X) :- bird(X), not exc` is the Op C rule type even though it names a guard. This is distinct from Operation G's multi-literal cross-predicate semantic rules (`father = parent and male`), which is what the cross-predicate row tests.

New domain generators required (parallel to `flux_domain` in `experiments/ex27_recursion.py`): within-predicate canonical (birds), within-predicate invented (glonk/zorp), cross-predicate invented (wibble/frob/quax). Each is deterministic given a seed, generates training facts from the true rule, and exposes the target rule for the proposal probe. The invented domains use disjoint invented vocabularies (no overlap with the flux nodes) so nothing leaks across experiments.

Design constraint for the cross-predicate-invented domain: the underlying rule (`wibble = frob and quax`) must be statistically identifiable from the facts (so the test is fair: there is a real signal), but the predicate names must carry no meaning (so success requires inference, not recall).

## Conditions (four per cell)

For each of the six cells:
- **symbolic-only**: the relevant symbolic operation only (Op C for within-predicate, Op I for recursive, none for cross-predicate), no LLM.
- **LLM-only**: Operation G on, with the rule-type's symbolic operation OFF, so the LLM is the sole route. (Cross-predicate has no symbolic operation, so it is LLM-only by nature.)
- **full**: both.
- **raw-LLM**: the LLM used directly as an inference engine, yes/no on held-out queries (the existing baseline).

The LLM's marginal contribution for a rule type is read as (LLM-only recovery) and (full minus symbolic-only).

## Metrics (two)

1. **Recovery**: fraction of held-out / new-entity derived facts the compressed KB derives (the existing new-entity protocol from EX27/EX25b).
2. **Correct-rule-proposal rate** (the decisive new metric): over N runs, the fraction in which Operation G proposes a rule **structurally equivalent** to the target rule. Structural equivalence is variable-renaming and body-order invariant; reuse the normalization in `dreamlog/skeleton.py` (which already canonicalizes variable names and functor roles) rather than string matching.

Sketch:

```python
def proposal_rate(domain, llm_client, n_runs=10):
    base, derived, target_rule = domain.train_with_target()
    hits = 0
    for i in range(n_runs):
        kb = build_kb(base + derived)
        proposed = run_op_g_proposal(kb, llm_client, seed=i)   # parsed Rule list
        if any(structurally_equivalent(r, target_rule) for r in proposed):
            hits += 1
    return hits / n_runs
```

## Models

- **`qwen2.5:3b`** (local, on the remote Ollama host `192.168.0.204:11434`) for development and the bulk of the runs. This is the model the other Claude Code session has loaded; reusing it avoids reload thrashing on the shared GPU.
- **Anthropic Haiku 4.5** anchors the headline cells (paper consistency: EX25b/EX27 use Haiku) and supports the strong claim ("even a capable model cannot do structure-on-invented").
- **Honesty caveat (bake into the writeup)**: a 3B model failing at structure is weaker evidence than a capable model failing. The 3b numbers characterize the bulk; the Haiku anchor is required before the paper asserts "the LLM cannot do structure."
- Model-scaling sweep (1.5b/7b) is deliberately OUT for now to keep `qwen2.5:3b` warm and avoid GPU eviction. The harness should be model-parameterized so a sweep is a later config change, not a rewrite.

## GPU and reproducibility discipline

- LLM calls run **sequentially in small batches**, never flooding the shared GPU.
- **Only `qwen2.5:3b`** is ever requested on the Ollama host (no other-model loads that would evict it).
- **Fixed per-run seed** and a **defined temperature** (0.3, matching EX25b) so the proposal-rate distribution is reproducible.
- **Generous per-call timeouts** so GPU contention from the other session causes slow calls, never spurious timeout-failures that would corrupt the proposal-rate.
- Timing variance does not affect results: every metric is a correctness measure (recovery, proposal rate), not throughput.

## Implementation surface

| Change | Location | Notes |
|---|---|---|
| 3 new domain generators | `experiments/ex28_llm_role.py` (or a shared domains module) | birds (within-pred canonical), glonk/zorp (within-pred invented), wibble/frob/quax (cross-pred invented); parallel to `flux_domain` |
| Op-C disable flag | `dreamlog/kb_dreamer.py` | a flag to skip Operation C, for the within-predicate LLM-only cell; Op I (`discover_recursion`) and Op G (`llm_client`) are already toggleable |
| Proposal-rate probe | `experiments/ex28_llm_role.py` + reuse `skeleton.py` | run Op G's proposal step in isolation, structural-equivalence check against the target rule |
| Ollama remote-host wiring | `dreamlog/llm_client.py` | confirm `LLMClient` (provider `ollama`) accepts a non-localhost host (`192.168.0.204:11434`) and a model name; the ollama Python client supports `Client(host=...)` |
| Multi-run harness | `experiments/ex28_llm_role.py` | N=10 for the proposal probe; N=1 to 3 for the costly raw-LLM baseline |
| EX28 script + registry | `experiments/ex28_llm_role.py`, `experiment_registry.yaml` | four conditions x six cells; a `--no-llm` cost guard like EX27 |

## Risks and mitigations

1. **`qwen2.5:3b` is weak**: LLM-only conditions may be ~0 across the board, which supports the thesis but on a weak model. Mitigation: Haiku anchor on the headline cells; frame the 3b results as the bulk characterization.
2. **Structural-equivalence false negatives/positives**: a correct rule written with different variable names or body order must count as a hit; a wrong rule must not. Mitigation: reuse `skeleton.py` normalization, with unit tests on known equivalent/non-equivalent pairs.
3. **Cross-predicate-invented domain ill-posed**: if the rule is not statistically identifiable from the facts, the cell tests nothing. Mitigation: validate that an oracle (or symbolic search) could in principle recover it from the facts; keep the relation simple (two-literal body).
4. **Op-C disable flag drift**: must not change default behavior. Mitigation: off by default, full-suite regression like the Op I flag.
5. **GPU coexistence**: covered above (sequential, single model, generous timeouts).

## Testing

- Unit: each domain generator (training facts entail the target rule; invented vocab is disjoint; cross-predicate-invented rule is identifiable). Structural-equivalence checker on equivalent/non-equivalent rule pairs.
- Integration: the four conditions run end to end on one cell with a mock LLM (deterministic), then on `qwen2.5:3b` for one real cell.
- Regression: the Op-C disable flag off-by-default leaves the full suite green.

## Success criteria

1. The clean 3x2 domains are built and validated (target rule identifiable, invented columns semantically empty).
2. EX28 runs all four conditions across the six cells on `qwen2.5:3b`, with the headline cells also on Haiku, producing the recovery table and the proposal-rate table.
3. The structure-vs-semantics pattern is characterized: where symbolic recovers regardless of vocabulary, where the LLM helps, and the cross-predicate-invented cell result.
4. Reproducible (fixed seeds), GPU-coexistent (single warm model), within the LLM budget (cost guard, raw-LLM at low N).
5. Existing tests stay green; new tests cover the generators and the equivalence checker.
