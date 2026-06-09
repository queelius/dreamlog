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

- **`qwen2.5:3b` only, for now** (local, on the remote Ollama host `192.168.0.204:11434`). It is free and unmetered, so EX28 runs entirely on it, and we lean into that: high run counts (N=30 or more for the proposal-rate, so confidence intervals are tight), all six cells, all four conditions, plus exploratory ablations. It is also the model the other Claude Code session has loaded, so reusing it keeps it warm and avoids reload thrashing on the shared GPU.
- **No Anthropic Haiku runs yet** (cost). The Haiku anchor is DEFERRED: it is added only once the local results are characterized and we are ready to support the strong claim ("even a capable model cannot do structure-on-invented vocabulary"). The harness is model-parameterized so swapping in Haiku (or a qwen 1.5b/7b scaling sweep) later is a config change, not a rewrite.
- **Honesty caveat (bake into the writeup)**: a 3B model failing at structure is weaker evidence than a capable model failing. The qwen2.5:3b numbers characterize the bulk and drive iteration; the Haiku anchor is required before the paper asserts "the LLM cannot do structure."
- **Because the local model is free, lean into the ablation**: sweep domain size, exception rates, fact density, recursion depth, and number of distractor predicates, and report the full curves. This is exactly the thorough ablation an A* paper wants and the API budget previously discouraged. Keep all of it on `qwen2.5:3b` (never request another model, to keep it warm and avoid GPU eviction).

## GPU and reproducibility discipline

- LLM calls run **sequentially in small batches**, never flooding the shared GPU.
- **Only `qwen2.5:3b`** is ever requested on the Ollama host (no other-model loads that would evict it).
- **Fixed per-run seed** and a **defined temperature** (0.3, matching EX25b) so the proposal-rate distribution is reproducible.
- **Generous per-call timeouts** so GPU contention from the other session causes slow calls, never spurious timeout-failures that would corrupt the proposal-rate.
- Timing variance does not affect results: every metric is a correctness measure (recovery, proposal rate), not throughput.

## Metadata and resumability (first-class requirement)

Every EX28 run must be richly documented and resumable, because the local model is slow on a shared GPU and the experiment is large (six cells x four conditions x high N), so runs will be long and may be interrupted.

**Work unit.** The atomic unit is `(cell, condition, run_index)`. The harness enumerates all units up front, checks the results store for already-completed units (keyed by a stable hash of the unit plus the resolved run configuration), and runs only the missing ones. After each unit completes, it appends the record and flushes to disk, so an interruption loses at most the in-flight unit.

**What is recorded per unit** (one JSONL record): the unit key (cell, condition, run_index); the resolved configuration (model, host, quantization, temperature, seed, domain parameters, n_runs); both metrics (recovery, and for the proposal probe the per-run proposed rules plus the structural-equivalence verdict against the target rule); and the **full LLM call log** for the unit (the exact prompt sent, the raw model response, the parsed rules, per-call latency, and token counts if the provider returns them). A wall-clock timestamp and the git commit SHA are recorded so each result is reproducible from a known code state.

**Run manifest.** Each invocation writes a manifest (start time, git SHA, the full argument set, model and host, and the list of planned vs already-complete units) so a run can be audited and resumed.

**Storage.** `experiments/data/ex28/results.jsonl` (the per-unit records, the resumable store, committed for provenance) plus `experiments/data/ex28/manifest-<timestamp>.json`. The JSONL is **append-only and never rewritten**, so a crash mid-write cannot corrupt completed records.

**Resumability contract.** Re-running with the same configuration is idempotent: completed units are skipped, missing units are run, and the final summary is recomputed from the full store. A `--fresh` flag forces a clean re-run. A `--summarize` flag recomputes the tables from `results.jsonl` without running anything.

## Implementation surface

| Change | Location | Notes |
|---|---|---|
| 3 new domain generators | `experiments/ex28_llm_role.py` (or a shared domains module) | birds (within-pred canonical), glonk/zorp (within-pred invented), wibble/frob/quax (cross-pred invented); parallel to `flux_domain` |
| Op-C disable flag | `dreamlog/kb_dreamer.py` | a flag to skip Operation C, for the within-predicate LLM-only cell; Op I (`discover_recursion`) and Op G (`llm_client`) are already toggleable |
| Proposal-rate probe | `experiments/ex28_llm_role.py` + reuse `skeleton.py` | run Op G's proposal step in isolation, structural-equivalence check against the target rule |
| Ollama remote-host wiring | `dreamlog/llm_client.py` | confirm `LLMClient` (provider `ollama`) accepts a non-localhost host (`192.168.0.204:11434`) and a model name; the ollama Python client supports `Client(host=...)` |
| Resumable, metadata-rich harness | `experiments/ex28_llm_role.py` | work-unit enumeration (cell x condition x run), skip-completed, append-only `results.jsonl` with full per-call logs (prompt, response, latency), run manifest, `--fresh` and `--summarize` flags. N=30+ for the proposal probe and the raw-LLM baseline (both free on the local model) |
| EX28 script + registry | `experiments/ex28_llm_role.py`, `experiment_registry.yaml` | four conditions x six cells; `qwen2.5:3b` only by default |

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
2. EX28 runs all four conditions across the six cells on `qwen2.5:3b` (no Haiku yet) at high N, producing the recovery table and the proposal-rate table with confidence intervals.
3. The structure-vs-semantics pattern is characterized: where symbolic recovers regardless of vocabulary, where the LLM helps, and the cross-predicate-invented cell result. Exploratory ablations (domain size, exception rate, fact density, recursion depth) are run and reported, since the local model is free.
4. Every run is **resumable** (re-running skips completed units; an interruption loses at most the in-flight unit) and records **full metadata** per unit (config, both metrics, the exact prompt/response/latency, and the git SHA).
5. Reproducible (fixed seeds), GPU-coexistent (single warm model, sequential calls, generous timeouts).
6. Existing tests stay green; new tests cover the generators, the equivalence checker, and the resume/skip logic.
