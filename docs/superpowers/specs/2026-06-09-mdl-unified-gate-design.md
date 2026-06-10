# MDL Unified Gate: Design (Phases 1+2)

**Date:** 2026-06-09
**Status:** Approved direction (user, 2026-06-09); spec pending user review
**Branch:** `mdl-unified-gate` (off checkpoint `v0.13.0`, commit `be831fe`)
**Scope:** Phases 1 and 2 of the four-phase MDL roadmap from the 2026-06-09 MDL review. Phase 3 (bits-based description length, error-priced open world) and Phase 4 (delta-DL agenda scheduler) are explicitly out of scope.

## 0. Decision record

The 2026-06-09 MDL review audited the nine sleep operations against the MDL principle and found: (a) the accept/verify/rollback machinery is re-implemented per operation; (b) A+B and D+E are each one principle implemented twice; (c) F and H are cache policies, not compression, and make the dream phase's objective incoherent; (d) description length (clause count) has no single owner in the code. The user approved the staged path and chose "Spec P1+P2 refactor" as the first step: unify the machinery now, keep decisions bit-identical, defer semantic changes (bit-cost DL, error pricing, scheduling) to P3/P4.

## 1. Motivation

- `dreamlog/kb_dreamer.py` exceeds 1,550 lines: orchestration, nine operations, verification suite, LLM prompt/parse, and scoring in one module. The Task-5 `_llm_propose` extraction (commit `7e61880`) was delicate precisely because proposal logic and acceptance logic are interleaved; this spec makes that separation total.
- F (dead-clause pruning) and H (lemma caching) are not compression: F is usage-based forgetting of non-seed clauses (lossy) and H adds clauses for speed (anti-MDL). In practice F evicts what H added in earlier cycles: they are a cache-management pair and belong outside the compression objective.
- A (subsumption elimination) and B (redundant-fact pruning) are both "delete a clause implied by the remainder". D (predicate invention) and E (body-pattern extraction) are both "factor shared structure into a new definition".
- The pipeline order is a fixed heuristic for the MDL objective. EX28 surfaced an order artifact (Operation C removing facts starves Operation G's prompt). Phase 4 addresses scheduling; this spec makes the schedule an explicit, single list as a prerequisite.

## 2. Goals

- **G1.** One implementation of trial-apply / verify / rollback / record (the gate), used by every compression operation.
- **G2.** Operations become pure proposal generators: read-only over the KB, yielding `Proposal` values.
- **G3.** Single source of truth for description length (`dl.py`); clause count in P1.
- **G4.** Merge A+B into `reduce`, D+E into `factor`; relocate F and H to `maintenance`.
- **G5.** Bit-identical compression decisions: every accept/reject decision made today is reproduced exactly. Policies are lifted verbatim from the current operation implementations.
- **G6.** Experiment-facing surface unchanged: `KnowledgeBaseDreamer(...)` constructor parameters and defaults, `dream()` signature and `DreamSession` shape, `dream_kb(...)` in the experiment helpers.

## 3. Non-goals (explicit)

- No bits-based DL. `dl.py` ships clause-count internals behind stable signatures; P3 swaps the internals.
- No error-priced open world. `open_world` remains a binary toggle on Operation G's false-positive check.
- No agenda or best-first scheduling. The fixed order is preserved exactly.
- No new operations (Plotkin clause reduction, absorption/identification, extensional-alias merge, additional fixpoint schemas: all backlog).
- No strengthening of `reduce` beyond today's exact A and B checks. In particular, derivability-based removal of RULES (general "delete rule r if KB minus r entails r") is backlog; P2 `reduce` is the union of today's checks, nothing more.
- No engine-level relocation of F/H to the wake phase. `maintenance` functions are still invoked from `dream()` at today's positions; moving the call sites is backlog.

## 4. Current state (verified inventory, 2026-06-09)

`dream()` pipeline (`dreamlog/kb_dreamer.py:316-400`, read directly):

1. Snapshot `kb.copy()`; save wake-phase usage maps (`_usage_counts`, `_derivation_counts`, `_derivation_terms`); build `VerificationSuite` if `verify`.
2. **F first**: `_prune_dead_clauses(kb, seed_terms=..., seed_rules=...)` (entry ~line 1488) using wake-phase usage only; seed facts/rules (present at dream start) are never pruned; requires `total_queries_tracked() >= 10` and >= 50% predicate coverage. `dream()` then prunes dead facts out of `suite.positive_queries` inline (lines 341-348): this coupling moves into maintenance.
3. **A**: `_eliminate_subsumed(kb)`: rules subsumed by more general rules (same-body-length `clause_subsumes`), facts subsumed by bodyless rules.
4. **B**: `_prune_redundant_facts(kb)`: facts derivable from rules + other facts; batch removal with one-at-a-time fallback for mutual dependencies.
5. **C**: `_generalize_facts(kb, suite)` (entry ~line 494), gated by `disable_op_c`.
6. `extend_verification_for_rules(suite, kb)`.
7. **D**: `_invent_predicates(kb, suite)`; **E**: `_extract_body_patterns(kb, suite)`.
8. **I**: `_discover_recursion(kb, suite)` (entry ~line 825), gated by `discover_recursion`.
9. `_name_invented_predicates(kb)` (LLM naming; not a compression op).
10. **G**: `_llm_compress(kb, suite)` (entry ~line 1285), built on `_build_op_g_prompt` (~1139) and `_llm_propose` (~1217, returns parsed + structurally validated + cycle-filtered rules). Acceptance: helper/main split via `existing_functors` = fact functors union rule-head functors; `MAX_CALLS = max(500, len(kb) * 10)`; per main rule: test-KB copy, >= 2 derivable facts, suite verify, false-positive enumeration unless `open_world`; then Phase 5 combined-set verify, rejecting ALL accepted rules if it fails.
11. **B again** (bounded, `max_calls=500`) only if G accepted anything.
12. **H**: `_cache_lemmas(kb)` (uses `_frequency_score`).
13. Final global `suite.verify(kb, ...)` with `max_total_calls = 500 if llm_ops else 0`; full `kb.restore_from(snapshot)` on failure; restore wake usage maps; return `DreamSession(compressed, operations, compression_ratio, verification)`.

Module-level helpers: `_strip_llm_noise` (51), `_filter_cyclic_rules` (63), `_collect_user_functors` (114), `_is_system_predicate`, `_frequency_score` (~1548).

Constructor (`__init__`, lines 286-314): `llm_client`, `min_group_size=3`, `shared_structure_threshold=0.1`, `max_prompt_facts=50`, `open_world=False`, `discover_recursion=False`, `min_base_facts=3`, `disable_op_c=False`.

Two batch-accept semantics exist today and must survive:
- B: remove all derivable facts at once; if verification of the batch fails, fall back to one-at-a-time.
- G Phase 5: accept rules individually, then verify the combined set; on combined failure, reject all.

## 5. Target architecture

```
dreamlog/compression/
  __init__.py
  dl.py            description_length(kb), clause_cost(clause), proposal_delta(p)
  proposal.py      Proposal (frozen): kind, remove, add, notes
  gate.py          apply_proposal(...), apply_batch(...); GateResult; rejection reasons
  policies.py      per-kind acceptance policies (lifted verbatim from current ops)
  util.py          _collect_user_functors, _filter_cyclic_rules, _strip_llm_noise,
                   _is_system_predicate (shared helpers move here)
  generators/
    __init__.py
    reduce.py      A+B detectors -> removal proposals
    generalize.py  C -> rule + not/1 exceptions proposals
    factor.py      D and E detectors -> introduce-definition-and-rewrite proposals
    closure.py     I -> base + right-recursive pair proposals
    llm.py         G proposal stage (_build_op_g_prompt, _llm_propose) -> add-only proposals
  maintenance.py   evict_dead_clauses (F, including the suite pruning), cache_lemmas (H)
```

`kb_dreamer.py` keeps: `KnowledgeBaseDreamer` facade (same constructor), `dream()` orchestration (its call sequence through the existing private method names, which become thin orchestrators), `DreamSession`, `CompressionCandidate`, the verification-suite code (`build_verification_suite`, `extend_verification_for_rules`, `VerificationSuite`), and `_name_invented_predicates` with `_rename_predicate`. Target size after the moves: under ~550 lines (revised from ~450: the thin orchestrator methods stay for monkeypatch and test compatibility).

### 5.1 Proposal

```python
@dataclass(frozen=True)
class Proposal:
    kind: str                                   # reduce | generalize | factor | closure | llm
    remove: Tuple[Union[Fact, Rule], ...]
    add: Tuple[Union[Fact, Rule], ...]
    notes: Mapping[str, Any]                    # provenance: detector, guard, witness, helper flag...
```

Generators MUST NOT mutate the KB. A gate property test enforces it (the KB's fact and rule multisets are unchanged after `propose()` is exhausted).

### 5.2 dl.py

`description_length(kb) -> int` returns `len(kb)` in P1. `clause_cost(clause) -> int` returns 1. `proposal_delta(p) -> int` returns `len(p.add) - len(p.remove)`. P3 replaces the internals (bit costs, functor-signature charge); the signatures are stable now so call sites never change again.

### 5.3 gate.py

- `apply_proposal(kb, proposal, policy) -> Accepted(candidate) | Rejected(reason)` (as-built: the suite lives IN the policy, set at construction, not passed per call). Mechanics: delta check (when the policy demands it) -> `policy.pre_check(kb, proposal)` -> trial = `kb.copy()` -> apply remove/add to trial -> `policy.verify(trial, proposal)` -> on pass, apply to the real KB and return `Accepted(CompressionCandidate(...))`; otherwise `Rejected(reason)` with the real KB untouched.
- `apply_batch(kb, proposals, suite, policy) -> (accepted, rejected)` with two modes covering today's two batch semantics: `fallback` (try all at once; on failure retry one-at-a-time: B's behavior) and `all_or_nothing_finalize` (accept individually, then verify the combined result; on failure undo all: G Phase 5's behavior).
- `RecursionError` inside any verify maps to `Rejected("budget")`, matching today's catch-and-skip semantics in G and the closure/redundancy paths.
- Rejection reasons are an enum: `delta | verify_failed | fp_check | budget | policy`. `DreamSession` gains an ADDITIVE `rejections` list (default empty) recording `(kind, reason, summary)`; no existing consumer changes.

### 5.4 Policies (decisions lifted verbatim)

The implementation plan lifts each accept condition by reading the current op and moving its logic unchanged:

- **reduce**: witness check only. Fact case: derivable from the remainder (B), or subsumed by a bodyless rule (A). Rule case: subsumed by a more general same-body-length rule (A, via existing `clause_subsumes`). No per-proposal suite run (matches today; the final global verify is the backstop). Batch mode: `fallback`. Internal detector order preserved: the subsumption pass (A) runs to completion before the derivability pass (B), exactly as the two ops run sequentially today. The bounded post-LLM re-pass is the same generator invoked with `max_calls=500`, derivability detector only (today's second pass is B alone).
- **generalize**: `min_group_size`, guard discovery, exception computation, the existing count criterion (rule + exception clauses fewer than facts removed), suite verification as today. Emitted clause shapes must be byte-identical to today's (guarded rule with `not/1` plus exception facts).
- **factor**: D's skeleton detector and E's body-subsequence detector kept as two detectors behind one generator; their interface-variable computations are NOT unified (they differ; unifying is backlog). Accept conditions and `shared_structure_threshold` usage unchanged; suite verification as today.
- **closure**: exact-match transitive-closure gate, `min_base_facts`, right-recursive pair synthesis, suite verification as today.
- **llm**: structural validation (today's `_llm_propose` Phases 1-2) stays in the generator. The policy battery: helper/main split via `existing_functors` (fact functors union rule-head functors), `MAX_CALLS = max(500, len(kb) * 10)`, >= 2 derivable existing facts, per-rule suite verify, false-positive enumeration unless `open_world`. Finalization: `apply_batch` in `all_or_nothing_finalize` mode (Phase 5).

**The delta exception (do not "fix" this):** llm proposals are add-only, so `proposal_delta` is positive. Compression is realized by the scheduled post-llm `reduce` pass that removes the facts the new rule derives. Therefore the llm policy does NOT require `delta < 0` in P1; its acceptance battery is the criterion. P3 prices this properly as cost(rule) minus cost(facts made removable). The `reduce`, `generalize`, and `closure` policies DO require `delta < 0`: their proposals are strictly net-removing by construction today. For `factor`, the 2026-06-09 source read SETTLED the open question: the D detector enforces strict reduction explicitly (`if k + n >= n * k: continue`, kb_dreamer.py:684), but the E detector ALWAYS has delta = +1 (it adds the extracted rule and rewrites each occurrence 1:1; kb_dreamer.py:823-826). E's value exists only under a symbol-cost description length, which is direct motivation for P3. In P1 the factor policy enforces strict delta for D proposals and exempts E proposals. G5 outranks delta purity everywhere.

**Two generator forms (settled by the source read):** C, D, and E interleave detection with application: C rebuilds its fact list after each accepted subgroup and its group snapshot deliberately excludes the exception predicates it introduces (kb_dreamer.py:610-624); D allocates invented names from the live KB before verification (line 687); E is round-based over the live KB with a failed-pattern memo (lines 762-767). A propose-all-then-gate pipeline would change candidate discovery. Therefore generators come in two forms: pure `propose(kb) -> Iterator[Proposal]` for closure and llm (and reduce's detection, which probes on a scratch copy because B's detection mutates the KB today), and `run(kb, gate_fn) -> List[CompressionCandidate]` for reduce, generalize, and factor, where the generator owns today's exact iteration order and calls the gate per candidate. The invariant that matters: generators NEVER mutate the KB directly; every mutation flows through the gate's commit step. The gate replaces, per op, exactly today's "copy, verify, continue-on-fail, apply, append" block, which the source read located in each op (C: 601-616, D: 732-750, E: 787-826, I: 884-900).

**Facade dispatch (settled by import analysis):** `experiments/ablation_and_scale.py` monkeypatches `KnowledgeBaseDreamer._llm_compress`; experiments call `_generalize_facts` and `_prune_dead_clauses`; tests call `_llm_propose`, `_discover_recursion`, `_prune_dead_clauses`, `_prune_redundant_facts`, `_build_op_g_prompt`. Therefore `dream()` keeps dispatching through the existing private method names, and those methods become thin orchestrators (build policy, run generator through gate). This preserves monkeypatchability and AC1/AC5 without compatibility shims: the method names ARE the schedule.

### 5.5 maintenance.py

- `evict_dead_clauses(kb, suite, min_query_threshold=10, seed_terms=..., seed_rules=...)`: F verbatim, including the >= 50% predicate-coverage requirement, seed protection, and the `suite.positive_queries` pruning currently inlined in `dream()` (moves here so the F/suite coupling lives in one place).
- `cache_lemmas(kb)`: H verbatim, with `_frequency_score`.
- Both return `CompressionCandidate` lists under their current operation names so `session.operations`, the TUI `/dream-status` output, and the experiment scripts see identical shapes. They do not pass through the gate and are exempt from the delta rule by construction.

### 5.6 Facade and schedule

`dream()` becomes: snapshot / usage save / suite build -> `maintenance.evict_dead_clauses` -> the schedule below, each step driving generator output through the gate -> `maintenance.cache_lemmas` -> final global verify with rollback -> usage restore -> `DreamSession`.

```python
SCHEDULE = [reduce,                      # A+B
            generalize,                  # C   (skipped when disable_op_c)
            # extend_verification_for_rules happens here, as today
            factor,                      # D then E detectors, in that order
            closure,                     # I   (only when discover_recursion)
            # _name_invented_predicates happens here, as today
            llm,                         # G   (only when llm_client)
            reduce_bounded]              # B re-pass, only if llm accepted anything
```

The order is a single literal in one place, identical to today's verified order.

## 6. Behavior-preservation contract (acceptance criteria)

- **AC1.** Full pytest suite green; every existing test in `tests/test_sleep_cycle.py` passes UNCHANGED (they exercise `dream()` behaviorally).
- **AC2.** Mock-LLM Operation G tests pass unchanged: `test_op_g_accepts_a_recursive_proposal`, `test_full_pipeline_op_i_then_op_g_no_duplicate_rules`, `test_full_pipeline_rejects_wrong_llm_rule_for_compressed_predicate`, `test_llm_propose_returns_parsed_rules_without_accepting`, `test_llm_propose_preserves_phase1_validation`.
- **AC3.** Deterministic symbolic regression at zero LLM cost, compared against the committed artifacts:
  - EX25b crafting symbolic: recall 0.526, precision 0.588, 5 rules, compression 0.919 (`experiments/data/ex25b/results.json`).
  - EX27 symbolic, both domains: recall 1.00, precision 1.00, 2 rules, ratio 0.278 (registry EX27).
  - EX28 symbolic column, all six cells: within 1.00 recall / 0.67 precision, recursive 1.00 / 1.00, cross 0.00, identical across vocabularies (`experiments/data/ex28*/results.jsonl`).
  Implemented as `tests/test_mdl_refactor_regression.py` (marked `slow`), re-running the symbolic cells and comparing exactly.
- **AC4.** Gate property tests: (a) generator detection never mutates the KB (pure `propose()` leaves it untouched; `run()` with an always-reject gate leaves it structurally identical); (b) a rejected proposal leaves the KB structurally identical (equal fact and rule multisets); (c) every accepted proposal preserves suite closure; (d) `delta >= 0` is never accepted for the strict kinds: reduce, generalize, closure, and factor's D (invention) detector. Factor's E (extraction) detector is exempt: its delta is +1 by design in clause-count terms (see Section 5.4).
- **AC5.** No public-surface change: import sites in `experiments/`, `integrations/`, and `tests/` resolve without edits (except the new regression/property test files).

## 7. Risks and mitigations

- **R1. Batch semantics drift** (B's fallback, G's Phase-5 reject-all): covered by `apply_batch`'s two explicit modes; AC2/AC3 catch any drift empirically.
- **R2. F/suite coupling**: the `positive_queries` pruning moves with F into maintenance; existing F tests plus AC1 cover it.
- **R3. C's emitted clause shapes** (guarded rule + `not/1` + exception facts) must be byte-identical; the generator emits the same constructions the op builds today, and AC3's EX25b symbolic figures are sensitive to any change.
- **R4. `kb.copy()` per proposal cost**: today G already copies per main rule; A/B/C/D/E currently mutate in place, so the gate adds copies on those paths. Test/experiment KBs are <= ~300 clauses. Measure the suite wall time (currently ~6-9 s); if it degrades more than ~2x, the plan may add an in-place-with-undo fast path inside the gate WITHOUT changing semantics.
- **R5. Ordering fidelity**: the schedule literal must match the verified order in Section 4; AC3 is the end-to-end check.

## 8. File structure

| File | Responsibility | Origin |
|---|---|---|
| `dreamlog/compression/dl.py` | description length, clause cost, proposal delta | new |
| `dreamlog/compression/proposal.py` | `Proposal` dataclass | new |
| `dreamlog/compression/gate.py` | apply_proposal, apply_batch, GateResult, reasons | new (machinery lifted from ops) |
| `dreamlog/compression/policies.py` | per-kind acceptance, lifted verbatim | from kb_dreamer ops |
| `dreamlog/compression/util.py` | shared helpers | from kb_dreamer module level |
| `dreamlog/compression/generators/reduce.py` | A+B detection | from `_eliminate_subsumed`, `_prune_redundant_facts` |
| `dreamlog/compression/generators/generalize.py` | C detection/construction | from `_generalize_facts` |
| `dreamlog/compression/generators/factor.py` | D+E detection/construction | from `_invent_predicates`, `_extract_body_patterns` |
| `dreamlog/compression/generators/closure.py` | I detection/construction | from `_discover_recursion` |
| `dreamlog/compression/generators/llm.py` | G proposal stage | from `_build_op_g_prompt`, `_llm_propose` |
| `dreamlog/compression/maintenance.py` | F + H + F's suite pruning | from `_prune_dead_clauses`, `_cache_lemmas`, dream() inline |
| `dreamlog/kb_dreamer.py` | facade, schedule, verification suite, naming | shrank to 555 lines (as-built) |
| `tests/test_mdl_refactor_regression.py` | AC3 symbolic regression vs committed artifacts | new |
| `tests/test_compression_gate.py` | AC4 property tests | new |

## 9. Backlog registered by this spec (not in P1/P2)

- P3: bits-based DL with functor-signature cost; error-priced open world (cost of rule plus cost of corrections); collapse per-kind policies into the single priced rule.
- P4: delta-DL agenda scheduler (also fixes the C-starves-G ordering artifact EX28 exposed).
- New generators: Plotkin clause reduction; absorption/identification; extensional-alias merge; additional fixpoint schemas; subset-match open-world closure gate (EX27c).
- Reduce upgrade: general "delete rule implied by remainder".
- Engine-level wake-phase relocation of maintenance.
- Paper: once P3 lands, the formal story ("operations are proposal generators into a single MDL gate") strengthens the Solomonoff section.
