# P3: Bits-Based Description Length (flag-gated) - Design

**Date:** 2026-06-10 (approved 2026-06-11)
**Status:** Approved direction; spec pending user review
**Branch:** `p3-bits-dl` (off master `5e104ff`)
**Builds on:** the MDL unified gate (P1+P2, spec `2026-06-09-mdl-unified-gate-design.md`, merged at `e758811`). P1 deliberately shipped clause-count DL behind stable signatures so this phase swaps the internals.

## 0. Decision record

User choices (2026-06-10/11): encoding = **prefix code over symbol tables** (vs weighted symbol count, vs empirical entropy); staging = **flag-gated with a decision-diff report** (default stays clause count; flipping the default is a separate, user-approved follow-up); error pricing = parameter-free correction clauses. The motivating fact from the P1 source read: Operation E is ALWAYS delta = +1 in clause count; only a symbol-cost code can justify (or correctly veto) extraction.

## 1. Goals

- **G1.** One bits-based code in `dl.py`, deterministic given KB state, exact and documented enough to hand-compute small examples.
- **G2.** Flag-gated: `KnowledgeBaseDreamer(dl_mode="clauses")` default preserves bit-identical current behavior (the existing regression test and full suite are untouched and stay green).
- **G3.** In `dl_mode="bits"`, the per-kind delta exemptions collapse toward the principle: E must earn its extraction in bits; G's "derives >= 2 facts" proxy becomes the priced criterion bits(rule) + bits(corrections) < bits(facts made removable).
- **G4.** A decision-diff tool replays the EX symbolic cells and the benchmark scenarios, scores every proposal under BOTH encodings, and emits a flip report for joint review.
- **G5.** Error pricing in open world is parameter-free: each over-derivation costs its correction clause under the same code.

## 2. Non-goals

- No default flip (that is P3b, after the diff review, with EX re-runs as new ground truth).
- No policy-hierarchy collapse into a single priced rule (after the flip).
- No P4 agenda scheduling.
- No change to suite verification, rollback, or any witness check: bits mode changes only the DELTA criterion and the G/E acceptance arithmetic.
- No change to Operation C's INTERNAL count heuristic (`cost_after >= cost_before`) in this phase, even in bits mode: it remains a detection-side prefilter. The diff report will show whether it ever disagrees with the bits gate (vetoing a bits-profitable group, or admitting a bits-unprofitable one that the gate then rejects); revisiting it is a P3b item informed by that data.

## 3. The code (pinned exactly)

Two-part code: DL(kb) = L(dictionary) + sum over clauses of L(clause | dictionary).

**Dictionaries** (derived from the KB on demand):
- F = the set of (functor_name, arity) pairs occurring anywhere (facts, rule heads, rule bodies, nested terms). `not` and `call` enter F like any other functor once used.
- C = the set of distinct constant values occurring anywhere.

**Dictionary cost:** L(D) = sum over entries of (8 * len(str(symbol_name)) + 8) bits (8 bits per character plus a terminator byte). For F entries the name is the functor name (arity is implied by the entry, so no separate arity payload). This is the one-time signature charge: a proposal whose clauses introduce a NEW functor or constant pays its name into the dictionary; a proposal that removes the LAST occurrence of a symbol gets the name cost credited back.

**Clause cost** L(clause | D):
- 2 bits clause-type tag (fact vs rule) + 2 bits terminator.
- Every symbol occurrence in the clause pays: 2 bits (type tag: functor / constant / variable) + payload:
  - functor occurrence: log2(max(1, |F|))
  - constant occurrence: log2(max(1, |C|))
  - variable occurrence: log2(max(1, |V_clause|)) where V_clause = distinct variables in THIS clause.
- Because F is keyed by (name, arity), arities are implied and the code is prefix-decodable without end-of-args markers; rule body length is implied by the terminator tag.

**Delta:** proposal_delta_bits(kb, p) = DL(kb after p) - DL(kb before p), computed EXACTLY: build (F, C) tables and the clause-cost sum for the before state, adjust tables for p.remove/p.add (including dictionary growth and last-occurrence shrinkage), and recompute. Note the global coupling: changing |F| or |C| by one re-prices every functor/constant occurrence in the KB (log2(n+1) vs log2(n)); the exact computation embraces this rather than approximating. Cost is O(KB symbols) per proposal; acceptable at current scales (<= ~400 clauses) and bounded by the benchmark check (Section 7).

**API:** `dl.py` keeps `clause_cost(clause, kb=None, mode="clauses")`, `description_length(kb, mode="clauses")`, `proposal_delta(p, kb=None, mode="clauses")`. Clauses mode ignores `kb` and reproduces P1 exactly (existing call sites pass no kb and are untouched). Bits mode requires `kb`. A small internal `_SymbolTables` helper builds and adjusts (F, C) and is the single place the encoding lives.

## 4. Mode plumbing

- `KnowledgeBaseDreamer.__init__(..., dl_mode: str = "clauses")`, stored, threaded to every policy at construction (policies already carry suite/budgets; they gain `dl_mode` and the kb-context needed for bits deltas).
- The gate's delta check becomes mode-aware: clauses mode = today's `proposal_delta(p) >= 0` reject; bits mode = `proposal_delta(p, kb=kb, mode="bits") >= 0` reject. The gate signature does not change (the policy exposes the mode and the gate already receives kb).
- Per-kind behavior in bits mode:
  - reduce / generalize / closure: unchanged witnesses and suite checks; the delta check now measures bits (removals always price negative; C's rule+exceptions trade and I's closure trade get honest margins).
  - factor: D unchanged except the honest invented-functor dictionary charge now appears in its delta. **E loses its exemption**: `ExtractionPolicy.require_negative_delta` is True in bits mode (False in clauses mode, preserving today's behavior).
  - llm (G): in bits mode the per-item criterion REPLACES the ">= 2 derivable facts" count with: bits(rule, incl. any dictionary growth) + bits(corrections) < sum of bits of the existing facts the rule derives (those facts become removable by the scheduled post-llm reduce pass). Corrections: in CLOSED world the FP check stays an infinite price (any over-derivation rejects, unchanged); in OPEN world each enumerated over-derivation is priced as the correction clause that would exclude it (a fact of a fresh exception functor applied via not/1, priced under the same code; the enumeration reuses the existing FP-check machinery instead of being skipped). Suite verification and the staged-combined batch mechanics are unchanged in both modes.

## 5. Decision-diff tool (the P3 deliverable for review)

`experiments/dl_decision_diff.py`:
- Scenarios: EX25b crafting symbolic, the six EX28 symbolic cells, and the 8 benchmark scenarios from `benchmarks/sleep_cycle_bench.py` (reusing their KB builders).
- Mechanism: an optional, additive `recorder` hook (callable, default None) rides on the POLICY object (`policy.recorder`), so no gate signature changes; the dreamer sets it on every policy it constructs when built with `KnowledgeBaseDreamer(decision_recorder=...)`. The gate invokes it for every proposal that reaches the delta/verify stage with: (kind, remove/add summary strings, delta_clauses, delta_bits, decision). In bits-mode G, the existing Phase-4 derivable-facts loop is retained to OBTAIN the derivable set; only the acceptance arithmetic on that set changes. The tool runs each scenario TWICE (dl_mode="clauses" = live behavior; dl_mode="bits") so both counterfactual scores AND cascade effects (different accepts changing later proposals) are captured.
- Output: `experiments/data/p3_decision_diff/report.md` (human review: a table per scenario with FLIPPED rows highlighted, plus totals) and `decisions.jsonl` (full records, append-only, git_sha-stamped like EX28's harness conventions).
- The report is the artifact we review together before any default flip.

## 6. File structure

| File | Change |
|---|---|
| `dreamlog/compression/dl.py` | bits mode + `_SymbolTables` (the encoding's single home) |
| `dreamlog/compression/gate.py` | mode-aware delta check; optional recorder hook (additive) |
| `dreamlog/compression/policies.py` | `dl_mode` threading; E's conditional exemption; G's priced criterion in bits mode |
| `dreamlog/kb_dreamer.py` | `dl_mode` + `decision_recorder` ctor params (defaults preserve behavior) |
| `experiments/dl_decision_diff.py` | the replay + report tool |
| `tests/test_dl_bits.py` | hand-computed encoding examples + mode-gating tests |
| `tests/test_mdl_refactor_regression.py` | UNTOUCHED (default-mode zero-drift judge) |

## 7. Acceptance criteria

- **AC1.** Default mode zero drift: full suite (881 passed / 11 skipped / 3 deselected at branch time) and the committed-artifact regression pass UNTOUCHED with all new params at defaults.
- **AC2.** Hand-computed unit tests for the code: a ground fact; a 2-goal rule with shared variables; D's invented-functor dictionary charge; E's worked example in both directions (an extraction that pays for itself in bits and one that does not); C's rule+exceptions vs facts margin; I's closure trade; last-occurrence dictionary credit. Each asserts exact bit values from the Section 3 formulas.
- **AC3.** Bits-mode dreams run green end to end on the EX KB builders (suite verification still passes; decisions may differ, and that is the point).
- **AC4.** The diff tool produces the report on all scenarios; the report explicitly lists every flipped decision with its bit arithmetic.
- **AC5.** Performance: clauses-mode benchmark unchanged (same op outputs as the regenerated baseline); bits-mode benchmark completes within 5x of clauses-mode wall time (the O(KB)-per-delta cost is acknowledged; if it exceeds 5x, memoize table sums, never approximate the code).

## 8. Risks

- **R1. Table-shrink credit complexity** (removing a symbol's last occurrence re-prices everything): handled by exact recomputation; AC2's credit test pins it.
- **R2. Open-world G enumeration cost** (pricing corrections requires enumerating over-derivations that the open-world path currently skips): bounded by the same MAX_CALLS evaluator budget as the closed-world FP check; RecursionError during enumeration rejects the item (budget), consistent with gate semantics.
- **R3. C's internal heuristic vs the bits gate** can disagree in bits mode (Section 2 non-goal): surfaced by the diff report, deferred to P3b by design.
- **R4. log2(1) = 0 makes symbols in singleton tables free**: information-theoretically correct (no choice, no information) and pinned in AC2's examples so nobody "fixes" it casually.
