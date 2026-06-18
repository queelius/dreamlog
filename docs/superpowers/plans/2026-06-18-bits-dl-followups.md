# Plan: Bits-DL follow-ups (context effect, rigor, Op I open-world, paper)

**Date:** 2026-06-18
**Context:** P3/P3b shipped the rename-invariant bits-DL; EX29-31 characterized the
abstraction-maturity threshold and the flip evaluation; EX32-35 added the noise
filter, the predictive-thesis correlation, cross-domain transfer, and the Op I
exact-match cliff. Two of those (EX32, EX33) surfaced the same confound: the bits
delta is GLOBAL, so earlier accepts amortize shared declarations and discount
later proposals. This plan resolves the loose threads and consolidates the work.

**Documentation discipline (applies to every experiment below):** all new
experiments use `experiments/_harness.py` (`experiment_run(exp_id=...)`), one
directory per experiment under `experiments/data/<id>/`, with the full meta.json
envelope (git, env, package versions, script sha256, params, seeds, output
hashes) + per-run history + latest.json. Every experiment also gets an
`experiment_registry.yaml` entry. No library drift in experiment-only work
(suite stays 902/11/3). Voice constraint: ASCII only (no U+2014, no the banned
l-word).

---

## Phase 1: Verify rigor + characterize the global-DL context effect

The EX32 ordering confound and the EX33 subfamily-C contamination are the same
phenomenon. Phase 1 both audits the existing claims and turns the confound into a
characterized property.

- [ ] **T1 -- Independent verification of EX32 + EX33.** A review (no code change)
  that: re-derives EX32's headline (precision 1.00 bits vs 0.17 clauses) and
  confirms the ordering dependence is exactly as documented (not worse);
  re-derives EX33's correlation range and confirms the recoverable-holdout
  protocol genuinely recovers (non-zero, varying) rather than the
  counterexample-exception trap; confirms the subfamily-C confound bounds the
  claim correctly. Output: a short verification note committed under
  `docs/superpowers/reviews/2026-06-18-ex32-ex33-verification.md`.
- [ ] **T2 -- EX36: the global-DL context effect.** A dedicated experiment
  quantifying how processing order and prior accepts change bits-mode decisions.
  Minimum content: (a) order-robustness of the noise filter -- run EX32's noisy
  KB under noise-first vs true-pattern-first vs interleaved orderings and report
  how the spurious-accept count and the per-proposal bits delta shift; (b) the
  amortization discount -- measure the bits delta of a fixed shallow proposal as
  a function of how many declarations (not/1, exception functors) are already in
  the dictionary, isolating the discount magnitude; (c) a statement of when the
  noise-filter / selectivity claims hold and when amortization erodes them.
  Harness-recorded; registry entry EX36; depends_on [EX29, EX32, EX33].

## Phase 2: Op I open-world partial-closure extension (library feature)

EX35 quantified the exact-match cliff. This phase closes it. Treated as a proper
feature with the same discipline as P3 (spec -> plan -> TDD subagent-driven ->
review -> zero-drift gating). NOT a fire-and-forget edit.

- [ ] **T3 -- Spec.** `docs/superpowers/specs/2026-06-18-opi-open-world.md`:
  relax Op I's exact-match gate to a subset/partial-closure match under an
  open-world option, MDL-gated (the recovered closure must pay under the active
  DL). Pin: detection (r subset of closure(b), with a density/coverage
  threshold), the open-world flag interaction, the verification-suite treatment
  of newly-derivable pairs (they are intended, not drift), and zero-drift in the
  default closed-world/exact mode. Acceptance criteria incl. the existing EX27
  recursion tests staying green.
- [ ] **T4 -- Plan + implement.** TDD tasks; subagent-driven implementer +
  spec/quality review per task; regression + suite gating each step; the default
  path (exact, closed-world) must stay bit-identical.
- [ ] **T5 -- EX37: closed-gap re-characterization.** Re-run the EX35 density
  sweep against the EXTENDED Op I to show partial closures now recovered at
  p<1.0 under the open-world gate, with precision preserved (no spurious pairs).
  Harness-recorded; registry entry EX37; depends_on [EX35].

## Phase 3: Paper consolidation

- [ ] **T6 -- Fold EX32-37 into `paper/dreamlog_paper.tex`.** Extend the
  bits-DL experiments subsection (sec:bitsdl) with: the noise filter (EX32) and
  its context caveat as characterized by EX36; the predictive correlation
  (EX33); cross-domain transfer (EX34) with the conservatism reversal; and the
  Op I cliff (EX35) now closed by the open-world extension (EX37), updating the
  recursion-scope limitation in the Discussion. State every caveat honestly.
  Build clean; no undefined refs; report page count.

---

## Sequencing

T1 and T2 run first (parallel: T1 is a review, T2 a fresh experiment; neither
commits -- the orchestrator integrates). Phase 2 (T3-T5) is the largest and runs
after Phase 1, on its own spec/plan/review cycle. Phase 3 (T6) is last so the
paper incorporates the closed Op I gap. Each phase ends with a commit + push and
a memory update; the registry and harness records are updated as work lands.
