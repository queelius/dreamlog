# Spec: Operation I open-world partial-closure discovery

**Date:** 2026-06-18
**Status:** Draft for review (Phase 2 / T3 of the bits-DL follow-up plan)
**Motivates:** EX35 quantified that Op I fires only under an exact-match gate
(`r_ext == transitive_closure(b_ext)`), so any holdout gap or missing pair
silences it -- at L=8, 90% complete, 4 recoverable pairs go uncaptured. This
spec relaxes the gate, under the existing `open_world` option, so a recursive
rule can be discovered from a PARTIAL closure and recover the missing pairs.

## 1. Goal

When the observed extension of a binary predicate R is a sufficiently complete
SUBSET of the transitive closure of another binary predicate B, synthesize the
right-recursive definition of R over B (the existing base + recursive pair). The
synthesized rule derives the FULL closure of B, recovering the closure pairs
absent from R's observed extension. This is desirable only when recovering absent
facts is the intent, so it is gated on `open_world=True`. The closed-world,
exact-match path is unchanged (zero drift).

## 2. Non-goals

- No change to closed-world behavior. With `open_world=False` (the default), Op I
  keeps the exact `==` gate, bit-for-bit. EX27's recursion tests must stay green.
- No new recursion shapes. Still single-base, right-recursive, exact transitive
  closure of B (not mutual recursion, accumulators, or list recursion).
- No change to the bounded-evaluator depth limit; long linear closures remain
  out of scope (documented in the closure generator).

## 3. The relaxed detection (open-world only)

For a candidate ordered pair (R, B) with observed extensions `r_ext`, `b_ext`
(Atom-only binary pairs, as today), let `C = transitive_closure(b_ext)`.

Current closed-world gate (kept): accept iff `r_ext == C`.

New open-world gate: accept the partial closure iff ALL of:
1. **Subset (soundness):** `r_ext is a subset of C`. Every observed R pair is
   justified by B's closure. If any observed pair is outside C, R is not a
   (partial) closure of B -- reject. This prevents recovering a relation that
   contradicts the observed data.
2. **Coverage (confidence):** `|r_ext| / |C| >= tau`, where `tau =
   min_closure_coverage` (new dreamer parameter, default 0.5). Below tau the
   observed extension is too sparse to confidently call it a closure rather than
   a coincidental subset. EX35/EX37 calibrate tau.
3. **Base-size guard (kept):** `|b_ext| >= min_base_facts`.
4. **Non-triviality:** `r_ext != C` would route to the exact path; the open-world
   path handles the strict-subset case `r_ext (strict subset) C` with at least
   one recoverable pair.

On acceptance the synthesized clauses are the SAME as today (base + right-
recursive rule over B). The difference is only in which (R, B) pairs pass the
gate and that the rule now derives `C \ r_ext` (the recovered pairs).

Detector order: the exact path is tried first (closed or open world); the
subset path is tried only in open world and only when the exact path did not
fire for that (R, B). At most one candidate per call, as today.

## 4. Verification-suite interaction (the delicate part)

The bounded suite (Definition: S+ all ground facts; S- synthetic negatives) is
verified after trial-applying the proposal. Two issues in open-world partial mode:

- **S+ (positives) -- safe.** The synthesized rule re-derives every pair in
  `r_ext` (since `r_ext subset C` and the rule computes exactly `C`). So all
  original R facts remain derivable; the rest of S+ is untouched. No relaxation
  needed.
- **S- (negatives) -- needs relaxation.** Some synthetic negatives for the R
  functor may fall inside `C` -- these are exactly the pairs we intend to
  recover. Under the default suite they would make verification FAIL. This is the
  Op I analog of Op G's open-world false-positive relaxation. **Decision:** in
  open-world partial mode, the policy excludes from the negative check any S-
  query of the form `R(a,b)` with `(a,b) in C`. Negatives for other functors, and
  R-negatives outside C (genuinely spurious), are still enforced. This makes the
  recovered closure pairs intended, while still catching real over-derivation
  (an R pair outside the mathematical closure cannot be derived by the rule
  anyway, so any such S- stays satisfied).

Implementation note: the closure generator knows `C` at proposal time; it passes
the predicted closure (or the recovered-pair set) to the policy via the
`Proposal.notes` field so the open-world policy can filter S- accordingly. The
closed-world/exact policy ignores `notes` and is unchanged.

## 5. Description-length interaction

Unchanged in form. The proposal still removes `r_ext` facts and adds two rules,
so both clause-count and bits deltas are computed exactly as today. In bits mode
the recovered pairs are NOT added as facts (they become derivable, not stored),
so the delta is strongly negative whenever `|r_ext|` exceeds the two-rule cost --
the same favorable economics as exact closure, and partial closures with high
coverage clear it easily.

## 6. Surface / parameters

- `KnowledgeBaseDreamer.__init__` gains `min_closure_coverage: float = 0.5`
  (only consulted when `open_world=True and discover_recursion=True`). Stored;
  threaded to the closure policy/generator like the other Op I params.
- `closure.run(...)` gains the coverage threshold and an `open_world` flag (read
  from the dreamer, as the other generators read their config).
- `BoundedSuitePolicy` (or a thin `ClosurePolicy` subclass) gains the open-world
  S- relaxation keyed on the predicted closure from `Proposal.notes`.
- No change to `proposal.py`, `gate.py`, `dl.py`.

## 7. Acceptance criteria

- **AC1 (zero drift):** with `open_world=False` (default), every existing test
  passes unchanged, including the EX27 recursion tests and
  `tests/test_mdl_refactor_regression.py`. The closed-world closure path is
  byte-identical. Full suite stays 902/11/3.
- **AC2 (recovery):** a new unit test builds a base chain B with a STRICT-subset
  R (coverage above tau, at least one missing closure pair) and asserts that,
  with `open_world=True`, Op I fires and the dreamed KB DERIVES the missing pairs
  (recovery > 0), with no derivation outside `C` for R (precision preserved).
- **AC3 (threshold):** below tau, the open-world path does NOT fire (a sparse
  subset is left alone); at/above tau it fires. A unit test pins both sides.
- **AC4 (soundness):** if an observed R pair lies outside `C` (R is not a closure
  of B), the open-world path rejects (no spurious recursive rule).
- **AC5 (EX37):** re-running the EX35 density sweep with `open_world=True` shows
  partial closures recovered at p<1.0 down to tau, precision preserved, recorded
  via the harness.

## 8. Risks

- **R1 (over-recovery / negatives):** the S- relaxation could mask a genuine
  over-derivation. Mitigated because the synthesized rule computes exactly
  `transitive_closure(B)`; it cannot derive an R pair outside `C`, so only
  intended pairs are ever recovered, and non-R negatives stay fully enforced.
- **R2 (coincidental subset):** a sparse R that happens to sit inside some B's
  closure could trigger a spurious recursive rule. Mitigated by the coverage
  threshold tau and `min_base_facts`; EX37 calibrates tau against the
  false-trigger rate.
- **R3 (multiple candidate B):** an R could be a partial closure of more than one
  B. Keep today's first-match, one-candidate-per-call behavior; the suite +
  DL gate filter wrong matches.
