"""EX37: Op I open-world closed-gap sweep.

EX35 characterized the partial-closure gap: Op I fires ONLY when the target
binary predicate R is the EXACT transitive closure of a base relation B
(closed-world exact-match gate). At L=8, p=0.9, four recoverable closure
pairs are left on the table because the gate requires p=1.0.

Commit 1d0928d introduced the open-world partial-closure extension: with
  KnowledgeBaseDreamer(discover_recursion=True, open_world=True,
                       min_closure_coverage=0.5)
Op I now fires on a SUBSET closure (coverage >= tau=0.5) and the synthesized
recursive rule RECOVERS the missing closure pairs.

This experiment re-runs the EX35 density sweep against the extended Op I and
shows the gap is closed, with precision preserved.

PROTOCOL
--------
1. CLOSED-GAP COMPLETENESS SWEEP: for chain lengths L in {4,6,8} and
   completeness fractions p in {0.4,0.5,0.6,0.7,0.8,0.9,1.0}:
   build the KB (all L base facts + a sampled p-fraction of closure pairs).
   Run BOTH open_world=False AND open_world=True. For each record:
     - did Op I gate accept (recursion proposal accepted by ClosurePolicy)?
     - did the dream COMMIT (session.compressed = True, no rollback)?
     - RECOVERY: fraction of held-out closure pairs now derivable (in
       committed KB; 0.0 if dream rolled back)
     - PRECISION: spurious R(a,b) derivations outside the true closure C

   IMPORTANT: "gate_accepted" and "committed" can differ. In open-world mode
   the ClosurePolicy relaxes the negative check for recovered pairs (spec
   2026-06-18 Section 4). But the FINAL suite verification in dream() uses
   the UNFILTERED original suite. If any synthetic negative for R falls in the
   recovered set AND was not excluded from the original suite, the final check
   fails and the dream rolls back. Whether a held-out pair appears as a
   synthetic negative depends on the KB structure and build_verification_suite
   (which generates few negatives by single-arg substitution). This is a
   measurement artifact, not a gate failure: at high p and short L, the
   synthetic negatives are unlikely to collide with held-out pairs, so
   committed=True; at lower p or longer L, collisions happen and the dream
   rolls back. Reported explicitly per (L, p, mode) so the ambiguity is
   visible.

2. THRESHOLD CONFIRMATION: the open-world gate fires at p>=tau=0.5 and NOT
   below (p=0.4 must not fire).

3. PRECISION GUARANTEE: across all committed points, zero spurious derivations.

4. GAP-CLOSED SUMMARY: pairs recovered that EX35 left on the table, counting
   only committed dreams.

Symbolic only, no LLM, deterministic (RNG seed fixed).

Usage: python experiments/ex37_opi_open_world.py
Writes: experiments/data/ex37/runs/<id>/{meta,results,summary} + latest.json
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import experiment_run  # noqa: E402

from dreamlog.factories import atom, compound
from dreamlog.knowledge import KnowledgeBase
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.evaluator import PrologEvaluator
from dreamlog.recursive_discovery import transitive_closure
from dreamlog.terms import Atom, Compound

# ---------------------------------------------------------------------------
# KB builders (mirrors EX35 exactly)
# ---------------------------------------------------------------------------

def _chain_edges(chain_len):
    """Directed edges n0->n1->...->n{chain_len}."""
    nodes = ["n%d" % i for i in range(chain_len + 1)]
    return {(nodes[i], nodes[i + 1]) for i in range(chain_len)}


def _full_closure_pairs(chain_len):
    """All transitive-closure pairs for a chain of length chain_len."""
    return transitive_closure(_chain_edges(chain_len))


def _partial_closure_kb(chain_len, fraction, seed):
    """Build a KB with all base-chain facts plus a p-fraction of closure pairs.

    Base relation: b(ni, ni+1) for i in 0..chain_len-1
    Closure relation: r(ni, nj) for a sampled subset of the full closure.

    Returns (kb, total_closure_pairs, sampled_count, sampled_set, all_pairs_set).
    """
    rng = random.Random(seed)
    kb = KnowledgeBase()
    # All base facts
    for a, b in sorted(_chain_edges(chain_len)):
        kb.add_fact(compound("b", atom(a), atom(b)))
    # Sampled closure facts
    all_pairs = sorted(_full_closure_pairs(chain_len))
    all_pairs_set = set(map(tuple, all_pairs))
    k = max(1, round(len(all_pairs) * fraction))
    if fraction >= 1.0:
        sampled = all_pairs
    else:
        sampled = rng.sample(all_pairs, k)
    sampled_set = set(map(tuple, sampled))
    for a, b in sampled:
        kb.add_fact(compound("r", atom(a), atom(b)))
    return kb, len(all_pairs), len(sampled), sampled_set, all_pairs_set


# ---------------------------------------------------------------------------
# Dream + measurement helpers
# ---------------------------------------------------------------------------

def _dream_and_check(kb, chain_len, all_closure_pairs, sampled_set, open_world):
    """Run the dreamer and measure outcomes.

    Returns a dict:
      gate_accepted: bool -- did ClosurePolicy accept the recursion proposal?
      committed:     bool -- did the dream commit (session.compressed=True)?
      recovery:      float -- fraction of held-out pairs now derivable (0 if rollback)
      spurious_count: int -- R(a,b) derivable but outside the true closure C

    gate_accepted and committed can differ: the ClosurePolicy (Op I's own gate)
    relaxes S- for recovered pairs, but the FINAL suite check in dream() uses
    the unfiltered original suite. If a held-out pair appears as a synthetic
    negative in that suite, the final check fails and the dream rolls back
    even though the gate accepted. This is explicitly tracked so the table
    distinguishes gate behavior from end-to-end commitment.
    """
    recs = []
    dreamed_kb = kb.copy()

    session = KnowledgeBaseDreamer(
        discover_recursion=True,
        open_world=open_world,
        min_closure_coverage=0.5,
        dl_mode="clauses",
        decision_recorder=recs.append,
    ).dream(dreamed_kb)

    hit = next((r for r in recs if r["kind"] == "recursion"), None)
    gate_accepted = (hit is not None and hit["decision"] == "accepted")
    committed = session.compressed

    held_out = all_closure_pairs - sampled_set   # C \ r_ext

    if not held_out:
        # p=1.0: nothing to recover; full recovery by definition
        recovery = 1.0
    elif not committed:
        recovery = 0.0
    else:
        ev = PrologEvaluator(dreamed_kb)
        recovered = sum(
            1 for (a, b) in held_out
            if ev.has_solution(Compound("r", [Atom(a), Atom(b)]))
        )
        recovery = recovered / len(held_out)

    spurious_count = _count_spurious(dreamed_kb, chain_len, all_closure_pairs)

    return {
        "gate_accepted": gate_accepted,
        "committed": committed,
        "recovery": round(recovery, 4),
        "spurious_count": spurious_count,
    }


def _count_spurious(dreamed_kb, chain_len, all_closure_pairs):
    """Count R(a,b) derivable from dreamed KB but NOT in the true closure C.

    Only checks node pairs from the chain (n0..n{chain_len}); spurious
    derivations from the recursive rule are impossible by construction
    (it computes exactly transitive_closure(B)), but we verify this empirically.
    """
    nodes = ["n%d" % i for i in range(chain_len + 1)]
    ev = PrologEvaluator(dreamed_kb)
    spurious = 0
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            q = Compound("r", [Atom(a), Atom(b)])
            if ev.has_solution(q) and (a, b) not in all_closure_pairs:
                spurious += 1
    return spurious


# ---------------------------------------------------------------------------
# Sweep parameters (mirrors EX35 where applicable, adds p=0.4)
# ---------------------------------------------------------------------------

CHAIN_LENGTHS = [4, 6, 8]
COMPLETENESS_FRACTIONS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAU = 0.5
SAMPLING_SEED = 42


# ---------------------------------------------------------------------------
# Main sweeps
# ---------------------------------------------------------------------------

def completeness_sweep():
    """Side-by-side sweep: closed-world vs open-world for each (L, p).

    Returns a list of rows, one per (L, p, mode) triple.
    """
    rows = []
    for chain_len in CHAIN_LENGTHS:
        for p in COMPLETENESS_FRACTIONS:
            kb, total, sampled_n, sampled_set, all_pairs = _partial_closure_kb(
                chain_len, p, SAMPLING_SEED)
            held_out_n = total - sampled_n
            coverage = sampled_n / total if total > 0 else 0.0

            for open_world in [False, True]:
                res = _dream_and_check(
                    kb, chain_len, all_pairs, sampled_set, open_world)
                rows.append({
                    "chain_len": chain_len,
                    "fraction": p,
                    "coverage": round(coverage, 4),
                    "total_closure_pairs": total,
                    "sampled_pairs": sampled_n,
                    "held_out_pairs": held_out_n,
                    "open_world": open_world,
                    "gate_accepted": res["gate_accepted"],
                    "committed": res["committed"],
                    "recovery": res["recovery"],
                    "spurious_count": res["spurious_count"],
                })
    return rows


def threshold_confirmation(sweep_rows):
    """Confirm open-world gate fires when actual coverage>=tau and NOT below.

    Uses gate_accepted (per-operation ClosurePolicy decision) rather than
    committed, because gate behavior is what tau controls; the final suite
    rollback is a separate artifact.

    Note: the 'fraction' parameter (p) is the REQUESTED fraction; due to
    integer rounding of k=max(1,round(N*p)), the ACTUAL coverage may differ
    slightly from p. The threshold gate uses actual coverage. This function
    tests the gate against actual coverage (the 'coverage' field) so results
    are accurate even when rounding shifts a point across tau.
    """
    ow_rows = [r for r in sweep_rows if r["open_world"]]
    # Gate fires below tau (actual coverage) -- should not happen
    gate_below_tau = any(r["gate_accepted"] for r in ow_rows
                         if r["coverage"] < TAU)
    # Gate fires at or above tau -- should always happen (when base facts >= 3)
    gate_at_or_above_tau = all(r["gate_accepted"] for r in ow_rows
                               if r["coverage"] >= TAU)
    per_L = {}
    for chain_len in CHAIN_LENGTHS:
        L_rows = sorted([r for r in ow_rows if r["chain_len"] == chain_len],
                        key=lambda r: r["coverage"])
        first_gate_cov = next(
            (r["coverage"] for r in L_rows if r["gate_accepted"]), None)
        first_gate_p = next(
            (r["fraction"] for r in L_rows if r["gate_accepted"]), None)
        per_L[chain_len] = {
            "first_gate_fire_fraction_label": first_gate_p,
            "first_gate_fire_actual_coverage": first_gate_cov,
            "expected_threshold": TAU,
            "no_fire_below_tau": all(
                not r["gate_accepted"]
                for r in L_rows if r["coverage"] < TAU),
        }
    return {
        "tau": TAU,
        "gate_fires_below_tau": gate_below_tau,
        "gate_fires_at_or_above_tau_all_L": gate_at_or_above_tau,
        "per_chain_len": per_L,
        "threshold_confirmed": (not gate_below_tau and gate_at_or_above_tau),
    }


def precision_guarantee(sweep_rows):
    """Confirm zero spurious derivations across all committed points.

    Precision is measured on the post-dream KB; only committed dreams have
    had the recursive rule applied, so we check committed points only.
    For rolled-back dreams the original r-facts remain, and trivially no
    spurious derivations exist (the rule was never committed).
    """
    committed_rows = [r for r in sweep_rows if r["committed"]]
    violations = [r for r in committed_rows if r["spurious_count"] > 0]
    return {
        "total_committed_points": len(committed_rows),
        "zero_spurious_at_all_committed_points": len(violations) == 0,
        "violations": violations,
        "max_spurious_observed": max(
            (r["spurious_count"] for r in committed_rows), default=0),
    }


def gap_closed_summary(sweep_rows):
    """Quantify how much EX35 left on the table vs how much EX37 recovers.

    Uses committed recovery (end-to-end, post-dream KB derivability).
    Includes separate columns for gate_accepted vs committed to show the
    final-suite artifact where it occurs.
    """
    summary_rows = []
    for chain_len in CHAIN_LENGTHS:
        for p in COMPLETENESS_FRACTIONS:
            if p >= 1.0:
                continue
            cw = next((r for r in sweep_rows
                       if r["chain_len"] == chain_len
                       and abs(r["fraction"] - p) < 0.001
                       and not r["open_world"]), None)
            ow = next((r for r in sweep_rows
                       if r["chain_len"] == chain_len
                       and abs(r["fraction"] - p) < 0.001
                       and r["open_world"]), None)
            if cw is None or ow is None:
                continue

            held_out = cw["held_out_pairs"]
            ex35_gap = held_out if not cw["committed"] else 0
            recovered = round(ow["recovery"] * held_out) if held_out > 0 else 0

            summary_rows.append({
                "chain_len": chain_len,
                "fraction": p,
                "coverage": cw["coverage"],
                "total_closure": cw["total_closure_pairs"],
                "held_out": held_out,
                "cw_committed": cw["committed"],
                "ow_gate_accepted": ow["gate_accepted"],
                "ow_committed": ow["committed"],
                "ow_recovery": ow["recovery"],
                "ex35_gap_pairs": ex35_gap,
                "ex37_recovered_pairs": recovered,
                "gap_closed": (ex35_gap > 0 and recovered == ex35_gap),
            })

    l8_p09 = next(
        (r for r in summary_rows
         if r["chain_len"] == 8 and abs(r["fraction"] - 0.9) < 0.001),
        None)
    total_ex35_gap = sum(r["ex35_gap_pairs"] for r in summary_rows)
    total_recovered = sum(r["ex37_recovered_pairs"] for r in summary_rows)

    return {
        "rows": summary_rows,
        "l8_p09": l8_p09,
        "total_ex35_gap_pairs": total_ex35_gap,
        "total_ex37_recovered_pairs": total_recovered,
        "fraction_gap_closed": (total_recovered / total_ex35_gap
                                 if total_ex35_gap > 0 else 1.0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    params = {
        "chain_lengths": CHAIN_LENGTHS,
        "completeness_fractions": COMPLETENESS_FRACTIONS,
        "tau": TAU,
        "min_base_facts_default": 3,
    }

    with experiment_run(
            exp_id="ex37",
            name="Op I open-world closed-gap sweep",
            description=(
                "Re-runs the EX35 density sweep with open_world=True and "
                "min_closure_coverage=0.5 (tau). Shows that the partial-closure "
                "gap from EX35 is CLOSED by the open-world extension: Op I gate "
                "fires for all p >= tau (not only p=1.0). End-to-end recovery "
                "depends additionally on the final suite check not encountering "
                "a held-out closure pair as a synthetic negative (build_verification_suite "
                "artifact). Key results: L=8 p=0.9 recovers all 4 held-out pairs; "
                "zero spurious derivations at all committed points (precision preserved). "
                "Side-by-side comparison: each (L,p) point run in both modes; "
                "gate_accepted and committed tracked separately."
            ),
            script=__file__,
            params=params,
            seeds={"sampling": SAMPLING_SEED}) as run:

        def emit(line=""):
            print(line)
            run.summary_lines.append(line)

        emit("EX37: Op I open-world closed-gap sweep")
        emit("=" * 80)

        # ---- 1. Completeness sweep ----
        emit("")
        emit("1. COMPLETENESS SWEEP -- CW vs OW side by side for each (L, p)")
        emit("   tau=%.1f; seed=%d" % (TAU, SAMPLING_SEED))
        emit("   gate=ClosurePolicy accept; committed=dream.compressed (no rollback)")
        emit("")

        sweep = completeness_sweep()
        run.results["completeness_sweep"] = sweep

        # Header
        emit("   %-4s %-5s %-5s %-7s %-8s | %-5s %-6s %-6s | %-5s %-5s %-6s %-6s %-6s"
             % ("L", "p", "cov", "total", "h_out",
                "CW-g", "CW-c", "CW-rec",
                "OW-g", "OW-c", "OW-rec", "OW-spu", "key"))
        emit("   " + "-" * 82)
        for chain_len in CHAIN_LENGTHS:
            for p in COMPLETENESS_FRACTIONS:
                cw = next((r for r in sweep
                           if r["chain_len"] == chain_len
                           and abs(r["fraction"] - p) < 0.001
                           and not r["open_world"]), None)
                ow = next((r for r in sweep
                           if r["chain_len"] == chain_len
                           and abs(r["fraction"] - p) < 0.001
                           and r["open_world"]), None)
                if cw is None or ow is None:
                    continue
                # Key annotation
                if p < TAU:
                    key = "below-tau"
                elif not ow["gate_accepted"]:
                    key = "no-gate"
                elif ow["committed"] and ow["recovery"] == 1.0 and ow["held_out_pairs"] > 0:
                    key = "CLOSED"
                elif ow["committed"] and ow["recovery"] > 0:
                    key = "partial"
                elif ow["gate_accepted"] and not ow["committed"]:
                    key = "rollback"
                else:
                    key = "--"
                emit("   %-4d %-5.2f %-5.3f %-7d %-8d | %-5s %-6s %-6s | %-5s %-5s %-6s %-6d %-8s"
                     % (chain_len, p, cw["coverage"],
                        cw["total_closure_pairs"], cw["held_out_pairs"],
                        "T" if cw["gate_accepted"] else "F",
                        "T" if cw["committed"] else "F",
                        ("%.3f" % cw["recovery"]) if cw["committed"] else "--",
                        "T" if ow["gate_accepted"] else "F",
                        "T" if ow["committed"] else "F",
                        ("%.3f" % ow["recovery"]) if ow["committed"] else "--",
                        ow["spurious_count"],
                        key))

        emit("")
        emit("   Legend: g=gate_accepted, c=committed; 'rollback'=gate fired but")
        emit("   final suite check rolled back (held-out pair appeared as synthetic")
        emit("   negative in unfiltered final suite -- separate from gate behavior).")

        # ---- 2. Threshold confirmation ----
        emit("")
        emit("2. THRESHOLD CONFIRMATION")
        tc = threshold_confirmation(sweep)
        run.results["threshold"] = tc

        emit("   tau = %.1f (gate uses actual coverage after rounding, not label p)"
             % tc["tau"])
        emit("   gate fires below actual-coverage tau: %s (expected: False)"
             % tc["gate_fires_below_tau"])
        emit("   gate fires at or above tau for all L: %s (expected: True)"
             % tc["gate_fires_at_or_above_tau_all_L"])
        emit("   threshold_confirmed: %s" % tc["threshold_confirmed"])
        for L, v in sorted(tc["per_chain_len"].items()):
            emit("   L=%-2d: first gate fire at p=%-4s (actual cov=%.3f), "
                 "no_fire_below_tau=%s"
                 % (L,
                    str(v["first_gate_fire_fraction_label"]),
                    v["first_gate_fire_actual_coverage"] if v["first_gate_fire_actual_coverage"] is not None else 0.0,
                    v["no_fire_below_tau"]))

        # ---- 3. Precision guarantee ----
        emit("")
        emit("3. PRECISION GUARANTEE")
        pg = precision_guarantee(sweep)
        run.results["precision_guarantee"] = pg

        emit("   Committed points checked: %d" % pg["total_committed_points"])
        emit("   Zero spurious at all committed points: %s"
             % pg["zero_spurious_at_all_committed_points"])
        emit("   Max spurious observed: %d" % pg["max_spurious_observed"])
        if pg["violations"]:
            emit("   VIOLATIONS found: %s" % pg["violations"])
        else:
            emit("   No violations -- precision property holds.")

        # ---- 4. Gap-closed summary ----
        emit("")
        emit("4. GAP-CLOSED SUMMARY (committed recovery only)")
        gs = gap_closed_summary(sweep)
        run.results["gap_closed_summary"] = gs

        emit("   EX35 left: %d total pairs; EX37 committed recovery: %d (%.1f%%)"
             % (gs["total_ex35_gap_pairs"],
                gs["total_ex37_recovered_pairs"],
                100.0 * gs["fraction_gap_closed"]))
        emit("")
        emit("   %-4s %-5s %-5s %-8s | %-10s | %-6s %-8s %-9s | %-6s"
             % ("L", "p", "cov", "h_out",
                "EX35 gap", "OW-g", "OW-c", "rec", "closed?"))
        emit("   " + "-" * 72)
        for r in gs["rows"]:
            if r["ow_committed"]:
                status = "YES" if r["gap_closed"] else "partial"
            elif r["ow_gate_accepted"]:
                status = "rollback"
            else:
                status = "NO"
            emit("   %-4d %-5.2f %-5.3f %-8d | %-10d | %-6s %-8s %-9s | %-6s"
                 % (r["chain_len"], r["fraction"], r["coverage"],
                    r["held_out"],
                    r["ex35_gap_pairs"],
                    "T" if r["ow_gate_accepted"] else "F",
                    "T" if r["ow_committed"] else "F",
                    ("%.3f" % r["ow_recovery"]) if r["ow_committed"] else "--",
                    status))

        l8_p09 = gs["l8_p09"]
        if l8_p09:
            emit("")
            emit("   KEY EXAMPLE (L=8, p=0.9):")
            emit("   EX35 gap: %d pairs. EX37: gate=%s, committed=%s, "
                 "recovered=%d pairs. Gap closed: %s."
                 % (l8_p09["ex35_gap_pairs"],
                    l8_p09["ow_gate_accepted"],
                    l8_p09["ow_committed"],
                    l8_p09["ex37_recovered_pairs"],
                    l8_p09["gap_closed"]))

        emit("")
        emit("=" * 80)
        emit("EX37 complete. Zero library changes (symbolic only, no LLM).")

    print("")
    print("Wrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
