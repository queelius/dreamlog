"""EX35: Operation I data-density characterization -- partial-closure gap.

Operation I fires ONLY when the target binary predicate R is the EXACT
transitive closure of a base relation B (closed-world exact-match gate in
compression/generators/closure.py: r_ext != transitive_closure(b_ext)).
This experiment quantifies what that means empirically:

  Sweep 1 -- COMPLETENESS: vary the fraction p of closure pairs present.
    Op I should fire only at p=1.0 (exact match). Confirms the boundary.

  Sweep 2 -- BASE-DENSITY: at p=1.0, vary chain length L. Confirms the
    min_base_facts=3 threshold and shows how the bits delta grows with L.

  Sweep 3 -- OVER-COMPLETE (noise on closure): at p=1.0, add spurious r-facts
    not in the true closure. Op I must NOT fire (r_ext != closure(b_ext)).
    Sweeps spurious count {0, 1, 2, 3}.

  Gap summary: at p<1.0, how many closure pairs are left on the table?

Symbolic only, no LLM, deterministic (RNG seed for sampling fixed).

Usage: python experiments/ex35_opi_density.py
Writes: experiments/data/ex35/runs/<id>/{meta,results,summary} + latest.json
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import experiment_run   # noqa: E402

from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, KnowledgeBase, Rule
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.recursive_discovery import transitive_closure

# ---------------------------------------------------------------------------
# KB builders
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
    Closure relation: r(ni, nj) for sampled subset of the full closure.
    """
    rng = random.Random(seed)
    kb = KnowledgeBase()
    # All base facts
    for a, b in sorted(_chain_edges(chain_len)):
        kb.add_fact(compound("b", atom(a), atom(b)))
    # Sampled closure facts
    all_pairs = sorted(_full_closure_pairs(chain_len))
    k = max(1, round(len(all_pairs) * fraction))
    if fraction >= 1.0:
        sampled = all_pairs
    else:
        sampled = rng.sample(all_pairs, k)
    for a, b in sampled:
        kb.add_fact(compound("r", atom(a), atom(b)))
    return kb, len(all_pairs), len(sampled)


def _spurious_kb(chain_len, n_spurious, seed):
    """p=1.0 (full closure) plus n_spurious r-facts whose pairs are NOT in
    the true closure. We need node labels outside the chain to guarantee
    the spurious facts are not in the closure."""
    rng = random.Random(seed)
    kb = KnowledgeBase()
    # Base facts
    for a, b in sorted(_chain_edges(chain_len)):
        kb.add_fact(compound("b", atom(a), atom(b)))
    # Full closure
    all_pairs = _full_closure_pairs(chain_len)
    for a, b in sorted(all_pairs):
        kb.add_fact(compound("r", atom(a), atom(b)))
    # Spurious facts using external node labels
    ext_nodes = ["x%d" % i for i in range(20)]
    spurious_pool = []
    for u in ext_nodes:
        for v in ext_nodes:
            if u != v:
                pair = (u, v)
                if pair not in all_pairs:
                    spurious_pool.append(pair)
    picked = rng.sample(spurious_pool, min(n_spurious, len(spurious_pool)))
    for a, b in picked:
        kb.add_fact(compound("r", atom(a), atom(b)))
    return kb, len(all_pairs), len(picked)


def _op_i_fired(kb):
    """Run the dreamer with discover_recursion=True; return whether a
    recursion proposal was accepted. Uses a recorder to catch the decision.
    The gate records decision="accepted" on success, "rejected" on failure."""
    recs = []
    KnowledgeBaseDreamer(
        discover_recursion=True,
        dl_mode="clauses",
        decision_recorder=recs.append,
    ).dream(kb)
    hit = next((r for r in recs if r["kind"] == "recursion"), None)
    if hit is not None:
        return hit["decision"] == "accepted", hit.get("delta_clauses"), hit.get("delta_bits")
    return False, None, None


# ---------------------------------------------------------------------------
# Sweep 1: completeness (vary p)
# ---------------------------------------------------------------------------

COMPLETENESS_FRACTIONS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
CHAIN_LENGTHS_COMPLETENESS = [4, 6, 8]
SAMPLING_SEED = 42


def completeness_sweep():
    """For each (L, p), build KB and record whether Op I fired."""
    rows = []
    for chain_len in CHAIN_LENGTHS_COMPLETENESS:
        for p in COMPLETENESS_FRACTIONS:
            kb, total, sampled = _partial_closure_kb(chain_len, p, SAMPLING_SEED)
            fired, delta_c, delta_bits = _op_i_fired(kb)
            rows.append({
                "chain_len": chain_len,
                "fraction": p,
                "total_closure_pairs": total,
                "sampled_pairs": sampled,
                "op_i_fired": fired,
                "delta_clauses": delta_c,
                "recoverable_pairs_not_present": total - sampled,
            })
    return rows


# ---------------------------------------------------------------------------
# Sweep 2: base-density (vary L at p=1.0)
# ---------------------------------------------------------------------------

CHAIN_LENGTHS_DENSITY = list(range(2, 11))   # L in {2..10}


def base_density_sweep():
    """At p=1.0 (exact closure), vary chain length L. Confirm min_base_facts=3
    threshold and track bits delta growth."""
    rows = []
    for chain_len in CHAIN_LENGTHS_DENSITY:
        base_facts = chain_len           # L base facts for chain of length L
        closure_pairs = chain_len * (chain_len + 1) // 2
        kb, _, _ = _partial_closure_kb(chain_len, 1.0, SAMPLING_SEED)
        recs = []
        KnowledgeBaseDreamer(
            discover_recursion=True,
            dl_mode="bits",
            decision_recorder=recs.append,
        ).dream(kb)
        hit = next((r for r in recs if r["kind"] == "recursion"), None)
        if hit is not None:
            decision = hit["decision"]
            delta_bits = round(hit["delta_bits"], 2)
            delta_clauses = hit["delta_clauses"]
        else:
            decision = "no_proposal"
            delta_bits = None
            delta_clauses = None
        rows.append({
            "chain_len": chain_len,
            "base_facts": base_facts,
            "closure_pairs": closure_pairs,
            "total_clauses_before": base_facts + closure_pairs,
            "op_i_fired": (decision == "accepted"),
            "decision": decision,
            "delta_clauses": delta_clauses,
            "delta_bits": delta_bits,
        })
    return rows


# ---------------------------------------------------------------------------
# Sweep 3: over-complete (spurious facts)
# ---------------------------------------------------------------------------

SPURIOUS_COUNTS = [0, 1, 2, 3]
CHAIN_LEN_SPURIOUS = 6   # L=6 gives a meaningful closure size


def overcomplete_sweep():
    """At p=1.0, add n_spurious r-facts not in the true closure. Op I must
    NOT fire (r_ext != closure(b_ext) when spurious facts are present)."""
    rows = []
    for n_spurious in SPURIOUS_COUNTS:
        kb, true_pairs, actual_spurious = _spurious_kb(
            CHAIN_LEN_SPURIOUS, n_spurious, SAMPLING_SEED)
        fired, delta_c, delta_bits = _op_i_fired(kb)
        total_r_facts = true_pairs + actual_spurious
        rows.append({
            "chain_len": CHAIN_LEN_SPURIOUS,
            "n_spurious_requested": n_spurious,
            "n_spurious_added": actual_spurious,
            "true_closure_pairs": true_pairs,
            "total_r_facts": total_r_facts,
            "op_i_fired": fired,
            "delta_clauses": delta_c,
        })
    return rows


# ---------------------------------------------------------------------------
# Gap summary
# ---------------------------------------------------------------------------

def gap_summary(completeness_rows, base_density_rows):
    """Quantify structure left on the table at p<1.0.

    For each (chain_len, p<1.0) row in the completeness sweep, the recoverable
    structure is the number of closure pairs NOT present that a recursive rule
    WOULD derive. Since Op I abstains, none of them are recovered.
    """
    gap_rows = []
    for r in completeness_rows:
        if r["fraction"] >= 1.0:
            continue
        gap_rows.append({
            "chain_len": r["chain_len"],
            "fraction": r["fraction"],
            "closure_pairs_present": r["sampled_pairs"],
            "closure_pairs_total": r["total_closure_pairs"],
            "pairs_left_on_table": r["recoverable_pairs_not_present"],
            "op_i_fired": r["op_i_fired"],
            "note": (
                "recursive rule would cover %d more pairs but Op I abstains "
                "(r_ext != closure(b_ext))"
                % r["recoverable_pairs_not_present"]
            ),
        })
    # Find the L=8, p=0.9 case for the specific call-out in the paper
    l8_p09 = next(
        (r for r in gap_rows
         if r["chain_len"] == 8 and abs(r["fraction"] - 0.9) < 0.01),
        None)
    boundary = {
        "op_i_fire_condition": "p == 1.0 AND n_spurious == 0 AND base_facts >= min_base_facts(3)",
        "exact_match_is_strict": True,
        "l8_p09_pairs_left_on_table": l8_p09["pairs_left_on_table"] if l8_p09 else None,
        "l8_p09_fraction": 0.9,
    }
    return {"rows": gap_rows, "boundary": boundary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    params = {
        "completeness_fractions": COMPLETENESS_FRACTIONS,
        "chain_lengths_completeness": CHAIN_LENGTHS_COMPLETENESS,
        "chain_lengths_density": CHAIN_LENGTHS_DENSITY,
        "chain_len_spurious": CHAIN_LEN_SPURIOUS,
        "spurious_counts": SPURIOUS_COUNTS,
        "min_base_facts_default": 3,
    }

    with experiment_run(
            exp_id="ex35",
            name="Op I data-density characterization -- partial-closure gap",
            description=(
                "Characterizes Operation I (recursive closure discovery) as a "
                "function of closure completeness, base-relation density, and "
                "over-completion noise. Quantifies the partial-closure gap: "
                "Op I fires only when r_ext == transitive_closure(b_ext) exactly "
                "(p=1.0, no spurious facts, base_facts >= 3). At p<1.0, "
                "Op I abstains and recoverable structure is left on the table."
            ),
            script=__file__,
            params=params,
            seeds={"sampling": SAMPLING_SEED}) as run:

        def emit(line=""):
            print(line)
            run.summary_lines.append(line)

        emit("EX35: Op I data-density characterization")
        emit("=" * 60)

        # -- Sweep 1: completeness --
        emit("\n1. COMPLETENESS SWEEP -- fraction p of closure pairs present")
        emit("   Op I fires ONLY at p=1.0 (exact match gate)")
        cs = completeness_sweep()
        run.results["completeness_sweep"] = cs

        emit("   %-4s  %-5s  %-8s  %-8s  %-10s  %-8s  %-12s"
             % ("L", "p", "total", "present", "Op_I_fired", "delta_c", "left_on_table"))
        for r in cs:
            emit("   %-4d  %-5.2f  %-8d  %-8d  %-10s  %-8s  %-12d"
                 % (r["chain_len"], r["fraction"],
                    r["total_closure_pairs"], r["sampled_pairs"],
                    str(r["op_i_fired"]),
                    str(r["delta_clauses"]) if r["delta_clauses"] is not None else "--",
                    r["recoverable_pairs_not_present"]))

        # -- Sweep 2: base density --
        emit("\n2. BASE-DENSITY SWEEP -- chain length L at p=1.0 (exact closure)")
        emit("   min_base_facts=3: Op I must NOT fire for L<3 (base_facts<3)")
        ds = base_density_sweep()
        run.results["base_density_sweep"] = ds

        emit("   %-4s  %-10s  %-13s  %-18s  %-10s  %-10s  %-10s"
             % ("L", "base_facts", "closure_pairs", "total_clauses_before",
                "decision", "delta_c", "delta_bits"))
        for r in ds:
            emit("   %-4d  %-10d  %-13d  %-18d  %-10s  %-10s  %-10s"
                 % (r["chain_len"], r["base_facts"], r["closure_pairs"],
                    r["total_clauses_before"], r["decision"],
                    str(r["delta_clauses"]) if r["delta_clauses"] is not None else "--",
                    str(r["delta_bits"]) if r["delta_bits"] is not None else "--"))

        # -- Sweep 3: over-complete --
        emit("\n3. OVER-COMPLETE SWEEP -- spurious r-facts at p=1.0 (L=%d)"
             % CHAIN_LEN_SPURIOUS)
        emit("   Op I must NOT fire when spurious facts are present")
        ocs = overcomplete_sweep()
        run.results["overcomplete_sweep"] = ocs

        emit("   %-10s  %-12s  %-13s  %-12s  %-10s"
             % ("n_spurious", "true_pairs", "total_r_facts", "op_i_fired", "delta_c"))
        for r in ocs:
            emit("   %-10d  %-12d  %-13d  %-12s  %-10s"
                 % (r["n_spurious_added"], r["true_closure_pairs"],
                    r["total_r_facts"], str(r["op_i_fired"]),
                    str(r["delta_clauses"]) if r["delta_clauses"] is not None else "--"))

        # -- Gap summary --
        emit("\n4. PARTIAL-CLOSURE GAP SUMMARY")
        gs = gap_summary(cs, ds)
        run.results["gap_summary"] = gs

        emit("   Op I fires only when: p=1.0 AND no spurious AND base_facts>=3")
        emit("   At p<1.0, recoverable closure pairs left on the table:")
        for r in gs["rows"]:
            if r["chain_len"] == 8:   # print L=8 row as the key example
                emit("   L=%-2d p=%.2f -- %d of %d closure pairs present, "
                     "%d left on the table (Op I abstains)"
                     % (r["chain_len"], r["fraction"],
                        r["closure_pairs_present"],
                        r["closure_pairs_total"],
                        r["pairs_left_on_table"]))

        boundary = gs["boundary"]
        l8_p09 = boundary["l8_p09_pairs_left_on_table"]
        emit("\n   KEY FINDING: At L=8, p=0.9 -- %s closure pairs recoverable "
             "by a recursive rule\n   are left on the table because Op I "
             "requires exact-match (p=1.0)." % str(l8_p09))
        emit("\n   Fire condition: %s" % boundary["op_i_fire_condition"])

        # Summarize base-density threshold
        threshold_fires_at = next(
            (r["chain_len"] for r in ds if r["op_i_fired"]), None)
        emit("\n   min_base_facts threshold confirmed: Op I first fires at "
             "L=%s (base_facts=%s, min_base_facts=3)"
             % (str(threshold_fires_at),
                str(threshold_fires_at) if threshold_fires_at else "N/A"))

        # Delta growth for base-density
        emit("\n   bits-delta growth with chain length (Op I, p=1.0):")
        for r in ds:
            if r["op_i_fired"]:
                emit("   L=%-2d  closure=%3d  delta_bits=%-8s  delta_clauses=%s"
                     % (r["chain_len"], r["closure_pairs"],
                        str(r["delta_bits"]), str(r["delta_clauses"])))

        emit("\n" + "=" * 60)
        emit("EX35 complete. Zero library changes (symbolic only, no LLM).")

    print("\nWrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
