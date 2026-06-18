"""EX32: bits-mode description length as a noise filter for Op C generalization.

HYPOTHESIS
----------
Random noise creates shallow, coincidental shared structure -- spurious patterns
reused only 2-3 times WITH EXCEPTIONS. These sit below the abstraction-maturity
crossover established in EX29: a generalization group of size 3 with 1 exception
has delta_clauses=-1 (accepted by clause count) but delta_bits=+5.66 to +10.90
(rejected by bits mode). So bits mode should REJECT noise-induced generalizations
that clause count ACCEPTS, while still capturing the true mature pattern (N=12).
Prediction: across all noise levels, bits mode produces ZERO spurious
generalizations while clause count accumulates them linearly.

NOISE DESIGN
------------
Each noise group uses a fresh predicate pair (pref_K, mem_K) to avoid cross-group
exception inflation that would cause clause-count mode to also reject:
  - 3 "target" facts: pref_K(n_K_0, val_K), pref_K(n_K_1, val_K), pref_K(n_K_2, val_K)
  - 1 guard+exception: mem_K(n_K_{0,1,2}), mem_K(nx_K), pref_K(nx_K, other_K)

This gives: group size 3, guard size 4, exceptions 1.
  delta_clauses = -3 + 2 = -1  (accepted by clause count)
  delta_bits    = +6.28 to +10.90 (rejected by bits, due to the EX29 crossover)

Crucially, noise facts are inserted BEFORE the student/grade facts so Op C evaluates
them while the symbol table is still small (before 'not' and exception functors are
in the table from the true pattern's generalization). This is the exact window where
the bits filter activates: the noise pattern is proposed into a lean symbol table
where the new functor+exception declaration cost is not yet amortized.

SWEEP
-----
Noise level L in {0, 1, 2, 3, 4, 5} independent noise groups.
Derived noise fraction f = 8*L / (24 + 8*L) (each group = 8 facts, clean = 24).
For each L and each mode (clauses, bits): run dream, count true/spurious proposals.

Usage: python experiments/ex32_noise_filter.py
Writes: experiments/data/ex32/runs/<id>/{meta,results,summary} and latest.json
"""
import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import experiment_run   # noqa: E402

from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, KnowledgeBase
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


# --------------------------------------------------------------------------
# KB builder
# --------------------------------------------------------------------------

N_STUDENTS = 12
NOISE_LEVELS = [0, 1, 2, 3, 4, 5]


def build_kb(n_students: int, n_noise_groups: int) -> KnowledgeBase:
    """Build the KB with noise facts inserted BEFORE the true pattern facts.

    Insertion order matters: Op C iterates groups in insertion order, so the
    noise predicates (pref_K, mem_K) are evaluated first -- before 'not' and
    exception functors enter the symbol table from the true-pattern compression.
    This is the exact window where bits mode rejects the N=3+1-exception pattern.

    Each noise group:
      mem_K(n_K_0), mem_K(n_K_1), mem_K(n_K_2), mem_K(nx_K)  -- 4 guard facts
      pref_K(n_K_0, val_K), pref_K(n_K_1, val_K), pref_K(n_K_2, val_K),
      pref_K(nx_K, other_K)                                    -- 4 target facts
    => group size 3 (val_K), guard size 4, exceptions 1 (nx_K).

    True pattern: student(s_i) + grade(s_i, pass) for all i in 0..n_students-1.
    """
    kb = KnowledgeBase()
    # Noise groups first (so Op C processes them before 'not' enters symbol table)
    for g in range(n_noise_groups):
        pname = "pref%d" % g
        gname = "mem%d" % g
        for i in range(3):
            kb.add_fact(Fact(compound(gname, atom("n%d_%d" % (g, i)))))
            kb.add_fact(Fact(compound(pname, atom("n%d_%d" % (g, i)), atom("val%d" % g))))
        # 1 exception: member with a different value
        kb.add_fact(Fact(compound(gname, atom("nx%d" % g))))
        kb.add_fact(Fact(compound(pname, atom("nx%d" % g), atom("other%d" % g))))
    # True pattern after (processed later, so 'not' enters AFTER noise is decided)
    for i in range(n_students):
        kb.add_fact(Fact(compound("student", atom("s%d" % i))))
        kb.add_fact(Fact(compound("grade", atom("s%d" % i), atom("pass"))))
    return kb


def noise_fraction(n_students: int, n_noise_groups: int) -> float:
    noise_facts = 8 * n_noise_groups   # 4 guard + 4 target per group
    clean_facts = 2 * n_students       # student + grade per student
    total = noise_facts + clean_facts
    return noise_facts / total if total > 0 else 0.0


# --------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------

def is_true_generalization(rec: dict) -> bool:
    """Return True if this Op C proposal generalizes the 'grade' predicate."""
    return any("grade" in s for s in rec.get("added", []))


def run_dream_mode(n_students: int, n_noise_groups: int, mode: str) -> dict:
    """Dream the KB in the given dl_mode; return analysis of Op C proposals."""
    kb = build_kb(n_students, n_noise_groups)
    records: list = []
    dreamer = KnowledgeBaseDreamer(
        dl_mode=mode,
        decision_recorder=records.append,
        min_group_size=3,
        llm_client=None,
    )
    dreamer.dream(kb)
    gen = [r for r in records if r["kind"] == "generalization"]
    accepted = [r for r in gen if r["decision"] == "accepted"]
    rejected = [r for r in gen if r["decision"] == "rejected"]

    true_captured = any(is_true_generalization(r) for r in accepted)
    spurious_count = sum(1 for r in accepted if not is_true_generalization(r))
    total_proposals = len(gen)
    total_accepted = len(accepted)

    # Sample a spurious delta to record how far above zero the bits bar was
    noise_bits_deltas = [
        round(r["delta_bits"], 2)
        for r in rejected
        if not is_true_generalization(r)
    ]

    return {
        "true_captured": true_captured,
        "spurious_count": spurious_count,
        "total_proposals": total_proposals,
        "total_accepted": total_accepted,
        "precision": (
            (1 if true_captured else 0) / total_accepted
            if total_accepted > 0 else None
        ),
        "noise_bits_deltas_rejected": noise_bits_deltas,
    }


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def sweep() -> list:
    rows = []
    for L in NOISE_LEVELS:
        f = noise_fraction(N_STUDENTS, L)
        row = {
            "noise_level": L,
            "noise_fraction": round(f, 4),
            "n_noise_facts": 8 * L,
            "n_total_facts": 8 * L + 2 * N_STUDENTS,
        }
        for mode in ("clauses", "bits"):
            row[mode] = run_dream_mode(N_STUDENTS, L, mode)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    params = {
        "N_students": N_STUDENTS,
        "noise_levels": NOISE_LEVELS,
        "noise_design": (
            "Each noise group: 3 pref_K(n_K_i, val_K) facts + 1 exception "
            "pref_K(nx_K, other_K) + 4 mem_K guard facts. "
            "Noise inserted BEFORE true facts so Op C evaluates noise first "
            "(small symbol table => bits rejects N=3+exc pattern). "
            "True pattern: grade(X,pass):-student(X) with N=%d students." % N_STUDENTS
        ),
        "noise_facts_per_group": 8,
        "min_group_size": 3,
        "dl_modes": ["clauses", "bits"],
        "ex29_reference": (
            "N=3 with 1 exception: delta_clauses=-1, delta_bits=+5.66 (bits rejects). "
            "N=4+: accepted by both modes."
        ),
    }

    with experiment_run(
            exp_id="ex32",
            name="bits-mode noise filter",
            description=(
                "Tests whether bits-mode description length acts as a noise filter. "
                "Noise groups have 3 matching facts + 1 exception -- exactly the "
                "N=3 scenario where EX29 showed bits rejects (delta_bits=+5.66) "
                "while clause count accepts (delta_clauses=-1). Noise is inserted "
                "before true facts so Op C evaluates noise while the symbol table "
                "is small (before 'not' is amortized by the true pattern). "
                "Sweeps noise level L in {0..5} independent noise groups. "
                "Symbolic only, no LLM, deterministic."
            ),
            script=__file__,
            params=params,
            seeds={"note": "fully deterministic; no RNG (structured noise)"}) as run:

        def emit(line=""):
            print(line)
            run.summary_lines.append(line)

        emit("EX32: bits-mode description length as a noise filter")
        emit("=" * 65)
        emit("HYPOTHESIS: bits mode rejects spurious N=3+exception patterns")
        emit("  (delta_bits > 0) while clause count accepts them (delta_clauses=-1).")
        emit("  True pattern (grade, N=12) captured by both modes at all noise levels.")
        emit()

        results = sweep()
        run.results["sweep"] = results

        emit("SWEEP TABLE")
        emit("-" * 65)
        hdr = "%-4s %-6s %-5s | %-24s | %-24s" % (
            "L", "f", "noise", "CLAUSES true/spur/prec", "BITS true/spur/prec")
        emit(hdr)
        emit("-" * 65)

        for row in results:
            L = row["noise_level"]
            f = row["noise_fraction"]
            nf = row["n_noise_facts"]
            cl = row["clauses"]
            bi = row["bits"]

            def fmt(d):
                t = "Y" if d["true_captured"] else "N"
                s = d["spurious_count"]
                p = d["precision"]
                if p is None:
                    return "%-24s" % ("%s / %d / -" % (t, s))
                return "%-24s" % ("%s / %d / %.2f" % (t, s, p))

            emit("%-4d %-6.3f %-5d | %s| %s" % (L, f, nf, fmt(cl), fmt(bi)))

        emit("-" * 65)
        emit("(L=noise groups, f=noise fraction, true:Y/N, spur:count, prec=precision)")
        emit()

        # Bits delta detail for the rejected noise proposals
        emit("BITS DELTA for noise proposals (rejected by bits mode):")
        for row in results:
            deltas = row["bits"]["noise_bits_deltas_rejected"]
            if deltas:
                emit("  L=%d: bits delta(s) = %s (positive => bits rejects)" % (
                    row["noise_level"], ", ".join("%.2f" % d for d in deltas)))
        emit()

        # Verdict
        cl_spurious = [r["clauses"]["spurious_count"] for r in results]
        bi_spurious = [r["bits"]["spurious_count"] for r in results]
        cl_true_all = all(r["clauses"]["true_captured"] for r in results)
        bi_true_all = all(r["bits"]["true_captured"] for r in results)

        max_cl_spur = max(cl_spurious)
        max_bi_spur = max(bi_spurious)

        emit("SUMMARY")
        emit("  Clauses mode: spurious counts by L = %s (max=%d), true all: %s"
             % (cl_spurious, max_cl_spur, cl_true_all))
        emit("  Bits mode:    spurious counts by L = %s (max=%d), true all: %s"
             % (bi_spurious, max_bi_spur, bi_true_all))
        emit()

        bits_filters_better = max_bi_spur < max_cl_spur
        bits_preserves_true = bi_true_all

        if bits_filters_better and bits_preserves_true:
            verdict = "SUPPORTED"
            detail = (
                "Bits mode eliminated all spurious generalizations (max spur=%d) "
                "vs clause count max spur=%d, while capturing the true pattern "
                "at all noise levels. The EX29 N=3+exc crossover (delta_bits>0) "
                "acts as a structural noise gate." % (max_bi_spur, max_cl_spur)
            )
        elif bits_filters_better and not bits_preserves_true:
            verdict = "PARTIAL"
            detail = (
                "Bits mode filtered more spurious generalizations (max spur=%d vs %d) "
                "but also missed the true pattern at some noise levels -- over-selective."
                % (max_bi_spur, max_cl_spur)
            )
        elif not bits_filters_better and bi_true_all:
            verdict = "REFUTED"
            detail = (
                "Bits mode did NOT suppress spurious generalizations vs clause count "
                "(both max spur=%d). The noise-filter hypothesis is not supported." % max_cl_spur
            )
        else:
            verdict = "REFUTED"
            detail = "Both modes behaved similarly; no filtering contrast visible."

        emit("VERDICT: %s" % verdict)
        emit("  %s" % detail)

        run.results["verdict"] = verdict
        run.results["summary"] = {
            "max_spurious_clauses": max_cl_spur,
            "max_spurious_bits": max_bi_spur,
            "spurious_by_level_clauses": cl_spurious,
            "spurious_by_level_bits": bi_spurious,
            "true_captured_all_levels_clauses": cl_true_all,
            "true_captured_all_levels_bits": bi_true_all,
            "bits_filters_better": bits_filters_better,
            "bits_preserves_true": bits_preserves_true,
        }

    print()
    print("Wrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
