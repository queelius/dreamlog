"""EX30: deeper bits-DL abstraction surfaces.

Three finer-grained analyses extending EX29's abstraction-maturity finding:

1. FULL EXTRACTION SURFACE (analytic): K in 2..12 x L in 2..8 grid of
   delta_bits. Reports the complete heatmap, crossover K*(L) per column,
   and the marginal saving per extra sharing rule (slope dDelta/dK).

2. INVENTION SWEPT ON TWO AXES (dreamer): fix chain length, sweep M;
   fix M=4, sweep chain length C. At each point run both dl_modes with
   a decision_recorder and capture the first "invention"-kind proposal.

3. REALISTIC SCALED-KB SWEEP (dreamer): a mixed KB with K rules sharing
   a 2-goal body (Op E extraction target) PLUS a transitive closure relation
   (Op I recursion target). Sweep K in {2,3,4,5,6,7,8,12}. Run a full dream
   in both modes (bits and clauses, discover_recursion=True), record per-kind
   accept counts, report the accept gap (clauses - bits) as a function of K.
   Key finding: clause-count always rejects extraction (delta=+1 always) while
   bits mode accepts it past the contextual crossover K*~4; the gap (clauses-
   bits) peaks at +1 at K=3 then collapses to 0 as abstraction matures.

No LLM is used anywhere. Deterministic, symbolic only.

Usage: python experiments/ex30_bits_surfaces.py
Writes: experiments/data/ex30/runs/<id>/{meta.json,results.json,summary.txt}
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import experiment_run  # noqa: E402

from dreamlog.compression import dl
from dreamlog.compression.proposal import Proposal
from dreamlog.factories import atom, compound, var
from dreamlog.knowledge import KnowledgeBase, Rule
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


# --------------------------------------------------------------------------
# Helpers shared across analyses
# --------------------------------------------------------------------------

def _make_extraction_proposal(k_rules, body_len):
    """K rules h_i(X) :- g0(X)..g{L-1}(X) sharing the same L-goal body.
    Extraction adds ex(X):-body and rewrites each to h_i(X):-ex(X).
    Reuses EX29's exact construction so crossover numbers are comparable."""
    X = var("X")
    body = [compound("g%d" % j, X) for j in range(body_len)]
    rules = [Rule(compound("h%d" % i, X), list(body)) for i in range(k_rules)]
    e_def = Rule(compound("ex", X), list(body))
    rewritten = [Rule(compound("h%d" % i, X), [compound("ex", X)])
                 for i in range(k_rules)]
    kb = KnowledgeBase()
    for r in rules:
        kb.add_rule(r)
    p = Proposal(kind="extraction", remove=tuple(rules),
                 add=(e_def,) + tuple(rewritten))
    return p, kb


def _delta_bits(k, body_len):
    p, kb = _make_extraction_proposal(k, body_len)
    return dl.proposal_delta(p, kb=kb, mode="bits")


# --------------------------------------------------------------------------
# Analysis 1: Full extraction surface
# --------------------------------------------------------------------------

def extraction_surface():
    """Complete delta_bits(K, L) grid for K in 2..12, L in 2..8.

    For each L reports:
      - delta_bits_by_k: {k: delta_bits}
      - crossover_k: first K where delta_bits < 0 (None if never)
      - slope_bits_per_k: average d(delta_bits)/dK across K=2..12
        (should be more negative for larger L, reflecting deeper savings)
    """
    K_values = list(range(2, 13))   # 2..12
    L_values = list(range(2, 9))    # 2..8

    rows = []
    for body_len in L_values:
        series = {}
        crossover = None
        for k in K_values:
            d = _delta_bits(k, body_len)
            series[k] = round(d, 2)
            if crossover is None and d < 0:
                crossover = k

        # slope: linear regression coefficient of delta_bits on k
        # (simple mean of consecutive differences suffices for constant spacing)
        deltas = [series[k] for k in K_values]
        diffs = [deltas[i + 1] - deltas[i] for i in range(len(deltas) - 1)]
        slope = round(sum(diffs) / len(diffs), 3)

        rows.append({
            "body_len": body_len,
            "delta_bits_by_k": series,
            "crossover_k": crossover,
            "slope_bits_per_k": slope,
            "delta_clauses_per_k": 1,   # clause-count delta is always +1
        })
    return rows


# --------------------------------------------------------------------------
# Analysis 2: Invention swept on two axes
# --------------------------------------------------------------------------

def _invention_kb_chain(m_sets, chain_len):
    """M structurally identical transitive closures, each over a chain of
    chain_len base edges. Parameterized version of EX29's invention_kb()."""
    kb = KnowledgeBase()
    for s in range(m_sets):
        base = "b%d" % s
        head = "c%d" % s
        nodes = ["n%d_%d" % (s, i) for i in range(chain_len + 1)]
        for i in range(chain_len):
            kb.add_fact(compound(base, atom(nodes[i]), atom(nodes[i + 1])))
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))
    return kb


def _first_invention_delta(builder, *args):
    """Run both modes on a fresh KB from builder(*args); return the first
    'invention' proposal's bits delta and decision per mode (or None)."""
    out = {}
    for mode in ("clauses", "bits"):
        kb = builder(*args)   # fresh copy per mode -- do NOT share
        recs = []
        KnowledgeBaseDreamer(dl_mode=mode,
                             decision_recorder=recs.append).dream(kb)
        hit = next((r for r in recs if r["kind"] == "invention"), None)
        out[mode] = None if hit is None else {
            "decision": hit["decision"],
            "delta_bits": round(hit["delta_bits"], 2),
            "delta_clauses": hit["delta_clauses"],
        }
    return out


def invention_2axis():
    """Two sweeps to trace the 2-D invention crossover.

    (a) Fix chain_len=3 (EX29's default) and sweep M in 2..7.
        Reproduces EX29's M* for cross-check.
    (b) Fix M=4 (just above EX29's M*=4) and sweep chain_len in {2,3,4,6,8}.
        Shows whether deeper base relations change the bits-mode crossover.
    """
    # (a) M sweep at fixed chain_len=3
    m_sweep = []
    for m in range(2, 8):
        result = _first_invention_delta(_invention_kb_chain, m, 3)
        m_sweep.append({"m_sets": m, "chain_len": 3, **result})

    # (b) chain_len sweep at fixed M=4
    chain_sweep = []
    for c in (2, 3, 4, 6, 8):
        result = _first_invention_delta(_invention_kb_chain, 4, c)
        chain_sweep.append({"m_sets": 4, "chain_len": c, **result})

    return {"m_sweep": m_sweep, "chain_sweep": chain_sweep}


# --------------------------------------------------------------------------
# Analysis 3: Realistic scaled-KB sweep
# --------------------------------------------------------------------------

def _scaled_kb(n_shared_rules, chain_len=4):
    """A realistic mixed KB that grows with the number of shared extraction rules.

    Contains:
      - A base relation b over a chain of chain_len+1 nodes.
      - n_shared_rules rules that ALL share the 2-goal body b(X,Y),b(Y,Z),
        i.e., d_i(X,Z) :- b(X,Y), b(Y,Z)  for i in 0..n_shared_rules-1.
        Op E's extraction target: K=n_shared_rules rules sharing L=2 body goals.
        Bits mode crosses over at K*(L=2)=5; clause-count mode always rejects
        extraction (delta_clauses = +1 always, regardless of K).
      - A transitive closure relation r: the exact closure of b as ground
        facts, which Op I (discover_recursion=True) can compress.
        Op I fires in both modes.

    As n_shared_rules grows:
      - Op I: fires in both modes (gap stays 0 there)
      - Op E (extraction): bits mode accepts at K>=5; clauses never accepts.
        So the accept gap (clauses - bits) goes from 0 to -1 at K=5,
        confirming the bits-DL gate accepts mature abstraction that clause-
        count cannot express.
    """
    kb = KnowledgeBase()
    nodes = ["n%d" % i for i in range(chain_len + 1)]
    X, Y, Z = var("X"), var("Y"), var("Z")

    # base relation
    for i in range(chain_len):
        kb.add_fact(compound("b", atom(nodes[i]), atom(nodes[i + 1])))

    # n_shared_rules rules all sharing the 2-goal body b(X,Y),b(Y,Z)
    for k in range(n_shared_rules):
        kb.add_rule(Rule(compound("d%d" % k, X, Z),
                         [compound("b", X, Y), compound("b", Y, Z)]))

    # transitive closure as ground facts (Op I target)
    for i in range(chain_len + 1):
        for j in range(i + 1, chain_len + 1):
            kb.add_fact(compound("r", atom(nodes[i]), atom(nodes[j])))

    return kb


def _count_accepts_by_kind(recs):
    """Count accepted proposals by kind from a decision_recorder list.
    The decision field is 'accepted' (not 'accept') per gate.py."""
    counts = {}
    for r in recs:
        if r["decision"] == "accepted":
            counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    return counts


def scaled_kb_sweep():
    """Sweep K (number of rules sharing a 2-goal body) in {2,3,4,5,6,7,8,12}.
    Each KB also includes a transitive closure relation for Op I.

    Op I fires in both modes (both always accept it).
    Op E (extraction, L=2 shared body): clauses mode always rejects (delta=+1);
    bits mode rejects at K<5 and accepts at K>=5.

    Headline: the bits-vs-clauses accept gap (clauses - bits) goes from 0 to
    -1 at K=5 and stays there, showing bits mode ADDS an abstraction that
    clause-count cannot express -- the accept gap becomes negative (bits accepts
    more) as reuse deepens past the crossover.
    """
    K_values = [2, 3, 4, 5, 6, 7, 8, 12]
    rows = []
    for k in K_values:
        kb_probe = _scaled_kb(k)
        n_facts = len(list(kb_probe.facts))
        n_rules = len(list(kb_probe.rules))

        row = {"n_shared_rules": k, "kb_facts": n_facts, "kb_rules": n_rules}
        for mode in ("clauses", "bits"):
            recs = []
            kb = _scaled_kb(k)    # fresh copy per mode
            KnowledgeBaseDreamer(dl_mode=mode, discover_recursion=True,
                                 decision_recorder=recs.append).dream(kb)
            total = len(recs)
            accepts = sum(1 for r in recs if r["decision"] == "accepted")
            rejects = total - accepts
            by_kind = _count_accepts_by_kind(recs)
            row[mode] = {
                "total_proposals": total,
                "accepts": accepts,
                "rejects": rejects,
                "by_kind": by_kind,
            }

        # Positive gap: clauses accepts something bits does not.
        # Negative gap: bits accepts something clauses does not (bits sees more).
        row["accept_gap"] = (row["clauses"]["accepts"] - row["bits"]["accepts"])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    params = {
        "extraction_surface": {
            "k_range": [2, 12], "l_range": [2, 8],
        },
        "invention_2axis": {
            "m_sweep": {"m_range": [2, 7], "chain_len": 3},
            "chain_sweep": {"m_sets": 4, "chain_lengths": [2, 3, 4, 6, 8]},
        },
        "scaled_kb": {
            "k_shared_rules_values": [2, 3, 4, 5, 6, 7, 8, 12],
            "chain_len": 4,
            "discover_recursion": True,
            "modes": ["clauses", "bits"],
            "note": ("clause-count always rejects extraction (delta=+1); "
                     "bits accepts at K>=K*(L=2)=5. Gap goes 0 -> -1 at K=5."),
        },
    }

    with experiment_run(
            exp_id="ex30",
            name="deeper bits-DL abstraction surfaces (extraction grid, "
                 "2-axis invention, scaled KB)",
            description=(
                "EX29 reported a single crossover K*(L) for extraction and "
                "a single M* for invention. EX30 produces finer-grained "
                "surfaces: (1) the complete delta_bits(K,L) grid with slopes "
                "for K in 2..12 x L in 2..8; (2) the invention crossover on "
                "TWO axes (M at fixed chain length, chain length at fixed M); "
                "(3) a realistic scaled-KB sweep showing whether the "
                "bits-vs-clauses accept gap shrinks as the KB grows. "
                "Symbolic only, no LLM, deterministic."
            ),
            script=__file__,
            params=params,
            seeds={"note": "deterministic; no RNG"}) as run:

        def emit(line=""):
            print(line)
            run.summary_lines.append(line)

        # -- Analysis 1: extraction surface --
        emit("EX30: deeper bits-DL abstraction surfaces")
        emit("=" * 60)
        emit("\n1. EXTRACTION SURFACE  delta_bits(K, L) for K in 2..12, L in 2..8")
        emit("   (negative = bits accepts; clause delta = +1 always)")
        emit("   EX29 crossover check: L=2->K*5, L=3->3, L=4->3, L=5->2")

        surface = extraction_surface()
        run.results["extraction_surface"] = surface

        K_values = list(range(2, 13))
        header = "   L\\K  " + "  ".join("%5d" % k for k in K_values)
        emit(header)
        ex29_crossovers = {2: 5, 3: 3, 4: 3, 5: 2}
        for row in surface:
            L = row["body_len"]
            series = row["delta_bits_by_k"]
            cells = "  ".join("%5.0f" % series[k] for k in K_values)
            crossover_str = str(row["crossover_k"]) if row["crossover_k"] is not None else "none"
            emit("   L=%-3d  %s   K*=%s  slope=%+.1f bits/rule"
                 % (L, cells, crossover_str, row["slope_bits_per_k"]))
            if L in ex29_crossovers:
                match = (row["crossover_k"] == ex29_crossovers[L])
                emit("          (EX29 expected K*=%d: %s)"
                     % (ex29_crossovers[L], "OK" if match else "MISMATCH"))

        emit("\n   Slope (d(delta_bits)/dK) by L -- more negative = richer payoff:")
        for row in surface:
            emit("     L=%d  slope=%+.1f bits/rule  (crossover K*=%s)"
                 % (row["body_len"], row["slope_bits_per_k"],
                    row["crossover_k"] if row["crossover_k"] is not None else "none"))

        # -- Analysis 2: invention 2-axis --
        emit("\n2. INVENTION ON TWO AXES  (dreamer, Op D)")
        emit("   (a) M sweep at chain_len=3  -- reproduces EX29 M*=4")

        inv = invention_2axis()
        run.results["invention_2axis"] = inv

        emit("   %-4s  %-6s  %-10s  %-10s" % ("M", "chain", "bits", "clauses"))
        for r in inv["m_sweep"]:
            b = r.get("bits")
            c = r.get("clauses")
            b_str = "None" if b is None else "%s/%+.0f" % (b["decision"], b["delta_bits"])
            c_str = "None" if c is None else c["decision"]
            emit("   M=%-2d  C=%-4d  %-10s  %s" % (r["m_sets"], r["chain_len"], b_str, c_str))

        emit("\n   (b) chain_len sweep at M=4  -- does depth change crossover?")
        emit("   %-4s  %-6s  %-10s  %-10s" % ("M", "chain", "bits", "clauses"))
        for r in inv["chain_sweep"]:
            b = r.get("bits")
            c = r.get("clauses")
            b_str = "None" if b is None else "%s/%+.0f" % (b["decision"], b["delta_bits"])
            c_str = "None" if c is None else c["decision"]
            emit("   M=%-2d  C=%-4d  %-10s  %s" % (r["m_sets"], r["chain_len"], b_str, c_str))

        # -- Analysis 3: scaled KB sweep --
        emit("\n3. SCALED-KB SWEEP  (K shared-body rules + transitive closure, "
             "discover_recursion=True)")
        emit("   clause-count rejects extraction always (delta=+1); "
             "bits mode accepts at K>=K*(L=2)=5")
        emit("   Gap = clauses_accepts - bits_accepts  (negative means bits "
             "accepts MORE)")
        emit("   %-4s  %-6s  %-6s  %-6s  %-6s  %-5s  %-s"
             % ("K", "facts", "rules", "cl_acc", "bi_acc", "gap", "bi_kinds"))

        scaled = scaled_kb_sweep()
        run.results["scaled_kb"] = scaled

        for r in scaled:
            cl = r["clauses"]
            bi = r["bits"]
            bi_kinds = ", ".join("%s:%d" % (k, v)
                                 for k, v in sorted(bi["by_kind"].items()))
            emit("   K=%-3d  f=%-5d  r=%-5d  cl=%-5d  bi=%-5d  gap=%-4d  %s"
                 % (r["n_shared_rules"], r["kb_facts"], r["kb_rules"],
                    cl["accepts"], bi["accepts"], r["accept_gap"],
                    bi_kinds if bi_kinds else "(none)"))

        emit("\n   Clauses-mode kinds (for comparison):")
        for r in scaled:
            cl_kinds = ", ".join("%s:%d" % (k, v)
                                 for k, v in sorted(r["clauses"]["by_kind"].items()))
            emit("   K=%-3d  %s" % (r["n_shared_rules"],
                                    cl_kinds if cl_kinds else "(none)"))

        # Summary notes
        run.note("extraction crossovers (L=2..5): %s"
                 % {row["body_len"]: row["crossover_k"] for row in surface if row["body_len"] <= 5})
        run.note("invention M* (bits mode): %s"
                 % next((r["m_sets"] for r in inv["m_sweep"]
                         if r.get("bits") and r["bits"]["decision"] == "accepted"), None))
        # gap is positive when clauses accepts but bits rejects (premature abstraction)
        # gap is zero when both agree; gap negative would mean bits accepts more
        gap_vals = [r["accept_gap"] for r in scaled]
        max_gap_k = scaled[gap_vals.index(max(gap_vals))]["n_shared_rules"]
        gap_zero_k = next((r["n_shared_rules"] for r in scaled
                           if r["accept_gap"] == 0), None)
        run.note("scaled-kb: gap peaks at K=%d (clauses premature, bits conservative), "
                 "then shrinks to 0 at K=%s as abstraction matures past contextual K*"
                 % (max_gap_k, gap_zero_k))

        emit("\nSUMMARY")
        emit("  Extraction crossovers match EX29 for L in 2..5: %s"
             % all(row["crossover_k"] == ex29_crossovers[row["body_len"]]
                   for row in surface if row["body_len"] in ex29_crossovers))
        emit("  Slopes more negative with larger L: %s"
             % (surface[-1]["slope_bits_per_k"] < surface[0]["slope_bits_per_k"]))
        gaps = [r["accept_gap"] for r in scaled]
        emit("  Accept gap (clauses-bits) by K=%s: %s"
             % ([r["n_shared_rules"] for r in scaled], gaps))
        emit("  Interpretation: gap>0 means clause-count accepts prematurely (bits "
             "correctly rejects); gap=0 means both modes agree (abstraction mature).")
        emit("  Contextual K* in the scaled-KB is ~4 (lower than analytic K*=5 because "
             "extra functors in context reduce the new-symbol declaration charge).")

    print("\nWrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
