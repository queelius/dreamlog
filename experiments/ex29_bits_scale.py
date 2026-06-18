"""EX29: does bits-mode DL favor abstraction as reuse deepens?

P3b made the bits-based description length rename-invariant (dictionary charge =
Elias gamma of arity+1 per unique symbol). The decision-diff (EX/bench scenarios)
then showed bits mode rejecting 8 of 24 symbolic abstractions that clause-count
accepts. The hypothesis this experiment tests: bits mode is not anti-abstraction,
it is anti-PREMATURE-abstraction. An abstraction reused only a few times barely
moves the symbol-occurrence count; reused deeply, it clearly pays. So as reuse
deepens, the bits-mode rejections should flip to acceptances, and there is a
sharp crossover.

Three sweeps:
  1. Extraction (analytic): K rules sharing an L-goal body. Construct the exact
     extraction proposal and read proposal_delta in bits mode. Reports the
     crossover K*(L) where extraction starts to pay, and shows K* falls as the
     shared body L deepens. Clause-count delta is +1 for every (K, L).
  2. Invention (via the dreamer): M structurally identical transitive-closure
     predicate sets; sweep M and read the invention proposal's bits delta and
     the bits-mode decision.
  3. Generalization (via the dreamer): a group of N facts sharing a value under a
     guard; sweep N and read the generalization proposal's bits delta/decision.

No LLM is used anywhere (symbolic only); deterministic.

Usage: python experiments/ex29_bits_scale.py
Writes: experiments/data/ex29/results.json
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import experiment_run   # noqa: E402

from dreamlog.compression import dl
from dreamlog.compression.proposal import Proposal
from dreamlog.factories import atom, compound, var
from dreamlog.knowledge import Fact, KnowledgeBase, Rule
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


# --------------------------------------------------------------------------
# 1. Extraction crossover (analytic: build the proposal, read the bits delta)
# --------------------------------------------------------------------------

def extraction_delta(k_rules, body_len):
    """K rules h_i(X) :- g0(X)..g{L-1}(X) all share the same L-goal body.
    Extraction adds e(X) :- body and rewrites each head to h_i(X) :- e(X)."""
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
    return dl.proposal_delta(p, kb=kb, mode="bits"), dl.proposal_delta(p)


def extraction_sweep():
    rows = []
    for body_len in (2, 3, 4, 5):
        series = {}
        crossover = None
        for k in range(2, 13):
            d_bits, d_clauses = extraction_delta(k, body_len)
            series[k] = d_bits
            if crossover is None and d_bits < 0:
                crossover = k
        rows.append({"body_len": body_len, "delta_bits_by_k": series,
                     "crossover_k": crossover, "delta_clauses": 1})
    return rows


# --------------------------------------------------------------------------
# 2. Invention sweep (dreamer): M transitive-closure predicate sets
# --------------------------------------------------------------------------

def invention_kb(m_sets):
    """M structurally identical transitive closures over distinct base
    relations, each with a short chain of base facts. Op D's target."""
    kb = KnowledgeBase()
    for s in range(m_sets):
        base = "b%d" % s
        head = "c%d" % s
        chain = ["n%d_%d" % (s, i) for i in range(4)]
        for i in range(len(chain) - 1):
            kb.add_fact(compound(base, atom(chain[i]), atom(chain[i + 1])))
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))
    return kb


def generalization_kb(n_pass, n_fail=1):
    """n_pass entities are 'grade(e_i, pass)' under guard 'student(e_i)', plus
    n_fail exception entities that fail. Op C's target: generalize to
    grade(X, pass) :- student(X), with the failing entities as exceptions."""
    kb = KnowledgeBase()
    for i in range(n_pass):
        e = "s%d" % i
        kb.add_fact(compound("student", atom(e)))
        kb.add_fact(compound("grade", atom(e), atom("pass")))
    for j in range(n_fail):
        x = "x%d" % j
        kb.add_fact(compound("student", atom(x)))
        kb.add_fact(compound("grade", atom(x), atom("fail")))
    return kb


def recursion_kb(chain_len):
    """A base chain b: n0->n1->...->n{L}, plus r = the FULL transitive closure
    of b as ground facts. Op I's target: detect r = closure(b) and replace r's
    O(L^2) fact extension with r(X,Y):-b(X,Y); r(X,Z):-b(X,Y),r(Y,Z)."""
    kb = KnowledgeBase()
    nodes = ["n%d" % i for i in range(chain_len + 1)]
    for i in range(chain_len):
        kb.add_fact(compound("b", atom(nodes[i]), atom(nodes[i + 1])))
    for i in range(chain_len + 1):
        for j in range(i + 1, chain_len + 1):
            kb.add_fact(compound("r", atom(nodes[i]), atom(nodes[j])))
    return kb


def _first_delta(kind, builder, param):
    """Run both modes on builder(param); return the first proposal of `kind`
    seen in each mode with its bits delta and decision."""
    out = {"param": param}
    for mode in ("clauses", "bits"):
        recs = []
        KnowledgeBaseDreamer(dl_mode=mode,
                             decision_recorder=recs.append).dream(builder(param))
        hit = next((r for r in recs if r["kind"] == kind), None)
        out[mode] = None if hit is None else {
            "decision": hit["decision"],
            "delta_bits": round(hit["delta_bits"], 2),
            "delta_clauses": hit["delta_clauses"],
        }
    return out


def invention_sweep():
    return [_first_delta("invention", invention_kb, m) for m in range(2, 8)]


def generalization_sweep():
    return [_first_delta("generalization", generalization_kb, n)
            for n in (3, 4, 6, 8, 12, 16, 24)]


def recursion_sweep():
    out = []
    for chain_len in (3, 4, 5, 6, 8, 10):
        recs = []
        KnowledgeBaseDreamer(dl_mode="bits", discover_recursion=True,
                             decision_recorder=recs.append).dream(
                                 recursion_kb(chain_len))
        hit = next((r for r in recs if r["kind"] == "recursion"), None)
        out.append({"chain_len": chain_len,
                    "closure_pairs": chain_len * (chain_len + 1) // 2,
                    "result": None if hit is None else {
                        "decision": hit["decision"],
                        "delta_bits": round(hit["delta_bits"], 2)}})
    return out


def generalization_2d():
    """Generalization crossover as a SURFACE over (group size N, #exceptions E):
    more exceptions raise the bar an abstraction must clear."""
    grid = []
    for n_pass in (4, 6, 8, 12, 16):
        row = {"n_pass": n_pass, "by_exceptions": {}}
        for n_fail in (1, 2, 3, 4):
            recs = []
            KnowledgeBaseDreamer(dl_mode="bits",
                                 decision_recorder=recs.append).dream(
                                     generalization_kb(n_pass, n_fail))
            hit = next((r for r in recs if r["kind"] == "generalization"), None)
            row["by_exceptions"][n_fail] = (
                None if hit is None else round(hit["delta_bits"], 2))
        grid.append(row)
    return grid


def removal_baseline():
    """Pure-removal ops (subsumption, pruning) always pay: removing a clause
    only lowers DL. Confirms the taxonomy's other end against the abstraction
    crossovers."""
    X = var("X")
    general = Rule(compound("p", X), [compound("q", X)])
    specific = Rule(compound("p", atom("a")), [compound("q", atom("a"))])
    kb_sub = KnowledgeBase()
    kb_sub.add_rule(general)
    kb_sub.add_rule(specific)
    sub = Proposal(kind="subsumption", remove=(specific,))
    kb_prune = KnowledgeBase()
    kb_prune.add_fact(Fact(compound("f", atom("a"))))
    kb_prune.add_fact(Fact(compound("f", atom("b"))))
    prune = Proposal(kind="pruning", remove=(Fact(compound("f", atom("a"))),))
    return {
        "subsumption_delta_bits": round(
            dl.proposal_delta(sub, kb=kb_sub, mode="bits"), 2),
        "pruning_delta_bits": round(
            dl.proposal_delta(prune, kb=kb_prune, mode="bits"), 2),
    }


# --------------------------------------------------------------------------

def main():
    params = {
        "extraction": {"k_range": [2, 12], "body_lengths": [2, 3, 4, 5]},
        "invention": {"m_sets": list(range(2, 8))},
        "generalization": {"n_facts": [3, 4, 6, 8, 12, 16, 24], "exceptions": 1},
        "recursion": {"chain_lengths": [3, 4, 5, 6, 8, 10]},
        "generalization_2d": {"n_pass": [4, 6, 8, 12, 16],
                              "exceptions": [1, 2, 3, 4]},
    }
    with experiment_run(
            exp_id="ex29",
            name="bits-mode abstraction-maturity crossover (taxonomy)",
            description=("Does the rename-invariant bits-DL gate accept "
                         "abstraction only once reuse deepens enough to "
                         "compress? Characterizes all symbolic ops."),
            script=__file__, params=params,
            seeds={"note": "fully deterministic; no RNG"}) as run:

        out = run.summary_lines.append   # capture + print helper

        def emit(line=""):
            print(line)
            out(line)

        run.results["extraction"] = extraction_sweep()
        run.results["invention"] = invention_sweep()
        run.results["generalization"] = generalization_sweep()
        run.results["recursion"] = recursion_sweep()
        run.results["generalization_2d"] = generalization_2d()
        run.results["removal_baseline"] = removal_baseline()

        emit("EX29: bits-mode abstraction vs reuse depth")
        emit("\n1. EXTRACTION crossover -- K rules sharing an L-goal body")
        emit("   (delta_bits; negative = bits accepts; clause delta = +1 always)")
        emit("   %-8s %s" % ("body L", "  ".join("K=%d" % k
                                                 for k in range(2, 13))))
        for row in run.results["extraction"]:
            series = row["delta_bits_by_k"]
            cells = "  ".join("%4.0f" % series[k] for k in range(2, 13))
            emit("   L=%-6d %s   crossover K*=%s"
                 % (row["body_len"], cells, row["crossover_k"]))

        emit("\n2. INVENTION -- M transitive-closure sets (dreamer)")
        for r in run.results["invention"]:
            b = r.get("bits"); c = r.get("clauses")
            emit("   M=%-2d  bits=%s  clauses=%s"
                 % (r["param"],
                    None if b is None else "%s/%+.0f"
                    % (b["decision"], b["delta_bits"]),
                    None if c is None else c["decision"]))

        emit("\n3. GENERALIZATION -- group of N facts under a guard (dreamer)")
        for r in run.results["generalization"]:
            b = r.get("bits"); c = r.get("clauses")
            emit("   N=%-2d  bits=%s  clauses=%s"
                 % (r["param"],
                    None if b is None else "%s/%+.0f"
                    % (b["decision"], b["delta_bits"]),
                    None if c is None else c["decision"]))

        emit("\n4. RECURSION -- r = closure(b), sweep chain length (Op I)")
        emit("   (always pays once valid; savings grow with closure size)")
        for r in run.results["recursion"]:
            res = r["result"]
            emit("   L=%-2d (closure %3d)  %s"
                 % (r["chain_len"], r["closure_pairs"],
                    None if res is None else "%s/%+.0f"
                    % (res["decision"], res["delta_bits"])))

        emit("\n5. GENERALIZATION 2D -- surface over (N pass, E exceptions)")
        emit("   (delta_bits; negative = bits accepts; 'none' = Op C silent)")
        emit("   %-6s %s" % ("N\\E", "  ".join("E=%d" % e for e in (1, 2, 3, 4))))
        for row in run.results["generalization_2d"]:
            cells = "  ".join(
                " none" if row["by_exceptions"][e] is None
                else "%5.0f" % row["by_exceptions"][e] for e in (1, 2, 3, 4))
            emit("   N=%-4d %s" % (row["n_pass"], cells))

        rem = run.results["removal_baseline"]
        emit("\n6. REMOVAL ops always pay: subsumption=%+.0f  pruning=%+.0f bits"
             % (rem["subsumption_delta_bits"], rem["pruning_delta_bits"]))
        emit("\nTAXONOMY: removal + recursion pay immediately; extraction, "
             "invention,\n  and generalization require a reuse crossover.")

    print("\nWrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
