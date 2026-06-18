"""EX34: CROSS-DOMAIN TRANSFER -- does compressing a merged multi-domain KB
discover abstractions SHARED across domains?

HYPOTHESIS: Two or more domains with the SAME relational skeleton but DIFFERENT
vocabulary (e.g. c0 = transitive closure of b0; c1 = transitive closure of b1)
share deep structure. Compressing the MERGED KB (all domains together) should
cause Op D (predicate invention) to unify the structurally identical rule sets
into ONE invented predicate dispatched via call/N -- a shared abstraction that
compressing each domain separately never produces. Because the bits code is
RENAME-INVARIANT (it prices structure, not symbol names), it should especially
reward this shared invention, and more co-domains sharing the skeleton should
make the shared abstraction pay increasingly (crossover, echoing EX29 M*=4).

THREE CONDITIONS per (M, mode):
  (a) SEPARATE: dream each domain alone; sum clause/bits costs across domains;
      record whether any invented predicate appears in any domain's result.
  (b) MERGED: dream the union of all M domains in one KB.
  (c) COMPARE: did MERGED discover a SHARED invented predicate? (one _invented_
      functor referenced by rules from >=2 domains.) Measure the super-additivity
      delta: merged compression vs sum-of-separate.

SWEEP: M in {2, 3, 4, 5, 6}.
MODES: "clauses" and "bits".
SYMBOLIC ONLY -- no LLM, deterministic.

Detection method for "shared abstraction": after a merged dream, inspect kb.rules
for any rule whose head functor contains "_invented_". For each such invented
functor, find the wrapper rules that call it (rules whose body contains the
invented functor or call/N dispatching to it). Check whether those wrapper rules
cover heads from >=2 distinct domain vocabularies (c0, c1, ...). If yes, the
merged dream produced a shared abstraction. The wrapper rules are the original
domain heads (c0, c1, ...) rewritten to delegate to the invented predicate; their
head functors name the domain they belong to, so domain membership is unambiguous.

Usage: python experiments/ex34_transfer.py
Writes: experiments/data/ex34/runs/<id>/{meta.json,results.json,summary.txt}
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from _harness import experiment_run                            # noqa: E402
from dreamlog.compression import dl                            # noqa: E402
from dreamlog.factories import atom, compound, var             # noqa: E402
from dreamlog.knowledge import Fact, KnowledgeBase, Rule       # noqa: E402
from dreamlog.kb_dreamer import KnowledgeBaseDreamer           # noqa: E402


# ---------------------------------------------------------------------------
# KB builders
# ---------------------------------------------------------------------------

def _domain_kb(domain_idx: int, chain_len: int = 4) -> KnowledgeBase:
    """One domain: base relation b<k> over a chain of 4 entities, plus two
    closure rules defining c<k> as the transitive closure of b<k>.

    Also adds a small within-domain generalizable group: tag_<k>(e, kind<k>)
    for the first 3 entities -- gives Op C something to find per-domain.
    """
    k = domain_idx
    base = "b%d" % k
    head = "c%d" % k
    tag = "tag%d" % k
    kind_val = "kind%d" % k

    kb = KnowledgeBase()
    nodes = ["e%d_%d" % (k, i) for i in range(chain_len)]
    for i in range(chain_len - 1):
        kb.add_fact(Fact(compound(base, atom(nodes[i]), atom(nodes[i + 1]))))

    # Closure rules for c<k>
    X, Y, Z = var("X"), var("Y"), var("Z")
    kb.add_rule(Rule(compound(head, X, Y), [compound(base, X, Y)]))
    kb.add_rule(Rule(compound(head, X, Z),
                     [compound(base, X, Y), compound(head, Y, Z)]))

    # Within-domain generalizable facts: tag<k>(e, kind<k>) for first 3 nodes
    for i in range(min(3, chain_len)):
        kb.add_fact(Fact(compound("entity%d" % k, atom(nodes[i]))))
        kb.add_fact(Fact(compound(tag, atom(nodes[i]), atom(kind_val))))

    return kb


def _merged_kb(m_domains: int, chain_len: int = 4) -> KnowledgeBase:
    """Union of M domain KBs, all poured into one KnowledgeBase."""
    merged = KnowledgeBase()
    for k in range(m_domains):
        dom = _domain_kb(k, chain_len=chain_len)
        for f in dom.facts:
            merged.add_fact(f)
        for r in dom.rules:
            merged.add_rule(r)
    return merged


# ---------------------------------------------------------------------------
# Shared-abstraction detection
# ---------------------------------------------------------------------------

def _domain_head_functors(m_domains: int) -> set:
    """The set of closure-head functor names c0..c{M-1}."""
    return {"c%d" % k for k in range(m_domains)}


def _detect_shared_abstraction(kb: KnowledgeBase, m_domains: int) -> dict:
    """Inspect the dreamed merged KB for a shared invented predicate.

    A "shared abstraction" exists when there is at least one invented predicate
    (functor containing "_invented_") such that the wrapper rules dispatching TO
    it cover >= 2 distinct domain closure heads (c0, c1, ...).

    Detection method:
      1. Collect all invented functor names from rule heads in the final KB.
      2. For each invented functor, find every rule in the KB whose body
         references it (these are the "wrapper" rules that domain heads use
         to delegate to the invention).
      3. Record the head functors of those wrapper rules.
      4. If any wrapper set spans >=2 domain closure heads -> shared abstraction.

    Also accepts the indirect call/N dispatch pattern: the invented predicate
    is defined over (R, X, Y) where R is the relation name atom, and wrapper
    rules for c<k>(X,Y) call invented(b<k>, X, Y) via call/N.
    """
    from dreamlog.terms import Compound, Atom

    domain_heads = _domain_head_functors(m_domains)

    # Collect invented functor names
    invented_functors = set()
    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            f = rule.head.functor
            if "_invented_" in f:
                invented_functors.add(f)

    if not invented_functors:
        return {
            "found": False,
            "invented_functors": [],
            "shared_invented": None,
            "domains_covered": [],
            "n_domains_covered": 0,
        }

    # For each invented functor, check which domain closure heads wrap it
    for inv_f in sorted(invented_functors):
        covered_domains = set()
        for rule in kb.rules:
            if isinstance(rule.head, Compound):
                head_f = rule.head.functor
                if head_f in domain_heads:
                    # Check body for reference to the invented functor
                    for goal in rule.body:
                        if isinstance(goal, Compound):
                            if goal.functor == inv_f:
                                covered_domains.add(head_f)
                                break
                            # call/N pattern: call(inv_f_atom, ...)
                            if goal.functor == "call" and goal.args:
                                first_arg = goal.args[0]
                                if (isinstance(first_arg, Atom)
                                        and first_arg.value == inv_f):
                                    covered_domains.add(head_f)
                                    break

        if len(covered_domains) >= 2:
            return {
                "found": True,
                "invented_functors": sorted(invented_functors),
                "shared_invented": inv_f,
                "domains_covered": sorted(covered_domains),
                "n_domains_covered": len(covered_domains),
            }

    return {
        "found": False,
        "invented_functors": sorted(invented_functors),
        "shared_invented": None,
        "domains_covered": [],
        "n_domains_covered": 0,
    }


# ---------------------------------------------------------------------------
# Per-condition measurements
# ---------------------------------------------------------------------------

def _measure_kb(kb: KnowledgeBase) -> dict:
    """Return clause count and bits DL for a KB."""
    return {
        "clauses": len(kb),
        "bits": round(dl.description_length(kb, mode="bits"), 2),
    }


def _dream_one(kb: KnowledgeBase, mode: str) -> dict:
    """Dream a single KB in the given mode; return before/after stats + session."""
    before = _measure_kb(kb)
    records = []
    dreamer = KnowledgeBaseDreamer(
        dl_mode=mode,
        decision_recorder=records.append,
        llm_client=None,
        min_group_size=3,
    )
    session = dreamer.dream(kb)
    after = _measure_kb(kb)

    # Collect accepted ops by kind
    accepted_by_kind = {}
    for r in records:
        if r.get("decision") == "accepted":
            k = r["kind"]
            accepted_by_kind[k] = accepted_by_kind.get(k, 0) + 1

    return {
        "before": before,
        "after": after,
        "clauses_saved": before["clauses"] - after["clauses"],
        "bits_saved": round(before["bits"] - after["bits"], 2),
        "accepted_by_kind": accepted_by_kind,
        "compressed": session.compressed,
    }


def _run_separate(m_domains: int, mode: str) -> dict:
    """SEPARATE condition: dream each domain alone; sum compression."""
    total_before = {"clauses": 0, "bits": 0.0}
    total_after = {"clauses": 0, "bits": 0.0}
    any_invented = False
    per_domain = []

    for k in range(m_domains):
        kb = _domain_kb(k)
        result = _dream_one(kb, mode)
        per_domain.append(result)
        total_before["clauses"] += result["before"]["clauses"]
        total_before["bits"] += result["before"]["bits"]
        total_after["clauses"] += result["after"]["clauses"]
        total_after["bits"] += result["after"]["bits"]
        if result["accepted_by_kind"].get("invention", 0) > 0:
            any_invented = True

    total_after["bits"] = round(total_after["bits"], 2)

    return {
        "condition": "separate",
        "mode": mode,
        "m_domains": m_domains,
        "total_before": {"clauses": total_before["clauses"],
                         "bits": round(total_before["bits"], 2)},
        "total_after": {"clauses": total_after["clauses"],
                        "bits": round(total_after["bits"], 2)},
        "sum_clauses_saved": total_before["clauses"] - total_after["clauses"],
        "sum_bits_saved": round(total_before["bits"] - total_after["bits"], 2),
        "any_invented": any_invented,
        "shared_abstraction_possible": False,  # impossible in separate condition
        "per_domain": per_domain,
    }


def _run_merged(m_domains: int, mode: str) -> dict:
    """MERGED condition: dream the union of M domains as one KB."""
    kb = _merged_kb(m_domains)
    result = _dream_one(kb, mode)
    detection = _detect_shared_abstraction(kb, m_domains)

    return {
        "condition": "merged",
        "mode": mode,
        "m_domains": m_domains,
        "before": result["before"],
        "after": result["after"],
        "clauses_saved": result["clauses_saved"],
        "bits_saved": result["bits_saved"],
        "accepted_by_kind": result["accepted_by_kind"],
        "compressed": result["compressed"],
        "shared_abstraction": detection,
    }


# ---------------------------------------------------------------------------
# Full sweep: M in M_VALUES, both modes
# ---------------------------------------------------------------------------

M_VALUES = [2, 3, 4, 5, 6]


def _run_sweep() -> list:
    """Sweep over M and modes; return list of row dicts (one per M x mode)."""
    rows = []
    for m in M_VALUES:
        for mode in ("clauses", "bits"):
            sep = _run_separate(m, mode)
            mer = _run_merged(m, mode)

            # Super-additivity: merged compresses MORE than sum-of-separate
            super_add_clauses = mer["clauses_saved"] - sep["sum_clauses_saved"]
            super_add_bits = round(mer["bits_saved"] - sep["sum_bits_saved"], 2)

            row = {
                "m": m,
                "mode": mode,
                # Merged stats
                "merged_before_clauses": mer["before"]["clauses"],
                "merged_after_clauses": mer["after"]["clauses"],
                "merged_clauses_saved": mer["clauses_saved"],
                "merged_before_bits": mer["before"]["bits"],
                "merged_after_bits": mer["after"]["bits"],
                "merged_bits_saved": mer["bits_saved"],
                # Sum-of-separate stats
                "sep_total_before_clauses": sep["total_before"]["clauses"],
                "sep_total_after_clauses": sep["total_after"]["clauses"],
                "sep_clauses_saved": sep["sum_clauses_saved"],
                "sep_total_before_bits": sep["total_before"]["bits"],
                "sep_total_after_bits": sep["total_after"]["bits"],
                "sep_bits_saved": sep["sum_bits_saved"],
                # Transfer signal
                "super_add_clauses": super_add_clauses,
                "super_add_bits": super_add_bits,
                # Shared abstraction
                "shared_abstraction_found": mer["shared_abstraction"]["found"],
                "shared_invented_functor": mer["shared_abstraction"]["shared_invented"],
                "domains_covered_by_shared": mer["shared_abstraction"]["n_domains_covered"],
                # Separate: any per-domain invention?
                "sep_any_invented": sep["any_invented"],
                # Accepted ops breakdown (merged)
                "merged_accepted_by_kind": mer["accepted_by_kind"],
            }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _shared_marker(found: bool) -> str:
    return "YES" if found else "no"


def _table_lines(rows: list) -> list:
    """Return human-readable table lines for the sweep."""
    lines = []
    header = ("%-4s  %-7s  %7s %7s %7s  %7s %7s %7s  %7s %7s  %-6s"
              % ("M", "mode",
                 "mer-sav", "sep-sav", "super+c",
                 "mer-bit", "sep-bit", "super+b",
                 "sh-cl", "sh-bi",
                 "shared"))
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            "%-4d  %-7s  %7d %7d %7d  %7.1f %7.1f %7.1f  %7s %7s  %-6s"
            % (r["m"], r["mode"],
               r["merged_clauses_saved"], r["sep_clauses_saved"], r["super_add_clauses"],
               r["merged_bits_saved"], r["sep_bits_saved"], r["super_add_bits"],
               "-", "-",  # crossover columns populated separately
               _shared_marker(r["shared_abstraction_found"])))
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    params = {
        "M_values": M_VALUES,
        "modes": ["clauses", "bits"],
        "chain_len": 4,
        "llm": False,
        "conditions": ["separate", "merged"],
        "detection_method": (
            "After merged dream, inspect kb.rules for any rule head containing "
            "'_invented_'. For each invented functor, find wrapper rules whose "
            "body references it (direct or via call/N dispatch). A shared "
            "abstraction is found when the wrapper rules' head functors span "
            ">=2 distinct domain closure heads (c0..c{M-1})."
        ),
    }

    with experiment_run(
        exp_id="ex34",
        name="cross-domain transfer via merged-KB predicate invention",
        description=(
            "Does compressing M merged single-skeleton domains (each with distinct "
            "vocabulary but same transitive-closure structure) produce a SHARED "
            "abstraction (one _invented_ predicate covering all domains) that "
            "separate per-domain dreams cannot? Sweeps M in {2,3,4,5,6} under "
            "both clause-count and bits DL modes. Symbolic only, no LLM. "
            "Measures super-additivity of merged vs sum-of-separate compression "
            "and whether bits mode favors the shared abstraction more than clauses."
        ),
        script=__file__,
        params=params,
        seeds={"note": "fully deterministic; no RNG"},
    ) as run:

        def emit(line=""):
            print(line)
            run.summary_lines.append(line)

        emit("EX34: Cross-Domain Transfer via Merged-KB Predicate Invention")
        emit("=" * 72)
        emit("SYMBOLIC ONLY -- no LLM, zero cost, deterministic")
        emit("")
        emit("HYPOTHESIS: compressing M domains merged together discovers a SHARED")
        emit("  _invented_ predicate (Op D) spanning all domains, which separate")
        emit("  per-domain dreams cannot produce. Bits mode (rename-invariant DL)")
        emit("  should accept the shared abstraction at fewer domains (lower M*).")
        emit("")
        emit("DOMAIN DESIGN: each domain k has:")
        emit("  base relation b<k>(X,Y) over a 4-entity chain")
        emit("  closure relation c<k> defined by two rules: c<k>(X,Y):-b<k>(X,Y)")
        emit("    and c<k>(X,Z):-b<k>(X,Y),c<k>(Y,Z)  (transitive closure)")
        emit("  within-domain generalizable group: tag<k>(e,kind<k>) for 3 entities")
        emit("")
        emit("DETECTION METHOD: after merged dream, find _invented_ functor in")
        emit("  rule heads; find wrapper rules calling it; if wrappers span >=2")
        emit("  distinct domain closure heads (c0,c1,...) -> shared abstraction.")
        emit("")

        emit("Running sweep (M x mode)...")
        rows = _run_sweep()
        run.results["sweep"] = rows
        run.results["detection_method"] = params["detection_method"]

        # Summary table
        emit("\nSWEEP TABLE")
        emit("  Columns: mer-sav=merged clauses saved, sep-sav=sum-separate clauses saved,")
        emit("    super+c=super-additive clause savings (merged-sep), mer-bit/sep-bit same for bits,")
        emit("    super+b=super-additive bits savings, shared=YES if shared abstraction found")
        emit("")
        emit("  %-4s  %-7s  %7s %7s %7s  %7s %7s %7s  %-6s"
             % ("M", "mode",
                "mer-cl", "sep-cl", "sup-cl",
                "mer-bt", "sep-bt", "sup-bt",
                "shared"))
        emit("  " + "-" * 68)

        for r in rows:
            sc = r["super_add_clauses"]
            sb = r["super_add_bits"]
            sc_str = ("%+d" % sc) if sc != 0 else "0"
            sb_str = ("%+.1f" % sb) if sb != 0.0 else "0.0"
            emit("  %-4d  %-7s  %7d %7d %7s  %7.1f %7.1f %7s  %-6s"
                 % (r["m"], r["mode"],
                    r["merged_clauses_saved"], r["sep_clauses_saved"], sc_str,
                    r["merged_bits_saved"], r["sep_bits_saved"], sb_str,
                    _shared_marker(r["shared_abstraction_found"])))

        emit("")
        emit("DETAILS: shared abstraction detection per (M, mode)")
        emit("  %-4s  %-7s  %-8s  %-14s  %-30s"
             % ("M", "mode", "found", "invented_fn", "domains_covered"))
        emit("  " + "-" * 72)
        for r in rows:
            emit("  %-4d  %-7s  %-8s  %-14s  %-30s"
                 % (r["m"], r["mode"],
                    _shared_marker(r["shared_abstraction_found"]),
                    str(r["shared_invented_functor"] or "-"),
                    str(r.get("domains_covered_by_shared", 0)) + " domains"))

        # Analysis: crossover M* per mode
        emit("")
        emit("CROSSOVER M* (first M where shared abstraction is found)")
        for mode in ("clauses", "bits"):
            mode_rows = [r for r in rows if r["mode"] == mode]
            first_shared = next(
                (r["m"] for r in mode_rows if r["shared_abstraction_found"]), None)
            emit("  mode=%-7s -> M*=%s" % (mode, str(first_shared) if first_shared else "none in range"))

        # Super-additivity analysis
        emit("")
        emit("SUPER-ADDITIVITY (merged saves MORE than sum of separates, positive = transfer)")
        emit("  %-4s  %-7s  %10s  %10s" % ("M", "mode", "super+cl", "super+bt"))
        emit("  " + "-" * 36)
        for r in rows:
            sc = r["super_add_clauses"]
            sb = r["super_add_bits"]
            sc_str = ("%+d" % sc) if sc != 0 else "0"
            sb_str = ("%+.1f" % sb) if sb != 0.0 else "0.0"
            emit("  %-4d  %-7s  %10s  %10s"
                 % (r["m"], r["mode"], sc_str, sb_str))

        # Verdict
        emit("")
        emit("=" * 72)
        emit("VERDICT")
        emit("=" * 72)

        any_shared_clauses = any(
            r["shared_abstraction_found"] for r in rows if r["mode"] == "clauses")
        any_shared_bits = any(
            r["shared_abstraction_found"] for r in rows if r["mode"] == "bits")
        max_super_cl = max(
            (r["super_add_clauses"] for r in rows if r["mode"] == "clauses"),
            default=0)
        max_super_bt = max(
            (r["super_add_bits"] for r in rows if r["mode"] == "bits"),
            default=0.0)

        clauses_star = next(
            (r["m"] for r in rows if r["mode"] == "clauses"
             and r["shared_abstraction_found"]), None)
        bits_star = next(
            (r["m"] for r in rows if r["mode"] == "bits"
             and r["shared_abstraction_found"]), None)

        transfer_found = any_shared_clauses or any_shared_bits
        bits_favors = (bits_star is not None
                       and (clauses_star is None or bits_star < clauses_star))

        emit("")
        emit("Transfer effect (shared abstraction found in merged but not separates):")
        emit("  clauses mode: %s (M*=%s)" % (
            "YES" if any_shared_clauses else "NO",
            str(clauses_star) if clauses_star else "none"))
        emit("  bits mode:    %s (M*=%s)" % (
            "YES" if any_shared_bits else "NO",
            str(bits_star) if bits_star else "none"))
        emit("")
        emit("Super-additivity (peak extra savings from merged vs sum-of-separate):")
        emit("  clauses mode: %+d clauses" % max_super_cl)
        emit("  bits mode:    %+.1f bits" % max_super_bt)
        emit("")

        if transfer_found:
            emit("TRANSFER EFFECT CONFIRMED: merged dream discovers a shared abstraction")
            if bits_favors:
                emit("  Bits mode accepts it at LOWER M (M*=%d) than clauses (M*=%s)."
                     % (bits_star,
                        str(clauses_star) if clauses_star else "none"))
                emit("  -> Bits (rename-invariant) DOES favor cross-domain abstraction.")
            elif clauses_star is not None and bits_star is not None:
                if clauses_star == bits_star:
                    emit("  Both modes find shared abstraction at same M*=%d." % clauses_star)
                    emit("  -> No evidence that bits favors it earlier than clauses.")
                else:
                    emit("  Clauses mode accepts it at M*=%d, bits at M*=%d."
                         % (clauses_star, bits_star))
                    emit("  -> Clauses mode is LESS conservative for this abstraction.")
            else:
                emit("  Only one mode finds the shared abstraction.")
        else:
            emit("NO TRANSFER EFFECT: neither mode found a shared abstraction")
            emit("  in the merged KB for M in %s." % M_VALUES)
            emit("  Possible reasons: the MDL gate rejects the invention at all")
            emit("  tested M values, or Op D is structurally blocked by the tag")
            emit("  predicates changing the skeleton (see ANALYSIS below).")

        # Store summary fields
        run.results["summary"] = {
            "transfer_found": transfer_found,
            "shared_abstraction_clauses_mode": any_shared_clauses,
            "shared_abstraction_bits_mode": any_shared_bits,
            "m_star_clauses": clauses_star,
            "m_star_bits": bits_star,
            "bits_favors_shared_abstraction": bits_favors,
            "peak_super_add_clauses": max_super_cl,
            "peak_super_add_bits": round(max_super_bt, 2),
        }

    print("")
    print("Wrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
