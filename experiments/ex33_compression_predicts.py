#!/usr/bin/env python3
"""
EX33: Does compression predict generalization?
      And does bits-saved predict better than clause count?

BACKGROUND
----------
The paper states: "two domains do not establish a predictive relationship
between compression ratio and generalization recall" and "connecting the
bits-length margin of an accepted abstraction to its downstream
generalization value remains open." EX33 fills this gap by generating a
FAMILY of 30 synthetic domains varying in compressibility and structure,
then testing whether compression predicts held-out recovery across the
family -- and whether the bits currency is the better predictor.

RECOVERABLE PROTOCOL (new-entity, not counterexample holdout)
-------------------------------------------------------------
EX31 found that the counterexample holdout gives 0% recovery because
Op C excepts the held-out entity. The NEW-ENTITY protocol works: dream
on the FULL KB, then add new entities with only their base/guard facts,
and check if derived facts are derivable. New entities are never seen
during dream so they are never excepted. EX25b gets ~53% symbolic recall
this way on the crafting domain.

DOMAIN FAMILY DESIGN (TWO TIERS + CLAUSES-ONLY LAYER)
------------------------------------------------------
Each domain has groups of one of three pure types:
  GOOD   (sz >= 4, n_exc=0): both clauses and bits accept -> 100% recovery
  CL_ONLY (sz = 3, n_exc=1): clauses accepts, bits REJECTS (per EX29)
  INCOMP  (sz <= 2): neither accepts -> 0% recovery

Critically, EX29 confirmed that (n_pass=3, n_exc=1) is REJECTED by bits
even in multi-group KBs with the same group type. The rejection only
flips in MIXED KBs (when other larger groups inflate the symbol table).
So pure-type domains give clean mode separation.

Three domain subfamilies (10 domains each):

A) GOOD + INCOMP (n_pass=5, n_exc=0 for compressible groups)
   Recovery_clauses = Recovery_bits = n_good / (n_good + n_incomp)
   Both modes agree. Tests basic prediction; no mode divergence.

B) CL_ONLY + INCOMP (n_pass=3, n_exc=1 for compressible groups)
   Recovery_clauses = n_cl_only / (n_cl_only + n_incomp)
   Recovery_bits    = 0%  (bits always rejects these groups)
   Clauses mode shows non-trivial recovery; bits shows 0%.

C) GOOD + CL_ONLY (n_incomp=0 but some groups only compressible by clauses)
   Recovery_clauses = (n_good + n_cl_only) / total = 100%
   Recovery_bits    = n_good / total
   Both modes derive some recovery; only clauses recovers cl_only groups.
   NOTE: This is the mixed case. Tested to ensure no symbol-table
   pollution (each group uses unique predicates; EX29 analysis shows the
   pollution only occurs when OTHER groups have been compressed first,
   changing the post-compression KB size).

ANALYSIS
--------
  (a) Spearman and Pearson correlations (compression vs recovery) per mode.
  (b) Mean recovery under bits vs clauses mode.
  (c) Domain-level table: do higher bits_saved domains have higher recovery?

Usage: python experiments/ex33_compression_predicts.py
Writes: experiments/data/ex33/runs/<run_id>/{meta.json,results.json,summary.txt}
        experiments/data/ex33/latest.json
"""
import pathlib
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _harness import experiment_run  # noqa: E402

from dreamlog.compression.dl import description_length  # noqa: E402
from dreamlog.evaluator import PrologEvaluator  # noqa: E402
from dreamlog.kb_dreamer import KnowledgeBaseDreamer  # noqa: E402
from dreamlog.knowledge import Fact, KnowledgeBase, Rule  # noqa: E402
from dreamlog.prefix_parser import parse_s_expression  # noqa: E402

try:
    from scipy.stats import spearmanr, pearsonr  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ==========================================================================
# CONSTANTS
# ==========================================================================

MASTER_SEED = 42
MIN_GROUP_SIZE_FOR_OP_C = 3
N_NEW_PER_GROUP = 3           # new entities per group for generalization test

# Group size parameters
SZ_GOOD = 5     # n_pass for GOOD groups (both modes accept)
SZ_CL_ONLY = 3  # n_pass for CL_ONLY groups
N_EXC_CL_ONLY = 1  # exceptions in CL_ONLY groups (n_pass=3, n_exc=1)
SZ_INCOMP = 2   # n_pass for INCOMP groups (neither mode fires)


# ==========================================================================
# DOMAIN BUILDER
# ==========================================================================

def _group_facts(group_idx: int,
                 n_pass: int,
                 n_exc: int = 0,
                 entity_prefix: str = "e") -> Tuple[List[str], List[str]]:
    """Base and derived facts for one group (cat/prop pair)."""
    base: List[str] = []
    derived: List[str] = []
    guard = "cat%d" % group_idx
    prop = "prop%d" % group_idx
    val = "v%d" % group_idx
    for j in range(n_pass):
        e = "%s%d_%d" % (entity_prefix, group_idx, j)
        base.append("(%s %s)" % (guard, e))
        derived.append("(%s %s %s)" % (prop, e, val))
    for k in range(n_exc):
        e = "%sx%d_%d" % (entity_prefix, group_idx, k)
        base.append("(%s %s)" % (guard, e))
        # No prop fact -- exception
    return base, derived


def build_domain_a(n_good: int, n_incomp: int,
                   ) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Subfamily A: GOOD groups (sz=5, n_exc=0) + INCOMP groups (sz=2).
    Both modes: recovery = n_good / (n_good + n_incomp).
    """
    base, derived = [], []
    gi = 0
    for _ in range(n_good):
        b, d = _group_facts(gi, SZ_GOOD, 0)
        base.extend(b); derived.extend(d)
        gi += 1
    for _ in range(n_incomp):
        b, d = _group_facts(gi, SZ_INCOMP, 0, entity_prefix="f")
        base.extend(b); derived.extend(d)
        gi += 1

    new_base, new_checks = [], []
    gi = 0
    for _ in range(n_good):
        guard = "cat%d" % gi; prop = "prop%d" % gi; val = "v%d" % gi
        for j in range(N_NEW_PER_GROUP):
            e = "ng%d_%d" % (gi, j)
            new_base.append("(%s %s)" % (guard, e))
            new_checks.append("(%s %s %s)" % (prop, e, val))
        gi += 1
    for _ in range(n_incomp):
        guard = "cat%d" % gi; prop = "prop%d" % gi; val = "v%d" % gi
        for j in range(N_NEW_PER_GROUP):
            e = "ng%d_%d" % (gi, j)
            new_base.append("(%s %s)" % (guard, e))
            new_checks.append("(%s %s %s)" % (prop, e, val))
        gi += 1
    return base, derived, new_base, new_checks


def build_domain_b(n_cl_only: int, n_incomp: int,
                   ) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Subfamily B: CL_ONLY groups (sz=3, n_exc=1) + INCOMP groups (sz=2).
    Clauses mode: recovery = n_cl_only / (n_cl_only + n_incomp).
    Bits mode: recovery = 0% (bits rejects all CL_ONLY groups).
    """
    base, derived = [], []
    gi = 0
    for _ in range(n_cl_only):
        b, d = _group_facts(gi, SZ_CL_ONLY, N_EXC_CL_ONLY)
        base.extend(b); derived.extend(d)
        gi += 1
    for _ in range(n_incomp):
        b, d = _group_facts(gi, SZ_INCOMP, 0, entity_prefix="f")
        base.extend(b); derived.extend(d)
        gi += 1

    new_base, new_checks = [], []
    gi = 0
    for _ in range(n_cl_only):
        guard = "cat%d" % gi; prop = "prop%d" % gi; val = "v%d" % gi
        for j in range(N_NEW_PER_GROUP):
            e = "ng%d_%d" % (gi, j)
            new_base.append("(%s %s)" % (guard, e))
            new_checks.append("(%s %s %s)" % (prop, e, val))
        gi += 1
    for _ in range(n_incomp):
        guard = "cat%d" % gi; prop = "prop%d" % gi; val = "v%d" % gi
        for j in range(N_NEW_PER_GROUP):
            e = "ng%d_%d" % (gi, j)
            new_base.append("(%s %s)" % (guard, e))
            new_checks.append("(%s %s %s)" % (prop, e, val))
        gi += 1
    return base, derived, new_base, new_checks


def build_domain_c(n_good: int, n_cl_only: int,
                   ) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Subfamily C: GOOD groups + CL_ONLY groups (no incomp groups).
    Clauses mode: recovery = 100% (all groups compressed by clauses).
    Bits mode: recovery = n_good / (n_good + n_cl_only).
    Groups use unique predicates so no symbol-table pollution.
    """
    base, derived = [], []
    gi = 0
    for _ in range(n_good):
        b, d = _group_facts(gi, SZ_GOOD, 0)
        base.extend(b); derived.extend(d)
        gi += 1
    for _ in range(n_cl_only):
        b, d = _group_facts(gi, SZ_CL_ONLY, N_EXC_CL_ONLY)
        base.extend(b); derived.extend(d)
        gi += 1

    new_base, new_checks = [], []
    gi = 0
    for _ in range(n_good):
        guard = "cat%d" % gi; prop = "prop%d" % gi; val = "v%d" % gi
        for j in range(N_NEW_PER_GROUP):
            e = "ng%d_%d" % (gi, j)
            new_base.append("(%s %s)" % (guard, e))
            new_checks.append("(%s %s %s)" % (prop, e, val))
        gi += 1
    for _ in range(n_cl_only):
        guard = "cat%d" % gi; prop = "prop%d" % gi; val = "v%d" % gi
        for j in range(N_NEW_PER_GROUP):
            e = "ng%d_%d" % (gi, j)
            new_base.append("(%s %s)" % (guard, e))
            new_checks.append("(%s %s %s)" % (prop, e, val))
        gi += 1
    return base, derived, new_base, new_checks


def build_kb_from_strings(fact_strings: List[str]) -> KnowledgeBase:
    kb = KnowledgeBase()
    for s in fact_strings:
        if ":-" in s:
            parts = s.split(":-", 1)
            head = parse_s_expression(parts[0].strip())
            body = [parse_s_expression(b.strip())
                    for b in parts[1].split(",")]
            kb.add_rule(Rule(head, body))
        else:
            kb.add_fact(Fact(parse_s_expression(s)))
    return kb


def is_derivable(kb: KnowledgeBase, query_str: str,
                 max_calls: int = 5000) -> bool:
    term = parse_s_expression(query_str)
    ev = PrologEvaluator(kb, max_total_calls=max_calls)
    return ev.has_solution(term)


# ==========================================================================
# PER-DOMAIN MEASUREMENT
# ==========================================================================

def measure_domain(
    domain_id: str,
    subfamily: str,       # "A", "B", or "C"
    n_good: int,
    n_cl_only: int,
    n_incomp: int,
    dl_mode: str,
) -> Dict:
    """Run one domain x mode measurement."""
    if subfamily == "A":
        base, derived, new_base, new_checks = build_domain_a(n_good, n_incomp)
    elif subfamily == "B":
        base, derived, new_base, new_checks = build_domain_b(n_cl_only, n_incomp)
    else:  # C
        base, derived, new_base, new_checks = build_domain_c(n_good, n_cl_only)

    full_facts = base + derived
    kb = build_kb_from_strings(full_facts)

    dl_bits_before = description_length(kb, mode="bits")
    dl_clauses_before = description_length(kb, mode="clauses")

    dreamer = KnowledgeBaseDreamer(
        llm_client=None,
        open_world=False,
        dl_mode=dl_mode,
        min_group_size=MIN_GROUP_SIZE_FOR_OP_C,
    )
    session = dreamer.dream(kb, verify=True)

    dl_bits_after = description_length(kb, mode="bits")
    dl_clauses_after = description_length(kb, mode="clauses")

    bits_saved = dl_bits_before - dl_bits_after
    clauses_saved = dl_clauses_before - dl_clauses_after
    clause_compression_ratio = (
        dl_clauses_after / dl_clauses_before if dl_clauses_before > 0 else 1.0)
    bits_compression_ratio = (
        dl_bits_after / dl_bits_before if dl_bits_before > 0 else 1.0)

    for s in new_base:
        kb.add_fact(Fact(parse_s_expression(s)))

    total_checks = len(new_checks)
    recovered = sum(1 for check in new_checks if is_derivable(kb, check))
    recovery = recovered / total_checks if total_checks > 0 else 0.0

    rules_found = sum(
        1 for op in session.operations
        for c in op.new_clauses
        if isinstance(c, Rule)
    )

    return {
        "domain_id": domain_id,
        "subfamily": subfamily,
        "dl_mode": dl_mode,
        "n_good": n_good,
        "n_cl_only": n_cl_only,
        "n_incomp": n_incomp,
        "n_groups_total": n_good + n_cl_only + n_incomp,
        "n_new_checks": total_checks,
        "dl_bits_before": round(dl_bits_before, 2),
        "dl_bits_after": round(dl_bits_after, 2),
        "bits_saved": round(bits_saved, 2),
        "bits_compression_ratio": round(bits_compression_ratio, 4),
        "dl_clauses_before": dl_clauses_before,
        "dl_clauses_after": dl_clauses_after,
        "clauses_saved": clauses_saved,
        "clause_compression_ratio": round(clause_compression_ratio, 4),
        "session_compression_ratio": round(session.compression_ratio, 4),
        "rules_found": rules_found,
        "recovered": recovered,
        "recovery": round(recovery, 4),
    }


# ==========================================================================
# CORRELATION HELPERS
# ==========================================================================

def _spearman(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if not _HAS_SCIPY or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan"), float("nan")
    r, p = spearmanr(xs, ys)
    return float(r), float(p)


def _pearson(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if not _HAS_SCIPY or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan"), float("nan")
    r, p = pearsonr(xs, ys)
    return float(r), float(p)


def compute_correlations(rows: List[Dict]) -> Dict:
    by_mode: Dict[str, List[Dict]] = {}
    for row in rows:
        by_mode.setdefault(row["dl_mode"], []).append(row)

    result = {}
    for mode, mode_rows in by_mode.items():
        recoveries = [r["recovery"] for r in mode_rows]
        bits_saved = [r["bits_saved"] for r in mode_rows]
        bits_compr = [1.0 - r["bits_compression_ratio"] for r in mode_rows]
        clauses_saved = [float(r["clauses_saved"]) for r in mode_rows]
        clause_compr = [1.0 - r["clause_compression_ratio"] for r in mode_rows]

        sr_bs, sp_bs = _spearman(bits_saved, recoveries)
        pr_bs, pp_bs = _pearson(bits_saved, recoveries)
        sr_bc, sp_bc = _spearman(bits_compr, recoveries)
        pr_bc, pp_bc = _pearson(bits_compr, recoveries)
        sr_cs, sp_cs = _spearman(clauses_saved, recoveries)
        pr_cs, pp_cs = _pearson(clauses_saved, recoveries)
        sr_cc, sp_cc = _spearman(clause_compr, recoveries)
        pr_cc, pp_cc = _pearson(clause_compr, recoveries)

        result[mode] = {
            "n": len(mode_rows),
            "recovery_mean": round(sum(recoveries) / len(recoveries), 4),
            "recovery_min": round(min(recoveries), 4),
            "recovery_max": round(max(recoveries), 4),
            "bits_saved_vs_recovery": {
                "spearman_r": round(sr_bs, 4), "spearman_p": round(sp_bs, 4),
                "pearson_r": round(pr_bs, 4), "pearson_p": round(pp_bs, 4),
            },
            "bits_compression_vs_recovery": {
                "spearman_r": round(sr_bc, 4), "spearman_p": round(sp_bc, 4),
                "pearson_r": round(pr_bc, 4), "pearson_p": round(pp_bc, 4),
            },
            "clauses_saved_vs_recovery": {
                "spearman_r": round(sr_cs, 4), "spearman_p": round(sp_cs, 4),
                "pearson_r": round(pr_cs, 4), "pearson_p": round(pp_cs, 4),
            },
            "clause_compression_vs_recovery": {
                "spearman_r": round(sr_cc, 4), "spearman_p": round(sp_cc, 4),
                "pearson_r": round(pr_cc, 4), "pearson_p": round(pp_cc, 4),
            },
        }
    return result


def bits_vs_clauses_recovery(per_domain: List[Dict]) -> Dict:
    bits_recs = [r["recovery"] for r in per_domain if r["dl_mode"] == "bits"]
    cl_recs = [r["recovery"] for r in per_domain if r["dl_mode"] == "clauses"]
    return {
        "mean_recovery_bits_mode": round(
            sum(bits_recs) / len(bits_recs) if bits_recs else 0.0, 4),
        "mean_recovery_clauses_mode": round(
            sum(cl_recs) / len(cl_recs) if cl_recs else 0.0, 4),
        "n_bits": len(bits_recs),
        "n_clauses": len(cl_recs),
    }


# ==========================================================================
# DOMAIN FAMILY SPECIFICATION
# ==========================================================================
# 30 domains: 10 per subfamily.
# Format: (domain_id, subfamily, n_good, n_cl_only, n_incomp)

DOMAIN_SPECS = [
    # ---- Subfamily A: GOOD + INCOMP (10 domains) ----
    # Both modes give same recovery = n_good/(n_good+n_incomp)
    ("D00", "A", 0, 0, 4),    # recovery = 0%
    ("D01", "A", 0, 0, 3),    # recovery = 0%
    ("D02", "A", 1, 0, 3),    # recovery = 25%
    ("D03", "A", 1, 0, 2),    # recovery = 33%
    ("D04", "A", 2, 0, 2),    # recovery = 50%
    ("D05", "A", 2, 0, 1),    # recovery = 67%
    ("D06", "A", 3, 0, 2),    # recovery = 60%
    ("D07", "A", 3, 0, 1),    # recovery = 75%
    ("D08", "A", 4, 0, 2),    # recovery = 67%
    ("D09", "A", 4, 0, 0),    # recovery = 100%

    # ---- Subfamily B: CL_ONLY + INCOMP (10 domains) ----
    # Clauses: recovery = n_cl_only/(n_cl_only+n_incomp)
    # Bits:    recovery = 0% (bits always rejects CL_ONLY groups)
    ("D10", "B", 0, 0, 4),    # clauses: 0%  bits: 0%
    ("D11", "B", 0, 0, 3),    # clauses: 0%  bits: 0%
    ("D12", "B", 0, 1, 3),    # clauses: 25% bits: 0%
    ("D13", "B", 0, 1, 2),    # clauses: 33% bits: 0%
    ("D14", "B", 0, 2, 2),    # clauses: 50% bits: 0%
    ("D15", "B", 0, 2, 1),    # clauses: 67% bits: 0%
    ("D16", "B", 0, 3, 2),    # clauses: 60% bits: 0%
    ("D17", "B", 0, 3, 1),    # clauses: 75% bits: 0%
    ("D18", "B", 0, 4, 2),    # clauses: 67% bits: 0%
    ("D19", "B", 0, 4, 0),    # clauses: 100% bits: 0%

    # ---- Subfamily C: GOOD + CL_ONLY (10 domains) ----
    # Clauses: recovery = 100% (all groups compressed)
    # Bits:    recovery = n_good / (n_good + n_cl_only)
    ("D20", "C", 0, 4, 0),    # clauses: 100% bits: 0%
    ("D21", "C", 0, 3, 0),    # clauses: 100% bits: 0%
    ("D22", "C", 1, 3, 0),    # clauses: 100% bits: 25%
    ("D23", "C", 1, 2, 0),    # clauses: 100% bits: 33%
    ("D24", "C", 2, 2, 0),    # clauses: 100% bits: 50%
    ("D25", "C", 2, 1, 0),    # clauses: 100% bits: 67%
    ("D26", "C", 3, 1, 0),    # clauses: 100% bits: 75%
    ("D27", "C", 3, 3, 0),    # clauses: 100% bits: 50%
    ("D28", "C", 4, 2, 0),    # clauses: 100% bits: 67%
    ("D29", "C", 4, 0, 0),    # clauses: 100% bits: 100%
]

assert len(DOMAIN_SPECS) == 30


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    params = {
        "n_domains": len(DOMAIN_SPECS),
        "n_new_per_group": N_NEW_PER_GROUP,
        "modes": ["clauses", "bits"],
        "min_group_size_for_op_c": MIN_GROUP_SIZE_FOR_OP_C,
        "protocol": "new_entity_generalization",
        "open_world": False,
        "subfamilies": {
            "A": "GOOD (sz=5,exc=0) + INCOMP: both modes same recovery",
            "B": "CL_ONLY (sz=3,exc=1) + INCOMP: clauses>0, bits=0",
            "C": "GOOD + CL_ONLY: clauses=100%, bits varies",
        },
        "sz_good": SZ_GOOD,
        "sz_cl_only": SZ_CL_ONLY,
        "n_exc_cl_only": N_EXC_CL_ONLY,
        "sz_incomp": SZ_INCOMP,
    }

    with experiment_run(
        exp_id="ex33",
        name="does compression predict generalization",
        description=(
            "Family of 30 synthetic domains in 3 subfamilies. "
            "A: GOOD+INCOMP groups (both modes agree). "
            "B: CL_ONLY+INCOMP (clauses recovers, bits gets 0%). "
            "C: GOOD+CL_ONLY (clauses=100%, bits varies with n_good). "
            "New-entity generalization protocol: dream full KB, add new "
            "guard-only entities, measure fraction recovered. Tests "
            "whether compression predicts recovery (Spearman/Pearson) "
            "and whether bits_saved predicts better than clause_saved."
        ),
        script=__file__,
        params=params,
        seeds={"note": "fully deterministic; no RNG; structures fixed by spec"},
    ) as run:

        per_domain: List[Dict] = []

        print("EX33: Does compression predict generalization?")
        print("  n_domains=%d  protocol=new-entity  modes=clauses+bits"
              % len(DOMAIN_SPECS))
        print("  Subfamilies: A=GOOD+INCOMP  B=CL_ONLY+INCOMP  C=GOOD+CL_ONLY")
        print("  scipy: %s" % _HAS_SCIPY)
        print()
        print("  %-5s %-3s %-3s %-3s %-3s | %-7s %-6s | %+9s %+9s"
              % ("ID", "sf", "ng", "nc", "ni",
                 "mode", "recov", "bits_sv", "cl_sv"))
        print("  " + "-" * 68)

        for domain_id, subfamily, n_good, n_cl_only, n_incomp in DOMAIN_SPECS:
            for mode in ("clauses", "bits"):
                row = measure_domain(
                    domain_id=domain_id,
                    subfamily=subfamily,
                    n_good=n_good,
                    n_cl_only=n_cl_only,
                    n_incomp=n_incomp,
                    dl_mode=mode,
                )
                per_domain.append(row)
                print(
                    "  %-5s %-3s %-3d %-3d %-3d | %-7s %-5.1f%% | %+9.1f %+9d"
                    % (
                        domain_id, subfamily, n_good, n_cl_only, n_incomp,
                        mode, row["recovery"] * 100,
                        row["bits_saved"], row["clauses_saved"],
                    )
                )

        print()
        correlations = compute_correlations(per_domain)
        bvc = bits_vs_clauses_recovery(per_domain)

        run.results["per_domain"] = per_domain
        run.results["correlations"] = correlations
        run.results["bits_vs_clauses_recovery"] = bvc

        all_recoveries = [r["recovery"] for r in per_domain]
        n_nonzero = sum(1 for v in all_recoveries if v > 0)
        n_varied = sum(1 for v in all_recoveries if 0 < v < 1.0)
        recovery_min = min(all_recoveries)
        recovery_max = max(all_recoveries)
        recovery_mean = sum(all_recoveries) / len(all_recoveries)

        # Count domains where modes diverge
        cl_map = {r["domain_id"]: r
                  for r in per_domain if r["dl_mode"] == "clauses"}
        bt_map = {r["domain_id"]: r
                  for r in per_domain if r["dl_mode"] == "bits"}
        n_diverge = sum(
            1 for did in cl_map
            if abs(cl_map[did]["recovery"]
                   - bt_map.get(did, {}).get("recovery", 0)) > 0.01
        )
        run.results["n_domains_where_modes_diverge"] = n_diverge

        lines: List[str] = []

        def emit(s: str = "") -> None:
            print(s)
            lines.append(s)

        emit()
        emit("=" * 72)
        emit("EX33 RESULTS SUMMARY")
        emit("=" * 72)
        emit()
        emit("RECOVERY STATISTICS (all domains x modes, N=%d):" % len(all_recoveries))
        emit("  Min=%.1f%%  Max=%.1f%%  Mean=%.1f%%"
             "  Nonzero=%d/%d  Partial(0<r<1)=%d"
             % (recovery_min * 100, recovery_max * 100, recovery_mean * 100,
                n_nonzero, len(all_recoveries), n_varied))
        emit("  Domains where clauses-mode != bits-mode recovery: %d/%d"
             % (n_diverge, len(cl_map)))
        emit()

        if n_nonzero < 5:
            emit("STOP: fewer than 5 non-zero recoveries. "
                 "Recoverable protocol failed.")
            run.results["verdict"] = "HOLDOUT_NOT_RECOVERABLE"
        else:
            emit("CORRELATIONS (compression metrics vs recovery):")
            emit()
            for mode in ("clauses", "bits"):
                c = correlations.get(mode, {})
                emit(
                    "  Mode=%-7s  N=%d  "
                    "recovery=[%.1f%%, %.1f%%]  mean=%.1f%%"
                    % (mode, c.get("n", 0),
                       c.get("recovery_min", 0) * 100,
                       c.get("recovery_max", 0) * 100,
                       c.get("recovery_mean", 0) * 100)
                )
                emit()
                bs = c.get("bits_saved_vs_recovery", {})
                br = c.get("bits_compression_vs_recovery", {})
                cs = c.get("clauses_saved_vs_recovery", {})
                cr = c.get("clause_compression_vs_recovery", {})
                emit(
                    "    bits_saved vs recovery:          "
                    "Spearman r=%.3f p=%.3f  Pearson r=%.3f p=%.3f"
                    % (bs.get("spearman_r", float("nan")),
                       bs.get("spearman_p", float("nan")),
                       bs.get("pearson_r", float("nan")),
                       bs.get("pearson_p", float("nan"))))
                emit(
                    "    bits_compression vs recovery:    "
                    "Spearman r=%.3f p=%.3f  Pearson r=%.3f p=%.3f"
                    % (br.get("spearman_r", float("nan")),
                       br.get("spearman_p", float("nan")),
                       br.get("pearson_r", float("nan")),
                       br.get("pearson_p", float("nan"))))
                emit(
                    "    clauses_saved vs recovery:       "
                    "Spearman r=%.3f p=%.3f  Pearson r=%.3f p=%.3f"
                    % (cs.get("spearman_r", float("nan")),
                       cs.get("spearman_p", float("nan")),
                       cs.get("pearson_r", float("nan")),
                       cs.get("pearson_p", float("nan"))))
                emit(
                    "    clause_compression vs recovery:  "
                    "Spearman r=%.3f p=%.3f  Pearson r=%.3f p=%.3f"
                    % (cr.get("spearman_r", float("nan")),
                       cr.get("spearman_p", float("nan")),
                       cr.get("pearson_r", float("nan")),
                       cr.get("pearson_p", float("nan"))))
                emit()

            emit("BITS MODE vs CLAUSES MODE MEAN RECOVERY:")
            emit("  bits_mode_mean_recovery    = %.1f%%  (N=%d)"
                 % (bvc["mean_recovery_bits_mode"] * 100, bvc["n_bits"]))
            emit("  clauses_mode_mean_recovery = %.1f%%  (N=%d)"
                 % (bvc["mean_recovery_clauses_mode"] * 100, bvc["n_clauses"]))
            emit()

            # Verdict
            c_bits = correlations.get("bits", {})
            c_cl = correlations.get("clauses", {})

            bs_sp_bits = abs(c_bits.get("bits_saved_vs_recovery", {})
                             .get("spearman_r", 0.0))
            bs_sp_cl = abs(c_cl.get("bits_saved_vs_recovery", {})
                           .get("spearman_r", 0.0))
            bc_sp_bits = abs(c_bits.get("bits_compression_vs_recovery", {})
                             .get("spearman_r", 0.0))
            bc_sp_cl = abs(c_cl.get("bits_compression_vs_recovery", {})
                           .get("spearman_r", 0.0))
            cs_sp_bits = abs(c_bits.get("clauses_saved_vs_recovery", {})
                             .get("spearman_r", 0.0))
            cs_sp_cl = abs(c_cl.get("clauses_saved_vs_recovery", {})
                           .get("spearman_r", 0.0))
            cc_sp_bits = abs(c_bits.get("clause_compression_vs_recovery", {})
                             .get("spearman_r", 0.0))
            cc_sp_cl = abs(c_cl.get("clause_compression_vs_recovery", {})
                           .get("spearman_r", 0.0))

            bits_currency_best = max(bs_sp_bits, bc_sp_bits, bs_sp_cl, bc_sp_cl)
            clause_currency_best = max(cs_sp_bits, cc_sp_bits, cs_sp_cl, cc_sp_cl)
            overall_best = max(bits_currency_best, clause_currency_best)

            THRESH = 0.3
            if overall_best >= THRESH:
                if bits_currency_best > clause_currency_best + 0.05:
                    verdict = (
                        "SUPPORTED: compression predicts generalization; "
                        "bits is the better currency"
                    )
                elif clause_currency_best > bits_currency_best + 0.05:
                    verdict = (
                        "SUPPORTED: compression predicts generalization; "
                        "clause-count is the better currency "
                        "(bits not decisively better)"
                    )
                else:
                    verdict = (
                        "SUPPORTED: compression predicts generalization; "
                        "bits and clause-count predict similarly"
                    )
            else:
                rv_std = (
                    sum((x - recovery_mean) ** 2 for x in all_recoveries)
                    / len(all_recoveries)
                ) ** 0.5
                verdict = (
                    "REFUTED: compression does not predict recovery "
                    "(best |Spearman r|=%.3f; std_recovery=%.2f)"
                    % (overall_best, rv_std)
                )

            emit("VERDICT: " + verdict)
            emit()
            emit("  bits_saved |Spearman r|:          "
                 "%.3f (bits-mode), %.3f (clauses-mode)"
                 % (bs_sp_bits, bs_sp_cl))
            emit("  bits_compression |Spearman r|:    "
                 "%.3f (bits-mode), %.3f (clauses-mode)"
                 % (bc_sp_bits, bc_sp_cl))
            emit("  clauses_saved |Spearman r|:        "
                 "%.3f (bits-mode), %.3f (clauses-mode)"
                 % (cs_sp_bits, cs_sp_cl))
            emit("  clause_compression |Spearman r|:   "
                 "%.3f (bits-mode), %.3f (clauses-mode)"
                 % (cc_sp_bits, cc_sp_cl))
            run.results["verdict"] = verdict

        # ---- per-domain table ----
        emit()
        emit("PER-DOMAIN TABLE (both modes side-by-side):")
        emit(
            "  %-5s %-3s %-3s %-3s %-3s |"
            " %-9s %+8s %+8s |"
            " %-9s %+8s %+8s"
            % ("ID", "sf", "ng", "nc", "ni",
               "cl_recov", "cl_bsv", "cl_csv",
               "bt_recov", "bt_bsv", "bt_csv")
        )
        emit("  " + "-" * 78)
        for domain_id, subfamily, n_good, n_cl_only, n_incomp in DOMAIN_SPECS:
            cl = cl_map.get(domain_id, {})
            bt = bt_map.get(domain_id, {})
            diverge = " *" if abs(cl.get("recovery", 0)
                                  - bt.get("recovery", 0)) > 0.01 else "  "
            emit(
                "  %-5s%s%-3s %-3d %-3d %-3d |"
                " %-8.1f%% %+8.1f %+8d |"
                " %-8.1f%% %+8.1f %+8d"
                % (
                    domain_id, diverge, subfamily,
                    n_good, n_cl_only, n_incomp,
                    cl.get("recovery", 0) * 100,
                    cl.get("bits_saved", 0), cl.get("clauses_saved", 0),
                    bt.get("recovery", 0) * 100,
                    bt.get("bits_saved", 0), bt.get("clauses_saved", 0),
                )
            )

        run.summary_lines.extend(lines)

    print()
    print("Wrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
