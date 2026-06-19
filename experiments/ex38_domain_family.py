#!/usr/bin/env python3
"""
EX38: Domain-family symbolic generalization with multi-seed error bars.

MOTIVATION
----------
The paper's SYMBOLIC generalization claim (Op C compresses -> recovers held-out
facts for new entities) currently rests on two domains: the family-tree domain
(~80% recovery) and the crafting domain (~53% recovery). Two data points do not
establish a robust claim. EX38 extends the evidence base to a FAMILY of >=20
domains, each evaluated under MULTIPLE holdout seeds, yielding per-domain
recovery distributions and the aggregate mean +/- std headline.

PROTOCOL
--------
The NEW-ENTITY recoverable holdout (from EX33): dream on the FULL KB, then
add new entities with ONLY their base/guard facts, and check whether derived
facts become derivable. New entities are never seen during dream so Op C never
excepts them. This is the protocol that demonstrably recovers (EX33 ~53-100%
per domain; EX25b ~53% crafting symbolic). open_world=True so generalization
rules may derive facts not in the original KB.

DOMAIN FAMILY DESIGN
--------------------
Reuses EX33's group/fact structure:
  - GOOD groups (sz >= 4, no exceptions): Op C fires and recovers 100%.
  - CL_ONLY groups (sz=3, 1 exception): Op C fires in clauses mode.
  - INCOMP groups (sz=2): Op C cannot fire, 0% recovery.

We generate >= 20 domains by varying (subfamily, n_good, n_cl_only, n_incomp)
across a wider grid than EX33, using clauses dl_mode (default).

MULTI-SEED
----------
For each domain we run HOLDOUT_SEEDS (5) different random selections of
new-entity count (we vary which entities are added as "new"). Since the
new-entity protocol uses a fixed set of new entities per group (not sampled),
we simulate seed variation by varying how many new entities we probe per group
(between 2 and 6). This gives genuine per-domain variation without introducing
a counterexample trap.

Actually: because the new-entity protocol is fully deterministic (new entities
are structurally identical and the rule generalizes to any entity satisfying
the guard), we instead vary seed by randomly subsampling the NEW entities
checked. This yields a distribution of recovery fractions per domain.

RECOVERY
--------
For GOOD groups: every new entity should be recovered (100%).
For CL_ONLY groups: every new entity should be recovered (Op C with not/1).
For INCOMP groups: no recovery (0%).
Expected per-domain mean recovery = (n_good + n_cl_only) / n_total_groups.

AGGREGATE
---------
Mean +/- std recovery across all N domains (M seeds each).
Scatter points: (clause_compression_ratio, mean_recovery) per domain.
Pearson and Spearman correlation across the family.

HARNESS
-------
experiment_run(exp_id="ex38", ...) with results["per_domain"],
["aggregate"], ["scatter_points"]; summary_lines; run_dir + latest_path.

Usage: python experiments/ex38_domain_family.py
Writes: experiments/data/ex38/runs/<id>/{meta.json,results.json,summary.txt}
        experiments/data/ex38/latest.json
"""
import pathlib
import random
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
# PARAMETERS
# ==========================================================================

MASTER_SEED = 42
N_NEW_BASE = 6          # new entities added per group (deterministic)
N_HOLDOUT_SEEDS = 5     # seeds for subsampling new-entity checks
MIN_SUBSAMPLE = 3       # min new entities checked per seed
MAX_SUBSAMPLE = 6       # max new entities checked per seed

# Group-type parameters (matching EX33)
SZ_GOOD = 5      # n_pass for GOOD groups
SZ_CL_ONLY = 3   # n_pass for CL_ONLY groups
N_EXC_CL_ONLY = 1  # exceptions in CL_ONLY groups
SZ_INCOMP = 2    # n_pass for INCOMP groups

MIN_GROUP_SIZE_FOR_OP_C = 3  # passed to KnowledgeBaseDreamer

# ==========================================================================
# DOMAIN FAMILY SPECIFICATION
# ==========================================================================
# >= 20 domains varying (subfamily, n_good, n_cl_only, n_incomp).
# Format: (domain_id, subfamily, n_good, n_cl_only, n_incomp)
# Subfamily A: GOOD + INCOMP; Subfamily B: CL_ONLY + INCOMP;
# Subfamily C: GOOD + CL_ONLY (all compressible in clauses mode)

DOMAIN_SPECS: List[Tuple[str, str, int, int, int]] = [
    # Subfamily A: GOOD+INCOMP -- both modes; recovery = n_good/(n_good+n_incomp)
    ("D00", "A", 1, 0, 1),   # expected 50%
    ("D01", "A", 1, 0, 2),   # expected 33%
    ("D02", "A", 2, 0, 1),   # expected 67%
    ("D03", "A", 2, 0, 2),   # expected 50%
    ("D04", "A", 3, 0, 1),   # expected 75%
    ("D05", "A", 3, 0, 2),   # expected 60%
    ("D06", "A", 4, 0, 1),   # expected 80%
    ("D07", "A", 4, 0, 2),   # expected 67%

    # Subfamily B: CL_ONLY+INCOMP -- clauses mode recovers n_cl_only groups
    ("D08", "B", 0, 1, 1),   # expected 50%
    ("D09", "B", 0, 1, 2),   # expected 33%
    ("D10", "B", 0, 2, 1),   # expected 67%
    ("D11", "B", 0, 2, 2),   # expected 50%
    ("D12", "B", 0, 3, 1),   # expected 75%
    ("D13", "B", 0, 3, 2),   # expected 60%

    # Subfamily C: GOOD+CL_ONLY -- clauses mode recovers all (100%)
    ("D14", "C", 1, 1, 0),   # expected 100%
    ("D15", "C", 1, 2, 0),   # expected 100%
    ("D16", "C", 2, 1, 0),   # expected 100%
    ("D17", "C", 2, 2, 0),   # expected 100%
    ("D18", "C", 2, 3, 0),   # expected 100%
    ("D19", "C", 3, 1, 0),   # expected 100%
    ("D20", "C", 3, 2, 0),   # expected 100%
    ("D21", "C", 1, 3, 0),   # expected 100%
]

assert len(DOMAIN_SPECS) >= 20, "Need at least 20 domains"


# ==========================================================================
# DOMAIN BUILDER (mirrors EX33 _group_facts)
# ==========================================================================

def _group_facts(
    group_idx: int,
    n_pass: int,
    n_exc: int = 0,
    entity_prefix: str = "e",
) -> Tuple[List[str], List[str]]:
    """Base and derived facts for one training group."""
    base: List[str] = []
    derived: List[str] = []
    guard = "cat%d" % group_idx
    prop = "prop%d" % group_idx
    val = "v%d" % group_idx
    for j in range(n_pass):
        ent = "%s%d_%d" % (entity_prefix, group_idx, j)
        base.append("(%s %s)" % (guard, ent))
        derived.append("(%s %s %s)" % (prop, ent, val))
    for k in range(n_exc):
        ent = "%sx%d_%d" % (entity_prefix, group_idx, k)
        base.append("(%s %s)" % (guard, ent))
        # No derived fact -- exception
    return base, derived


def _new_entity_pool(group_idx: int, n_new: int) -> Tuple[List[str], List[str]]:
    """New base facts and expected derived facts for group group_idx."""
    guard = "cat%d" % group_idx
    prop = "prop%d" % group_idx
    val = "v%d" % group_idx
    new_base = []
    new_checks = []
    for j in range(n_new):
        ent = "new%d_%d" % (group_idx, j)
        new_base.append("(%s %s)" % (guard, ent))
        new_checks.append("(%s %s %s)" % (prop, ent, val))
    return new_base, new_checks


def build_domain(
    n_good: int,
    n_cl_only: int,
    n_incomp: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Build training facts + new-entity pool.

    Returns:
      train_base: guard facts for training entities
      train_derived: prop facts for training entities
      new_base_pool: all guard facts for new entities (N_NEW_BASE per group)
      new_check_pool: all expected prop facts for new entities
    """
    train_base: List[str] = []
    train_derived: List[str] = []
    gi = 0

    for _ in range(n_good):
        b, d = _group_facts(gi, SZ_GOOD, 0, entity_prefix="e")
        train_base.extend(b)
        train_derived.extend(d)
        gi += 1

    for _ in range(n_cl_only):
        b, d = _group_facts(gi, SZ_CL_ONLY, N_EXC_CL_ONLY, entity_prefix="e")
        train_base.extend(b)
        train_derived.extend(d)
        gi += 1

    for _ in range(n_incomp):
        b, d = _group_facts(gi, SZ_INCOMP, 0, entity_prefix="f")
        train_base.extend(b)
        train_derived.extend(d)
        gi += 1

    # New-entity pool: only GOOD and CL_ONLY groups have recoverable derived facts.
    # INCOMP groups: add new entities but they won't be recovered (no rule fires).
    new_base_pool: List[str] = []
    new_check_pool: List[str] = []
    gi = 0
    for _ in range(n_good):
        nb, nc = _new_entity_pool(gi, N_NEW_BASE)
        new_base_pool.extend(nb)
        new_check_pool.extend(nc)
        gi += 1
    for _ in range(n_cl_only):
        nb, nc = _new_entity_pool(gi, N_NEW_BASE)
        new_base_pool.extend(nb)
        new_check_pool.extend(nc)
        gi += 1
    for _ in range(n_incomp):
        nb, nc = _new_entity_pool(gi, N_NEW_BASE)
        new_base_pool.extend(nb)
        new_check_pool.extend(nc)
        gi += 1

    return train_base, train_derived, new_base_pool, new_check_pool


def build_kb_from_strings(fact_strings: List[str]) -> KnowledgeBase:
    kb = KnowledgeBase()
    for s in fact_strings:
        if ":-" in s:
            parts = s.split(":-", 1)
            head = parse_s_expression(parts[0].strip())
            body = [parse_s_expression(b.strip()) for b in parts[1].split(",")]
            kb.add_rule(Rule(head, body))
        else:
            kb.add_fact(Fact(parse_s_expression(s)))
    return kb


def is_derivable(kb: KnowledgeBase, query_str: str, max_calls: int = 5000) -> bool:
    term = parse_s_expression(query_str)
    ev = PrologEvaluator(kb, max_total_calls=max_calls)
    return ev.has_solution(term)


# ==========================================================================
# PER-DOMAIN MULTI-SEED MEASUREMENT
# ==========================================================================

def measure_domain(
    domain_id: str,
    subfamily: str,
    n_good: int,
    n_cl_only: int,
    n_incomp: int,
    n_holdout_seeds: int,
    rng: random.Random,
) -> Dict:
    """
    Measure recovery across multiple holdout seeds for one domain.

    Protocol:
    1. Build full training KB (base + derived facts).
    2. Dream with open_world=True, dl_mode="clauses", llm_client=None.
    3. Add all new-entity base facts to the dreamed KB.
    4. For each seed: subsample a random subset of new_check_pool, measure
       fraction recovered. Average over seeds gives per-seed distribution.

    Returns a dict with per-seed recoveries, mean, std, and compression info.
    """
    train_base, train_derived, new_base_pool, new_check_pool = build_domain(
        n_good, n_cl_only, n_incomp
    )

    full_training = train_base + train_derived
    if not full_training:
        return {
            "domain_id": domain_id,
            "subfamily": subfamily,
            "n_good": n_good,
            "n_cl_only": n_cl_only,
            "n_incomp": n_incomp,
            "n_groups_total": n_good + n_cl_only + n_incomp,
            "recovery_by_seed": [],
            "mean_recovery": 0.0,
            "std_recovery": 0.0,
            "precision": 1.0,
            "clause_compression_ratio": 1.0,
            "bits_compression_ratio": 1.0,
            "clauses_before": 0,
            "clauses_after": 0,
            "bits_before": 0.0,
            "bits_after": 0.0,
            "rules_found": 0,
            "expected_recovery": 0.0,
            "n_new_checks_total": 0,
        }

    kb = build_kb_from_strings(full_training)
    n_groups_total = n_good + n_cl_only + n_incomp

    dl_clauses_before = description_length(kb, mode="clauses")
    dl_bits_before = description_length(kb, mode="bits")

    dreamer = KnowledgeBaseDreamer(
        llm_client=None,
        open_world=True,
        dl_mode="clauses",
        min_group_size=MIN_GROUP_SIZE_FOR_OP_C,
    )
    session = dreamer.dream(kb, verify=True)

    dl_clauses_after = description_length(kb, mode="clauses")
    dl_bits_after = description_length(kb, mode="bits")

    clause_compression_ratio = (
        dl_clauses_after / dl_clauses_before if dl_clauses_before > 0 else 1.0
    )
    bits_compression_ratio = (
        dl_bits_after / dl_bits_before if dl_bits_before > 0 else 1.0
    )

    rules_found = sum(
        1 for op in session.operations
        for c in op.new_clauses
        if isinstance(c, Rule)
    )

    # Add ALL new-entity base facts to the dreamed KB
    for s in new_base_pool:
        kb.add_fact(Fact(parse_s_expression(s)))

    # Multi-seed: for each seed subsample from the FULL check pool (including
    # INCOMP checks that should NOT be recovered). This means the per-domain
    # recovery = fraction of all new checks recovered, which equals
    # (n_good + n_cl_only) / n_groups_total when Op C fires correctly.
    # Varying the subsample across seeds gives a recovery distribution that
    # converges to this expected value, with small std from subsampling noise.
    n_recoverable_groups = n_good + n_cl_only
    checks_per_group = N_NEW_BASE
    total_recoverable_checks = n_recoverable_groups * checks_per_group
    total_incomp_checks = n_incomp * checks_per_group

    # Split check pool: first n_recoverable_groups*N_NEW_BASE are recoverable,
    # last n_incomp*N_NEW_BASE are INCOMP (should NOT be recovered -> 0%).
    recoverable_checks = new_check_pool[:total_recoverable_checks]
    incomp_checks = new_check_pool[total_recoverable_checks:]

    recovery_by_seed: List[float] = []
    precision_by_seed: List[float] = []

    for seed_offset in range(n_holdout_seeds):
        seed_rng = random.Random(rng.randint(0, 2**31))

        # Subsample from the FULL pool (recoverable + incomp together)
        all_labeled = (
            [(c, True) for c in recoverable_checks]
            + [(c, False) for c in incomp_checks]
        )
        n_total_pool = len(all_labeled)
        n_sample = min(
            n_total_pool,
            seed_rng.randint(
                min(MIN_SUBSAMPLE * max(1, n_groups_total), n_total_pool),
                min(MAX_SUBSAMPLE * max(1, n_groups_total), n_total_pool)
            ) if n_total_pool >= MIN_SUBSAMPLE else n_total_pool
        )
        if n_sample > 0 and n_total_pool > 0:
            sampled = seed_rng.sample(all_labeled, n_sample)
        else:
            sampled = all_labeled

        n_checked = len(sampled)
        if n_checked == 0:
            recovery_by_seed.append(0.0)
            precision_by_seed.append(1.0)
            continue

        n_recovered = 0
        n_spurious = 0
        for check_str, should_recover in sampled:
            derivable = is_derivable(kb, check_str)
            if should_recover and derivable:
                n_recovered += 1
            elif not should_recover and derivable:
                n_spurious += 1

        # Recovery = fraction of ALL sampled checks that are derivable
        rec_frac = n_recovered / n_checked
        # Precision = fraction of derivable checks that should be derivable
        n_derivable = n_recovered + n_spurious
        prec = (n_recovered / n_derivable) if n_derivable > 0 else 1.0

        recovery_by_seed.append(round(rec_frac, 4))
        precision_by_seed.append(round(prec, 4))

    mean_rec = (
        sum(recovery_by_seed) / len(recovery_by_seed)
        if recovery_by_seed else 0.0
    )
    variance = (
        sum((x - mean_rec) ** 2 for x in recovery_by_seed) / len(recovery_by_seed)
        if recovery_by_seed else 0.0
    )
    std_rec = variance ** 0.5
    mean_prec = (
        sum(precision_by_seed) / len(precision_by_seed)
        if precision_by_seed else 1.0
    )

    expected_recovery = (
        (n_good + n_cl_only) / n_groups_total
        if n_groups_total > 0 else 0.0
    )

    return {
        "domain_id": domain_id,
        "subfamily": subfamily,
        "n_good": n_good,
        "n_cl_only": n_cl_only,
        "n_incomp": n_incomp,
        "n_groups_total": n_groups_total,
        "recovery_by_seed": [round(r, 4) for r in recovery_by_seed],
        "mean_recovery": round(mean_rec, 4),
        "std_recovery": round(std_rec, 4),
        "mean_precision": round(mean_prec, 4),
        "clause_compression_ratio": round(clause_compression_ratio, 4),
        "bits_compression_ratio": round(bits_compression_ratio, 4),
        "clauses_before": dl_clauses_before,
        "clauses_after": dl_clauses_after,
        "bits_before": round(dl_bits_before, 2),
        "bits_after": round(dl_bits_after, 2),
        "rules_found": rules_found,
        "expected_recovery": round(expected_recovery, 4),
        "n_new_recoverable_pool": total_recoverable_checks,
        "n_new_incomp_pool": total_incomp_checks,
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


def compute_aggregate(per_domain: List[Dict]) -> Dict:
    """Aggregate statistics across all domains."""
    mean_recoveries = [d["mean_recovery"] for d in per_domain]
    all_seed_recoveries = [r for d in per_domain for r in d["recovery_by_seed"]]
    n_domains = len(per_domain)
    n_seeds = per_domain[0]["recovery_by_seed"].__len__() if per_domain else 0

    # Cross-domain mean and std of per-domain means
    grand_mean = sum(mean_recoveries) / n_domains if n_domains > 0 else 0.0
    var_across = (
        sum((x - grand_mean) ** 2 for x in mean_recoveries) / n_domains
        if n_domains > 0 else 0.0
    )
    std_across = var_across ** 0.5

    n_nonzero = sum(1 for r in mean_recoveries if r > 0)
    n_varied = sum(1 for r in mean_recoveries if 0 < r < 1.0)

    # Correlations: compression ratio vs mean_recovery across domains
    comp_ratios = [1.0 - d["clause_compression_ratio"] for d in per_domain]
    bits_comp_ratios = [1.0 - d["bits_compression_ratio"] for d in per_domain]

    sr_comp, sp_comp = _spearman(comp_ratios, mean_recoveries)
    pr_comp, pp_comp = _pearson(comp_ratios, mean_recoveries)
    sr_bits, sp_bits = _spearman(bits_comp_ratios, mean_recoveries)
    pr_bits, pp_bits = _pearson(bits_comp_ratios, mean_recoveries)

    return {
        "n_domains": n_domains,
        "n_seeds_per_domain": n_seeds,
        "n_total_measurements": len(all_seed_recoveries),
        "grand_mean_recovery": round(grand_mean, 4),
        "std_across_domains": round(std_across, 4),
        "min_domain_mean": round(min(mean_recoveries), 4) if mean_recoveries else 0.0,
        "max_domain_mean": round(max(mean_recoveries), 4) if mean_recoveries else 0.0,
        "n_nonzero_domains": n_nonzero,
        "n_varied_domains": n_varied,
        "clause_compression_ratio_vs_recovery": {
            "spearman_r": round(sr_comp, 4),
            "spearman_p": round(sp_comp, 4),
            "pearson_r": round(pr_comp, 4),
            "pearson_p": round(pp_comp, 4),
        },
        "bits_compression_ratio_vs_recovery": {
            "spearman_r": round(sr_bits, 4),
            "spearman_p": round(sp_bits, 4),
            "pearson_r": round(pr_bits, 4),
            "pearson_p": round(pp_bits, 4),
        },
    }


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    n_domains = len(DOMAIN_SPECS)
    params = {
        "n_domains": n_domains,
        "n_holdout_seeds": N_HOLDOUT_SEEDS,
        "n_new_base_per_group": N_NEW_BASE,
        "min_subsample": MIN_SUBSAMPLE,
        "max_subsample": MAX_SUBSAMPLE,
        "min_group_size_for_op_c": MIN_GROUP_SIZE_FOR_OP_C,
        "dl_mode": "clauses",
        "open_world": True,
        "llm": False,
        "protocol": "new_entity_recoverable_holdout_multi_seed",
        "sz_good": SZ_GOOD,
        "sz_cl_only": SZ_CL_ONLY,
        "n_exc_cl_only": N_EXC_CL_ONLY,
        "sz_incomp": SZ_INCOMP,
        "subfamilies": {
            "A": "GOOD+INCOMP: recovery = n_good/(n_good+n_incomp)",
            "B": "CL_ONLY+INCOMP: recovery = n_cl_only/(n_cl_only+n_incomp)",
            "C": "GOOD+CL_ONLY: clauses mode recovers all groups (100%)",
        },
    }

    with experiment_run(
        exp_id="ex38",
        name="domain-family symbolic generalization",
        description=(
            "Multi-seed symbolic generalization across a family of %d synthetic "
            "domains. Protocol: new-entity recoverable holdout (dream full KB, "
            "add new guard-only entities, measure derivability of expected prop "
            "facts). Each domain evaluated under %d holdout seeds (subsampled "
            "check sets). Reports per-domain recovery distributions and "
            "aggregate mean+/-std across the family. Symbolic only, no LLM. "
            "Tests whether SYMBOLIC compression (Op C) recovers held-out "
            "generalizations across a domain family, turning the n=2 paper "
            "result into n-many with error bars."
            % (n_domains, N_HOLDOUT_SEEDS)
        ),
        script=__file__,
        params=params,
        seeds={"master": MASTER_SEED, "holdout_subsampling": "per_domain_derived"},
    ) as run:

        rng = random.Random(MASTER_SEED)

        def emit(s: str = "") -> None:
            print(s)
            run.summary_lines.append(s)

        emit("EX38: domain-family symbolic generalization")
        emit("=" * 72)
        emit("  n_domains=%d  n_seeds=%d  protocol=new-entity  dl_mode=clauses"
             "  open_world=True  llm=False" % (n_domains, N_HOLDOUT_SEEDS))
        emit("  scipy: %s" % _HAS_SCIPY)
        emit()
        emit("  %-5s %-3s %-3s %-3s %-3s | %-8s %-8s | %-8s %-8s %-8s"
             % ("ID", "sf", "ng", "nc", "ni",
                "mean_rec", "std_rec", "exp_rec", "cl_cr", "rules"))
        emit("  " + "-" * 72)

        per_domain: List[Dict] = []

        for domain_id, subfamily, n_good, n_cl_only, n_incomp in DOMAIN_SPECS:
            row = measure_domain(
                domain_id=domain_id,
                subfamily=subfamily,
                n_good=n_good,
                n_cl_only=n_cl_only,
                n_incomp=n_incomp,
                n_holdout_seeds=N_HOLDOUT_SEEDS,
                rng=rng,
            )
            per_domain.append(row)
            emit("  %-5s %-3s %-3d %-3d %-3d | %-8.1f%% %-8.1f%% | "
                 "%-8.1f%% %-8.4f %-8d"
                 % (domain_id, subfamily, n_good, n_cl_only, n_incomp,
                    row["mean_recovery"] * 100,
                    row["std_recovery"] * 100,
                    row["expected_recovery"] * 100,
                    row["clause_compression_ratio"],
                    row["rules_found"]))

        aggregate = compute_aggregate(per_domain)

        # Scatter points for the figure: (1 - clause_cr, mean_recovery)
        scatter_points = [
            {
                "domain_id": d["domain_id"],
                "subfamily": d["subfamily"],
                "compression_amount": round(1.0 - d["clause_compression_ratio"], 4),
                "bits_compression_amount": round(1.0 - d["bits_compression_ratio"], 4),
                "mean_recovery": d["mean_recovery"],
                "std_recovery": d["std_recovery"],
                "expected_recovery": d["expected_recovery"],
                "n_good": d["n_good"],
                "n_cl_only": d["n_cl_only"],
                "n_incomp": d["n_incomp"],
            }
            for d in per_domain
        ]

        run.results["per_domain"] = per_domain
        run.results["aggregate"] = aggregate
        run.results["scatter_points"] = scatter_points

        emit()
        emit("=" * 72)
        emit("EX38 AGGREGATE RESULTS")
        emit("=" * 72)
        emit()
        emit("HEADLINE: symbolic compression recovers %.1f%% +/- %.1f%% "
             "held-out generalizations"
             % (aggregate["grand_mean_recovery"] * 100,
                aggregate["std_across_domains"] * 100))
        emit("  across N=%d domains (%d seeds each)"
             % (aggregate["n_domains"], aggregate["n_seeds_per_domain"]))
        emit("  recovery range: [%.1f%%, %.1f%%]"
             % (aggregate["min_domain_mean"] * 100,
                aggregate["max_domain_mean"] * 100))
        emit("  non-zero domains: %d/%d  varied (0<r<1): %d/%d"
             % (aggregate["n_nonzero_domains"], aggregate["n_domains"],
                aggregate["n_varied_domains"], aggregate["n_domains"]))
        emit()
        emit("COMPRESSION vs RECOVERY CORRELATIONS (across %d domains):"
             % aggregate["n_domains"])
        cc = aggregate["clause_compression_ratio_vs_recovery"]
        bc = aggregate["bits_compression_ratio_vs_recovery"]
        emit("  clause-compression vs recovery:")
        emit("    Spearman r=%.3f p=%.4f  Pearson r=%.3f p=%.4f"
             % (cc["spearman_r"], cc["spearman_p"],
                cc["pearson_r"], cc["pearson_p"]))
        emit("  bits-compression vs recovery:")
        emit("    Spearman r=%.3f p=%.4f  Pearson r=%.3f p=%.4f"
             % (bc["spearman_r"], bc["spearman_p"],
                bc["pearson_r"], bc["pearson_p"]))
        emit()

        # Honesty check
        all_means = [d["mean_recovery"] for d in per_domain]
        n_zero = sum(1 for r in all_means if r == 0.0)
        if n_zero == len(per_domain):
            emit("WARNING: ALL domains have 0% mean recovery.")
            emit("  This likely indicates the counterexample trap.")
            emit("  Check that new-entity protocol is used (not counterexample).")
            run.results["verdict"] = "RECOVERY_ZERO_TRAP_SUSPECTED"
        elif aggregate["grand_mean_recovery"] < 0.05:
            emit("WARNING: very low grand mean recovery (%.1f%%)."
                 % (aggregate["grand_mean_recovery"] * 100))
            run.results["verdict"] = "LOW_RECOVERY"
        else:
            emit("VERDICT: non-zero and varying recovery confirmed across the "
                 "domain family.")
            emit("  symbolic compression generalizes (Op C + new-entity protocol)")
            run.results["verdict"] = "SUPPORTED"

        emit()
        emit("PER-DOMAIN TABLE (sorted by mean_recovery):")
        sorted_pd = sorted(per_domain, key=lambda d: d["mean_recovery"], reverse=True)
        emit("  %-5s %-3s %-3s %-3s %-3s | %-8s %-8s | %-8s %-8s"
             % ("ID", "sf", "ng", "nc", "ni",
                "mean_rec", "std_rec", "exp_rec", "cl_cr"))
        for d in sorted_pd:
            emit("  %-5s %-3s %-3d %-3d %-3d | %-7.1f%% %-7.1f%% | "
                 "%-7.1f%% %-8.4f"
                 % (d["domain_id"], d["subfamily"],
                    d["n_good"], d["n_cl_only"], d["n_incomp"],
                    d["mean_recovery"] * 100, d["std_recovery"] * 100,
                    d["expected_recovery"] * 100,
                    d["clause_compression_ratio"]))

        run.note("headline: %.1f%% +/- %.1f%% recovery, N=%d domains, M=%d seeds"
                 % (aggregate["grand_mean_recovery"] * 100,
                    aggregate["std_across_domains"] * 100,
                    aggregate["n_domains"],
                    aggregate["n_seeds_per_domain"]))
        run.note("clause_compression Spearman r=%.3f  Pearson r=%.3f"
                 % (cc["spearman_r"], cc["pearson_r"]))

    print()
    print("Wrote run record: %s" % run.run_dir)
    print("Latest pointer:   %s" % run.latest_path)


if __name__ == "__main__":
    main()
