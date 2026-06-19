#!/usr/bin/env python3
"""
EX39: LLM-augmented cross-predicate recall across multiple family-tree domains.

The paper's full-pipeline result (family tree ~80% cross-predicate recall) rests
on a single domain (n=1). This experiment extends that to n=6 distinct
family-tree variants (varied structure, seed-controlled) and measures:

  1. Symbolic-only cross-predicate recall (baseline; expected ~0 because
     symbolic operations cannot synthesize cross-predicate rules from facts alone).
  2. Full-pipeline cross-predicate recall (symbolic + Op G with Haiku).
  3. Precision on negative examples (no spurious derivations).
  4. Aggregate: mean +/- std across domains.

Cross-predicate relations: father, mother, grandfather, grandmother
(each requires combining base predicates male/female/parent/grandparent).

Protocol: new-entity generalization -- after dreaming on the training KB,
add new individuals with base facts only and check if the pipeline derives
their cross-predicate facts.

BUDGET GUARD: $2.00 hard abort. Uses one shared Haiku client.

Usage:
    python experiments/ex39_llm_multidomain.py
"""

import sys
import time
import math
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import experiment_run

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.prefix_parser import parse_s_expression
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient


# ---------------------------------------------------------------------------
# BUDGET
# ---------------------------------------------------------------------------

BUDGET_USD = 2.00
MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# FAMILY TREE GENERATOR (parameterized by seed + size)
# ---------------------------------------------------------------------------

def _make_name(prefix: str, idx: int) -> str:
    return f"{prefix}{idx}"


def generate_family_tree(seed: int, n_gen0: int = 2, n_gen1_pairs: int = 2,
                         n_gen2_per_pair: int = 2) -> Dict:
    """
    Generate a deterministic multi-generation family tree.

    Returns a dict: name -> {"gender": "m"|"f", "parents": [str, ...]}

    Structure:
      Gen 0: n_gen0 male + n_gen0 female (great-grandparents, no parents)
      Gen 1: n_gen1_pairs couples (male born from gen0 pair, female marries in)
             each gen0 male + gen0 female produce one gen1 male
      Gen 2: n_gen1_pairs * n_gen2_per_pair children (born from gen1 pairs)
      Gen 3: 2 children per gen2 male (leaves)
    """
    rng = random.Random(seed)
    tree: Dict = {}

    def add(name: str, gender: str, parents: List[str] = None):
        tree[name] = {"gender": gender, "parents": parents or []}

    # Gen 0 - great-grandparents (n_gen0 couples)
    gen0_males = []
    gen0_females = []
    for i in range(n_gen0):
        m = _make_name("ggf", i)
        f = _make_name("ggm", i)
        add(m, "m")
        add(f, "f")
        gen0_males.append(m)
        gen0_females.append(f)

    # Gen 1 - grandparents: n_gen1_pairs couples
    # Distribute gen0 parents across gen1 children
    gen1_pairs: List[Tuple[str, str]] = []  # (male, female)
    for i in range(n_gen1_pairs):
        # Pick gen0 parents (wrap around if needed)
        g0m = gen0_males[i % len(gen0_males)]
        g0f = gen0_females[i % len(gen0_females)]
        gf_name = _make_name("gf", i)
        gm_name = _make_name("gm", i)
        add(gf_name, "m", [g0m, g0f])
        add(gm_name, "f")  # married in, no gen0 parents
        gen1_pairs.append((gf_name, gm_name))

    # Gen 2 - parents: n_gen2_per_pair children per gen1 couple
    gen2_nodes: List[Tuple[str, str, str, str]] = []  # (name, gender, gf, gm)
    for pi, (gf, gm) in enumerate(gen1_pairs):
        for ci in range(n_gen2_per_pair):
            gender = "m" if ci % 2 == 0 else "f"
            name = _make_name(f"p{pi}_", ci)
            add(name, gender, [gf, gm])
            gen2_nodes.append((name, gender, gf, gm))

    # Gen 3 - leaves: pair up gen2 males with spouses, produce children
    gen2_males = [(n, gf, gm) for n, g, gf, gm in gen2_nodes if g == "m"]
    for mi, (dad, gf, gm) in enumerate(gen2_males):
        mom = _make_name(f"mom{mi}", 0)
        add(mom, "f")  # married in
        for ci in range(2):
            gender = "m" if ci == 0 else "f"
            child = _make_name(f"ch{mi}_", ci)
            add(child, gender, [dad, mom])

    return tree


def tree_base_facts(tree: Dict) -> List[str]:
    facts = []
    for name, info in tree.items():
        facts.append(f"(person {name})")
        facts.append(f"({'male' if info['gender'] == 'm' else 'female'} {name})")
        for par in info["parents"]:
            facts.append(f"(parent {par} {name})")
    return facts


def tree_derived_facts(tree: Dict) -> Dict[str, List[str]]:
    """Compute cross-predicate derived facts (father, mother, grandfather, grandmother)."""
    d: Dict[str, List[str]] = {
        "father": [],
        "mother": [],
        "grandfather": [],
        "grandmother": [],
    }
    for name, info in tree.items():
        for par in info["parents"]:
            if tree[par]["gender"] == "m":
                d["father"].append(f"(father {par} {name})")
            else:
                d["mother"].append(f"(mother {par} {name})")

        for par in info["parents"]:
            for gp in tree[par]["parents"]:
                if tree[gp]["gender"] == "m":
                    d["grandfather"].append(f"(grandfather {gp} {name})")
                else:
                    d["grandmother"].append(f"(grandmother {gp} {name})")
    # Deduplicate (structure can yield same pair via multiple paths)
    for k in d:
        d[k] = list(dict.fromkeys(d[k]))
    return d


def make_new_entities(seed: int) -> Tuple[List[str], List[Tuple[str, bool, str]]]:
    """
    Create a small self-contained sub-family (new entities with base facts only)
    for cross-predicate generalization checks.
    """
    prefix = f"nx{seed}_"
    ggf = prefix + "ggf"
    ggm = prefix + "ggm"
    dad = prefix + "dad"
    mom = prefix + "mom"
    son = prefix + "son"
    dau = prefix + "dau"

    base = [
        f"(person {ggf})", f"(male {ggf})",
        f"(person {ggm})", f"(female {ggm})",
        f"(person {dad})", f"(male {dad})",
        f"(parent {ggf} {dad})", f"(parent {ggm} {dad})",
        f"(person {mom})", f"(female {mom})",
        f"(person {son})", f"(male {son})",
        f"(parent {dad} {son})", f"(parent {mom} {son})",
        f"(person {dau})", f"(female {dau})",
        f"(parent {dad} {dau})", f"(parent {mom} {dau})",
    ]

    # Cross-predicate checks (the ones the LLM must help with)
    checks: List[Tuple[str, bool, str]] = [
        (f"(father {dad} {son})", True, "father: male parent of son"),
        (f"(father {dad} {dau})", True, "father: male parent of daughter"),
        (f"(mother {mom} {son})", True, "mother: female parent of son"),
        (f"(mother {mom} {dau})", True, "mother: female parent of daughter"),
        (f"(father {mom} {son})", False, "not-father: female cannot be father"),
        (f"(mother {dad} {son})", False, "not-mother: male cannot be mother"),
        (f"(grandfather {ggf} {son})", True, "grandfather: male grandparent of son"),
        (f"(grandfather {ggf} {dau})", True, "grandfather: male grandparent of daughter"),
        (f"(grandmother {ggm} {son})", True, "grandmother: female grandparent of son"),
        (f"(grandmother {ggm} {dau})", True, "grandmother: female grandparent of daughter"),
        (f"(grandfather {ggm} {son})", False, "not-grandfather: female cannot be grandfather"),
        (f"(grandmother {ggf} {son})", False, "not-grandmother: male cannot be grandmother"),
    ]
    return base, checks


# ---------------------------------------------------------------------------
# KB / inference helpers
# ---------------------------------------------------------------------------

def build_kb(fact_strings: List[str]) -> KnowledgeBase:
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


def is_derivable(kb: KnowledgeBase, query: str, max_calls: int = 10000) -> bool:
    term = parse_s_expression(query)
    ev = PrologEvaluator(kb, max_total_calls=max_calls)
    return ev.has_solution(term)


def dream_and_get_rules(kb: KnowledgeBase,
                        llm_client: Optional[LLMClient] = None,
                        max_prompt_facts: int = 200) -> Tuple[float, List[str]]:
    dreamer = KnowledgeBaseDreamer(
        llm_client=llm_client,
        max_prompt_facts=max_prompt_facts,
    )
    session = dreamer.dream(kb, verify=True)
    rules = []
    for op in session.operations:
        for c in op.new_clauses:
            if isinstance(c, Rule):
                rules.append(str(c))
    return session.compression_ratio, rules


# ---------------------------------------------------------------------------
# PER-DOMAIN EVALUATION
# ---------------------------------------------------------------------------

def run_domain(
    domain_idx: int,
    seed: int,
    llm_client: LLMClient,
) -> Dict:
    """
    Run one family-tree domain through symbolic and full-pipeline conditions.
    Returns per-domain results dict.
    """
    print(f"\n  --- Domain {domain_idx+1} (seed={seed}) ---")

    # Generate tree with varied parameters keyed to seed
    n_gen1 = 2 + (seed % 2)           # 2 or 3 gen1 pairs
    n_gen2 = 2 + ((seed // 2) % 2)    # 2 or 3 gen2 per pair
    tree = generate_family_tree(seed=seed, n_gen0=2,
                                n_gen1_pairs=n_gen1,
                                n_gen2_per_pair=n_gen2)
    base_facts = tree_base_facts(tree)
    derived = tree_derived_facts(tree)
    all_derived = [f for fs in derived.values() for f in fs]

    # New-entity facts and checks
    new_entity_base, checks = make_new_entities(seed)

    # Positives and negatives from checks
    positives = [(q, desc) for q, exp, desc in checks if exp]
    negatives = [(q, desc) for q, exp, desc in checks if not exp]

    print(f"    Tree: {len(tree)} people, {len(base_facts)} base, "
          f"{len(all_derived)} derived cross-predicate")
    for pred, fs in derived.items():
        print(f"      {pred}: {len(fs)}")

    domain_result: Dict = {
        "seed": seed,
        "n_gen1_pairs": n_gen1,
        "n_gen2_per_pair": n_gen2,
        "n_people": len(tree),
        "n_base": len(base_facts),
        "n_derived_cross_pred": len(all_derived),
        "conditions": {},
    }

    for condition, client in [("symbolic", None), ("full", llm_client)]:
        kb = build_kb(base_facts + all_derived)
        t0 = time.perf_counter()
        ratio, rules = dream_and_get_rules(kb, llm_client=client)
        elapsed = time.perf_counter() - t0

        # Add new-entity base facts
        for f in new_entity_base:
            kb.add_fact(Fact(parse_s_expression(f)))

        # Evaluate cross-predicate positives (recall)
        tp = 0
        fn = 0
        for q, desc in positives:
            if is_derivable(kb, q):
                tp += 1
            else:
                fn += 1

        # Evaluate cross-predicate negatives (precision)
        tn = 0
        fp = 0
        for q, desc in negatives:
            if is_derivable(kb, q):
                fp += 1
            else:
                tn += 1

        total_pos = tp + fn
        total_neg = tn + fp
        recall = tp / total_pos if total_pos else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        print(f"    [{condition}] dream: {elapsed:.1f}s, "
              f"ratio={ratio:.3f}, {len(rules)} rules")
        print(f"      cross-pred recall={recall:.1%} "
              f"(TP={tp}/{total_pos}), "
              f"precision={precision:.1%} "
              f"(FP={fp}/{total_neg})")
        if rules:
            for r in rules[:4]:
                print(f"        + {r}")
            if len(rules) > 4:
                print(f"        ... +{len(rules)-4} more")

        domain_result["conditions"][condition] = {
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
            "recall": recall,
            "precision": precision,
            "compression_ratio": ratio,
            "n_rules": len(rules),
            "rules": rules,
            "elapsed_s": round(elapsed, 2),
        }

    return domain_result


# ---------------------------------------------------------------------------
# AGGREGATE STATS
# ---------------------------------------------------------------------------

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return m, math.sqrt(variance)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

DOMAIN_SEEDS = [42, 123, 456, 789, 1024, 2048]   # 6 domains


def main():
    params = {
        "model": MODEL,
        "n_domains": len(DOMAIN_SEEDS),
        "domain_seeds": DOMAIN_SEEDS,
        "budget_usd": BUDGET_USD,
        "conditions": ["symbolic", "full"],
        "metric": "cross_predicate_recall",
        "new_entity_protocol": True,
        "temperature": 0.1,
    }

    print("=" * 72)
    print("  EX39: LLM-augmented cross-predicate recall across domains")
    print(f"  Model: {MODEL}")
    print(f"  Domains: {len(DOMAIN_SEEDS)} (seeds {DOMAIN_SEEDS})")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print("=" * 72)

    # Verify API key available
    api_key = os.getenv("MY_ANTHROPIC_API_KEY")
    if not api_key:
        print("  ERROR: MY_ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # One shared client accumulates usage across all domains
    shared_client = LLMClient(
        provider="anthropic",
        model=MODEL,
        api_key_env="MY_ANTHROPIC_API_KEY",
        temperature=0.1,
        max_tokens=1500,
    )

    with experiment_run(
        exp_id="ex39",
        name="LLM-augmented cross-predicate recall across domains",
        description=(
            "Extends the single-domain (n=1) full-pipeline cross-predicate "
            "generalization result from EX25 to n=6 distinct family-tree "
            "variants. Measures symbolic vs full-pipeline (Haiku) cross-predicate "
            "recall using new-entity protocol: add unseen individuals with base "
            "facts only, check whether the dreamed KB derives their father/mother/"
            "grandfather/grandmother relations."
        ),
        script=__file__,
        params=params,
        seeds={"domain_seeds": DOMAIN_SEEDS},
    ) as run:

        per_domain = []
        budget_aborted = False

        for di, seed in enumerate(DOMAIN_SEEDS):
            # Budget check BEFORE running domain
            current_cost = shared_client.estimated_cost()
            print(f"\n  [Cost so far: ${current_cost:.4f}]")
            if current_cost >= BUDGET_USD:
                print(f"  BUDGET ABORT: cost ${current_cost:.4f} >= ${BUDGET_USD:.2f}")
                budget_aborted = True
                break

            try:
                domain_res = run_domain(di, seed, shared_client)
                per_domain.append(domain_res)

                # Cost after each domain
                cost_now = shared_client.estimated_cost()
                print(f"  [Cost after domain {di+1}: ${cost_now:.4f}]")

                if cost_now >= BUDGET_USD:
                    print(f"  BUDGET ABORT: cost ${cost_now:.4f} >= ${BUDGET_USD:.2f}")
                    budget_aborted = True
                    break

            except Exception as exc:
                print(f"  ERROR in domain {di+1} (seed={seed}): {exc}")
                import traceback
                traceback.print_exc()
                # Record partial error and continue to next domain
                per_domain.append({
                    "seed": seed,
                    "error": str(exc),
                    "conditions": {},
                })

        # ---------------------------------------------------------------------------
        # Aggregate
        # ---------------------------------------------------------------------------

        symbolic_recalls = []
        full_recalls = []
        full_precisions = []

        for dr in per_domain:
            conds = dr.get("conditions", {})
            if "symbolic" in conds:
                symbolic_recalls.append(conds["symbolic"]["recall"])
            if "full" in conds:
                full_recalls.append(conds["full"]["recall"])
                full_precisions.append(conds["full"]["precision"])

        sym_mean, sym_std = mean_std(symbolic_recalls)
        full_mean, full_std = mean_std(full_recalls)
        prec_mean, prec_std = mean_std(full_precisions)

        n_completed = len(per_domain)

        # LLM accounting
        usage = shared_client.usage
        total_cost = shared_client.estimated_cost()

        run.log_llm(
            calls=usage.calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=total_cost,
        )

        run.results["per_domain"] = per_domain
        run.results["aggregate"] = {
            "n_domains_completed": n_completed,
            "n_domains_requested": len(DOMAIN_SEEDS),
            "budget_aborted": budget_aborted,
            "symbolic_recall_mean": sym_mean,
            "symbolic_recall_std": sym_std,
            "full_recall_mean": full_mean,
            "full_recall_std": full_std,
            "full_precision_mean": prec_mean,
            "full_precision_std": prec_std,
        }
        run.results["cost"] = {
            "total_usd": total_cost,
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "model": MODEL,
        }

        # ---------------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------------

        def emit(line=""):
            print(line)
            run.summary_lines.append(line)

        emit()
        emit("=" * 72)
        emit("  EX39 SUMMARY")
        emit("=" * 72)
        emit(f"  Domains completed: {n_completed}/{len(DOMAIN_SEEDS)}"
             + (" [BUDGET ABORT]" if budget_aborted else ""))
        emit()
        emit("  Per-domain cross-predicate recall (new-entity protocol):")
        emit(f"  {'Domain':>8}  {'Seed':>6}  {'Symbolic':>10}  {'Full':>10}  "
             f"{'Precision':>10}  {'Rules':>6}")
        emit(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
        for i, dr in enumerate(per_domain):
            conds = dr.get("conditions", {})
            sym_r = conds.get("symbolic", {}).get("recall", float("nan"))
            full_r = conds.get("full", {}).get("recall", float("nan"))
            prec = conds.get("full", {}).get("precision", float("nan"))
            n_rules = conds.get("full", {}).get("n_rules", 0)
            err = dr.get("error", "")
            if err:
                emit(f"  {i+1:>8}  {dr['seed']:>6}  ERROR: {err[:40]}")
            else:
                emit(f"  {i+1:>8}  {dr['seed']:>6}  {sym_r:>9.1%}  "
                     f"{full_r:>9.1%}  {prec:>9.1%}  {n_rules:>6}")
        emit()
        emit(f"  AGGREGATE ({n_completed} domains):")
        emit(f"    Symbolic recall:       {sym_mean:.1%} +/- {sym_std:.1%}")
        emit(f"    Full-pipeline recall:  {full_mean:.1%} +/- {full_std:.1%}")
        emit(f"    Full-pipeline precision: {prec_mean:.1%} +/- {prec_std:.1%}")
        emit()
        emit(f"  n=1 -> n={n_completed}: full-pipeline recall "
             f"{full_mean:.1%} +/- {full_std:.1%} vs symbolic {sym_mean:.1%} +/- {sym_std:.1%}")
        emit(f"  (LLM non-determinism note: temperature=0.1; "
             "small residual variance expected)")
        emit()
        emit(f"  LLM: {usage.calls} calls, "
             f"{usage.input_tokens:,} in / {usage.output_tokens:,} out")
        emit(f"  Total cost: ${total_cost:.4f} / ${BUDGET_USD:.2f} budget")

    # After context manager exits, run_dir and latest_path are set
    print()
    print(f"  run_dir:     {run.run_dir}")
    print(f"  latest.json: {run.latest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
