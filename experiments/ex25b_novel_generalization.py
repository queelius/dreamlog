#!/usr/bin/env python3
"""
EX25b: Generalization on a Novel Synthetic Domain.

EX25 used the family tree — a canonical Prolog example that Haiku has
likely memorized. This experiment tests whether compression-driven
generalization works on a domain the LLM has NEVER seen: a synthetic
crafting/alchemy system with invented material names.

The LLM cannot rely on prior knowledge of "lumite" or "vexal" — it must
discover rules purely from structural patterns in the data.

Domain: Alchemical crafting workshop
  - Materials with invented names and properties (phase, class, hazard)
  - Recipes combining materials into products
  - Artisans with skills
  - Tool requirements
  - Derived: can_craft, safe_recipe, dual_phase_recipe, material_compatible,
    master_artisan, hazardous_product, artisan_speciality

Compared against the family tree (canonical) and run through the same
generalization pipeline for direct comparison.

Usage:
    python experiments/ex25b_novel_generalization.py
"""

import sys
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact
from dreamlog.prefix_parser import parse_s_expression
from dreamlog.llm_client import LLMClient


# Helpers (build_kb, is_derivable, dream_kb) are imported from
# ex25_generalization below, alongside the family-tree fixtures.


# ═══════════════════════════════════════════════════════════════════
# CRAFTING DOMAIN: invented materials, real structural patterns
# ═══════════════════════════════════════════════════════════════════

# Materials: invented names, no real-world semantics
MATERIALS = {
    # (name, phase, material_class, hazardous?)
    "lumite":    ("solid", "metallic", False),
    "vexal":     ("solid", "metallic", False),
    "drossite":  ("solid", "metallic", True),
    "pyreth":    ("solid", "crystalline", False),
    "quarl":     ("solid", "crystalline", False),
    "noctite":   ("solid", "crystalline", True),
    "aerine":    ("liquid", "volatile", True),
    "sylphex":   ("liquid", "volatile", False),
    "thalline":  ("liquid", "organic", False),
    "morven":    ("liquid", "organic", False),
    "zephore":   ("gas", "volatile", True),
    "ethylis":   ("gas", "volatile", False),
    "fenrik":    ("solid", "composite", False),
    "glacine":   ("solid", "composite", False),
    "bramith":   ("solid", "metallic", True),
}

# Recipes: (product, input1, input2)
# Products have their own names — they are NOT in MATERIALS
RECIPES = [
    # Metallic alloys (solid + solid metallic)
    ("alloy_lv", "lumite", "vexal"),
    ("alloy_ld", "lumite", "drossite"),
    ("alloy_vb", "vexal", "bramith"),
    # Crystal fusions (crystalline + crystalline)
    ("crystal_pq", "pyreth", "quarl"),
    ("crystal_pn", "pyreth", "noctite"),
    # Solutions (liquid + solid)
    ("solution_tl", "thalline", "lumite"),
    ("solution_tp", "thalline", "pyreth"),
    ("solution_mp", "morven", "pyreth"),
    ("solution_ml", "morven", "lumite"),
    # Volatile mixes (liquid volatile + gas)
    ("vapor_az", "aerine", "zephore"),
    ("vapor_sz", "sylphex", "zephore"),
    ("vapor_se", "sylphex", "ethylis"),
    # Composites (composite + any solid)
    ("composite_fl", "fenrik", "lumite"),
    ("composite_fq", "fenrik", "quarl"),
    ("composite_gv", "glacine", "vexal"),
    ("composite_gp", "glacine", "pyreth"),
]

# Artisans with skills
ARTISANS = {
    # (name, [skills])
    "kael":    ["metalwork", "forging"],
    "sera":    ["metalwork", "alloying"],
    "thom":    ["crystallography"],
    "yara":    ["crystallography", "solution_craft"],
    "dax":     ["volatile_handling", "vapor_craft"],
    "mira":    ["solution_craft", "organic_processing"],
    "orin":    ["forging", "composite_work"],
    "fen":     ["composite_work"],
    "zara":    ["metalwork", "volatile_handling"],
}

# Skill requirements for recipe types
SKILL_REQUIREMENTS = {
    "alloy": "alloying",
    "crystal": "crystallography",
    "solution": "solution_craft",
    "vapor": "vapor_craft",
    "composite": "composite_work",
}


def crafting_base_facts() -> List[str]:
    """Generate base facts for the crafting domain."""
    facts = []

    # Materials
    for name, (phase, mclass, hazardous) in MATERIALS.items():
        facts.append(f"(material {name})")
        facts.append(f"(phase {name} {phase})")
        facts.append(f"(material_class {name} {mclass})")
        if hazardous:
            facts.append(f"(hazardous {name})")

    # Recipes
    for product, m1, m2 in RECIPES:
        facts.append(f"(recipe {product} {m1} {m2})")
        facts.append(f"(product {product})")

    # Artisans and skills
    for name, skills in ARTISANS.items():
        facts.append(f"(artisan {name})")
        for skill in skills:
            facts.append(f"(skill {name} {skill})")

    # Skill requirements per recipe TYPE (identified by product prefix)
    for prefix, skill in SKILL_REQUIREMENTS.items():
        facts.append(f"(requires_skill {prefix} {skill})")

    return facts


def crafting_derived_facts() -> Dict[str, List[str]]:
    """Compute derived facts from the crafting data."""
    d: Dict[str, List[str]] = {
        "hazardous_recipe": [],
        "safe_recipe": [],
        "same_phase_recipe": [],
        "metallic_alloy": [],
        "can_craft": [],
        "master_artisan": [],
    }

    for product, m1, m2 in RECIPES:
        phase1 = MATERIALS[m1][0]
        phase2 = MATERIALS[m2][0]
        class1 = MATERIALS[m1][1]
        class2 = MATERIALS[m2][1]
        haz1 = MATERIALS[m1][2]
        haz2 = MATERIALS[m2][2]

        # hazardous_recipe: recipe uses at least one hazardous material
        if haz1 or haz2:
            d["hazardous_recipe"].append(
                f"(hazardous_recipe {product})")

        # safe_recipe: neither input is hazardous
        if not haz1 and not haz2:
            d["safe_recipe"].append(f"(safe_recipe {product})")

        # same_phase_recipe: both inputs share phase
        if phase1 == phase2:
            d["same_phase_recipe"].append(
                f"(same_phase_recipe {product})")

        # metallic_alloy: recipe where both inputs are metallic
        if class1 == "metallic" and class2 == "metallic":
            d["metallic_alloy"].append(
                f"(metallic_alloy {product})")

        # can_craft: artisan has the skill matching recipe type
        recipe_type = product.split("_")[0]
        required_skill = SKILL_REQUIREMENTS.get(recipe_type)
        if required_skill:
            for artisan, skills in ARTISANS.items():
                if required_skill in skills:
                    d["can_craft"].append(
                        f"(can_craft {artisan} {product})")

    # master_artisan: has 2+ skills
    for artisan, skills in ARTISANS.items():
        if len(skills) >= 2:
            d["master_artisan"].append(
                f"(master_artisan {artisan})")

    return d


def crafting_negatives() -> List[Tuple[str, bool]]:
    """Negative examples that should NOT be derivable."""
    negs = []

    # Artisans who can't craft certain recipes
    # thom only has crystallography — can't craft alloys
    negs.append("(can_craft thom alloy_lv)")
    negs.append("(can_craft thom solution_tl)")
    negs.append("(can_craft fen alloy_lv)")
    negs.append("(can_craft dax alloy_lv)")

    # Non-hazardous recipes are not hazardous
    negs.append("(hazardous_recipe crystal_pq)")  # pyreth+quarl both safe
    negs.append("(hazardous_recipe composite_fl)")  # fenrik+lumite both safe
    negs.append("(hazardous_recipe alloy_lv)")  # lumite+vexal both safe

    # Not a metallic alloy (crystalline fusion)
    negs.append("(metallic_alloy crystal_pq)")
    negs.append("(metallic_alloy solution_tl)")

    # Single-skill artisan is not master
    negs.append("(master_artisan thom)")
    negs.append("(master_artisan fen)")

    # Safe recipes ARE NOT hazardous
    negs.append("(hazardous_recipe solution_ml)")  # morven+lumite both safe

    return [(n, False) for n in negs]


# ── New entities for generalization test ──────────────────────────

NEW_CRAFTING_BASE = [
    # ── New materials (6 total) ──
    "(material novex)", "(phase novex solid)", "(material_class novex metallic)",
    "(material draith)", "(phase draith liquid)", "(material_class draith organic)",
    "(hazardous draith)",
    "(material crystex)", "(phase crystex solid)", "(material_class crystex crystalline)",
    "(material ignara)", "(phase ignara gas)", "(material_class ignara volatile)",
    "(hazardous ignara)",
    "(material pallum)", "(phase pallum solid)", "(material_class pallum composite)",
    "(material solvine)", "(phase solvine liquid)", "(material_class solvine volatile)",

    # ── New recipes (8 total) ──
    "(recipe solution_dl draith novex)", "(product solution_dl)",
    "(recipe crystal_cn crystex novex)", "(product crystal_cn)",
    "(recipe alloy_np novex pallum)", "(product alloy_np)",
    "(recipe vapor_si solvine ignara)", "(product vapor_si)",
    "(recipe composite_pn pallum novex)", "(product composite_pn)",
    "(recipe solution_sc solvine crystex)", "(product solution_sc)",
    "(recipe alloy_nn novex novex)", "(product alloy_nn)",
    "(recipe crystal_cc crystex crystex)", "(product crystal_cc)",

    # ── New artisans (4 total) ──
    "(artisan venn)", "(skill venn solution_craft)", "(skill venn alloying)",
    "(artisan lira)", "(skill lira crystallography)",
    "(artisan torvak)", "(skill torvak vapor_craft)",
    "(skill torvak composite_work)", "(skill torvak metalwork)",
    "(artisan quill)", "(skill quill metalwork)",
]

NEW_CRAFTING_CHECKS = [
    # ── hazardous_recipe (uses >= 1 hazardous input) ──
    ("(hazardous_recipe solution_dl)", True,
     "draith is hazardous"),
    ("(hazardous_recipe vapor_si)", True,
     "ignara is hazardous"),
    ("(hazardous_recipe crystal_cn)", False,
     "crystex+novex both safe"),
    ("(hazardous_recipe composite_pn)", False,
     "pallum+novex both safe"),
    ("(hazardous_recipe alloy_nn)", False,
     "novex+novex both safe"),

    # ── safe_recipe (no hazardous input) ──
    ("(safe_recipe crystal_cn)", True,
     "both inputs safe"),
    ("(safe_recipe composite_pn)", True,
     "both inputs safe"),
    ("(safe_recipe alloy_nn)", True,
     "both inputs safe"),
    ("(safe_recipe crystal_cc)", True,
     "both inputs safe"),
    ("(safe_recipe solution_dl)", False,
     "draith is hazardous"),
    ("(safe_recipe vapor_si)", False,
     "ignara is hazardous"),

    # ── same_phase_recipe (both inputs share phase) ──
    ("(same_phase_recipe crystal_cn)", True,
     "solid+solid"),
    ("(same_phase_recipe alloy_nn)", True,
     "solid+solid"),
    ("(same_phase_recipe crystal_cc)", True,
     "solid+solid"),
    ("(same_phase_recipe composite_pn)", True,
     "solid+solid"),
    ("(same_phase_recipe solution_dl)", False,
     "liquid+solid"),
    ("(same_phase_recipe vapor_si)", False,
     "liquid+gas"),
    ("(same_phase_recipe solution_sc)", False,
     "liquid+solid"),

    # ── metallic_alloy (both inputs metallic) ──
    ("(metallic_alloy crystal_cn)", False,
     "crystex is crystalline"),
    ("(metallic_alloy alloy_nn)", True,
     "novex+novex both metallic"),

    # ── master_artisan (2+ skills) ──
    ("(master_artisan venn)", True,
     "venn has 2 skills"),
    ("(master_artisan torvak)", True,
     "torvak has 3 skills"),
    ("(master_artisan lira)", False,
     "lira has 1 skill"),
    ("(master_artisan quill)", False,
     "quill has 1 skill"),

    # ── can_craft (artisan has required skill for recipe type) ──
    ("(can_craft venn solution_dl)", True,
     "venn has solution_craft"),
    ("(can_craft venn alloy_lv)", True,
     "venn has alloying"),
    ("(can_craft lira crystal_cn)", True,
     "lira has crystallography"),
    ("(can_craft lira crystal_cc)", True,
     "lira has crystallography"),
    ("(can_craft torvak vapor_si)", True,
     "torvak has vapor_craft"),
    ("(can_craft torvak composite_pn)", True,
     "torvak has composite_work"),
    ("(can_craft venn crystal_cn)", False,
     "venn lacks crystallography"),
    ("(can_craft lira alloy_nn)", False,
     "lira lacks alloying"),
    ("(can_craft quill crystal_cc)", False,
     "quill lacks crystallography"),
]


# ═══════════════════════════════════════════════════════════════════
# Imports from EX25 (shared helpers + family-tree baseline fixtures)
# ═══════════════════════════════════════════════════════════════════

from experiments.ex25_generalization import (
    build_kb, is_derivable, dream_kb,
    build_family_tree, family_base_facts, family_derived_facts,
    family_negatives, NEW_FAMILY_BASE, NEW_ENTITY_CHECKS,
)


# ═══════════════════════════════════════════════════════════════════
# RAW LLM BASELINE
# ═══════════════════════════════════════════════════════════════════

def run_raw_llm_check(llm_client: LLMClient,
                      kb_facts: List[str],
                      new_base: List[str],
                      query: str, desc: str) -> bool:
    """Ask the LLM directly whether a query holds, given all facts."""
    all_facts = kb_facts + new_base
    fact_block = "\n".join(all_facts[:300])  # cap at 300 for prompt size
    prompt = (
        f"Given these facts from a knowledge base:\n\n{fact_block}\n\n"
        f"Based ONLY on these facts and logical reasoning, "
        f"is the following derivable? Answer YES or NO only.\n\n"
        f"Query: {query}\n\nAnswer:"
    )
    try:
        response = llm_client.complete(prompt).strip().upper()
        return response.startswith("YES")
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════

def evaluate_checks(kb: KnowledgeBase, new_base: List[str],
                    new_checks: list) -> Dict:
    """Add new entities to KB and evaluate checks."""
    for fact in new_base:
        kb.add_fact(Fact(parse_s_expression(fact)))

    tp, tn, fp, fn = 0, 0, 0, 0
    detail = []
    for query, expected, desc in new_checks:
        got = is_derivable(kb, query)
        if expected and got:
            tp += 1
        elif expected and not got:
            fn += 1
        elif not expected and not got:
            tn += 1
        else:
            fp += 1
        status = "PASS" if got == expected else "FAIL"
        detail.append((status, query, desc))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "detail": detail,
    }


def run_domain_test(domain_name: str, base: List[str],
                    derived: Dict[str, List[str]],
                    negatives: List[Tuple[str, bool]],
                    new_base: List[str],
                    new_checks: list,
                    llm_client: Optional[LLMClient],
                    n_runs: int = 1,
                    discover_recursion: bool = False):
    """Run generalization test on a domain with optional multi-run."""
    print(f"\n{'─'*72}")
    print(f"  Domain: {domain_name}")
    print(f"{'─'*72}")

    all_derived = [f for fs in derived.values() for f in fs]
    all_facts_str = base + all_derived
    print(f"  Base facts: {len(base)}")
    print(f"  Derived facts: {len(all_derived)}")
    for pred, fs in sorted(derived.items()):
        print(f"    {pred}: {len(fs)}")
    pos_checks = sum(1 for _, exp, _ in new_checks if exp)
    neg_checks = sum(1 for _, exp, _ in new_checks if not exp)
    print(f"  New-entity checks: {len(new_checks)} "
          f"({pos_checks} positive, {neg_checks} negative)")
    if n_runs > 1:
        print(f"  Runs per condition: {n_runs}")

    conditions = [
        ("no_dream", None),
        ("symbolic", None),
        ("full", llm_client),
    ]

    # Add raw LLM baseline if client available
    if llm_client:
        conditions.append(("raw_llm", llm_client))

    results = {}

    for condition, client in conditions:

        # Collect runs
        run_results = []
        for run_i in range(n_runs):
            if condition == "raw_llm":
                # Direct LLM prompting, no compression
                tp, tn, fp, fn = 0, 0, 0, 0
                for query, expected, desc in new_checks:
                    got = run_raw_llm_check(
                        client, all_facts_str, new_base, query, desc)
                    if expected and got:
                        tp += 1
                    elif expected and not got:
                        fn += 1
                    elif not expected and not got:
                        tn += 1
                    else:
                        fp += 1
                total = tp + tn + fp + fn
                run_results.append({
                    "rules": 0, "compression": 1.0,
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "accuracy": (tp + tn) / total if total else 0,
                    "precision": tp / (tp + fp) if (tp + fp) else 0,
                    "recall": tp / (tp + fn) if (tp + fn) else 0,
                    "rule_list": [], "detail": [],
                })
            else:
                kb = build_kb(base + all_derived)
                before = len(kb)
                rules = []
                ratio = 1.0

                if condition != "no_dream":
                    ratio, rules = dream_kb(kb, llm_client=client,
                                            discover_recursion=discover_recursion)

                res = evaluate_checks(kb, new_base, new_checks)
                res["rules"] = len(rules)
                res["compression"] = ratio
                res["rule_list"] = rules
                run_results.append(res)

        # Aggregate
        if n_runs == 1:
            r = run_results[0]
            results[condition] = r
            if condition not in ("no_dream", "raw_llm") and r.get("rule_list"):
                print(f"\n  [{condition}] {len(r['rule_list'])} rules, "
                      f"compression {r['compression']:.3f}")
                for rule in r["rule_list"][:5]:
                    print(f"    + {rule}")
            print(f"\n  [{condition}] TP={r['tp']} TN={r['tn']} "
                  f"FP={r['fp']} FN={r['fn']}")
            print(f"    Acc={r['accuracy']:.1%}  "
                  f"Prec={r['precision']:.1%}  "
                  f"Recall={r['recall']:.1%}")
        else:
            accs = [r["accuracy"] for r in run_results]
            recs = [r["recall"] for r in run_results]
            precs = [r["precision"] for r in run_results]
            mean_acc = sum(accs) / len(accs)
            mean_rec = sum(recs) / len(recs)
            mean_prec = sum(precs) / len(precs)
            std_acc = (sum((x - mean_acc)**2 for x in accs) / len(accs))**0.5
            std_rec = (sum((x - mean_rec)**2 for x in recs) / len(recs))**0.5
            std_prec = (sum((x - mean_prec)**2 for x in precs) / len(precs))**0.5
            mean_rules = sum(r["rules"] for r in run_results) / len(run_results)

            results[condition] = {
                "accuracy": mean_acc, "std_accuracy": std_acc,
                "recall": mean_rec, "std_recall": std_rec,
                "precision": mean_prec, "std_precision": std_prec,
                "rules": mean_rules,
                "compression": run_results[0]["compression"],
                "runs": run_results,
            }
            print(f"\n  [{condition}] ({n_runs} runs)")
            print(f"    Acc={mean_acc:.1%} +/- {std_acc:.1%}  "
                  f"Recall={mean_rec:.1%} +/- {std_rec:.1%}  "
                  f"Rules={mean_rules:.1f}")

    # Summary
    print(f"\n  {'Condition':<15} {'Rules':>5} {'Acc':>7} {'Prec':>7} "
          f"{'Recall':>7} {'Comp':>7}")
    print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cond, r in results.items():
        acc = r["accuracy"]
        prec = r.get("precision", 0)
        rec = r["recall"]
        rules = r["rules"]
        comp = r.get("compression", 1.0)
        std_str = ""
        if "std_recall" in r:
            std_str = f" +/- {r['std_recall']:.1%}"
        print(f"  {cond:<15} {rules:>5.0f} {acc:>6.1%} "
              f"{prec:>6.1%} {rec:>5.1%}{std_str:>8} "
              f"{comp:>6.3f}")

    return results


def run_experiment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--family-only", action="store_true")
    parser.add_argument("--crafting-only", action="store_true")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per LLM condition for variance")
    args = parser.parse_args()

    import os
    llm_client = None
    if os.getenv(args.api_key_env):
        llm_client = LLMClient(provider="anthropic", api_key_env=args.api_key_env,
                               temperature=0.3, max_tokens=1200)

    print(f"{'='*72}")
    print(f"  EX25b: Novel Domain Generalization")
    print(f"  LLM: {'Anthropic Haiku' if llm_client else 'NONE'}")
    print(f"  Runs per LLM condition: {args.runs}")
    print(f"{'='*72}")

    all_results = {}
    t_total = time.perf_counter()

    # ── Crafting domain (novel) ───────────────────────────────────
    if not args.family_only:
        c_base = crafting_base_facts()
        c_derived = crafting_derived_facts()
        c_negs = crafting_negatives()
        all_results["crafting"] = run_domain_test(
            "Synthetic Crafting (novel)",
            c_base, c_derived, c_negs,
            NEW_CRAFTING_BASE, NEW_CRAFTING_CHECKS,
            llm_client, n_runs=args.runs)

    # ── Family tree (canonical baseline) ──────────────────────────
    if not args.crafting_only:
        tree = build_family_tree()
        f_base = family_base_facts(tree)
        f_derived = family_derived_facts(tree)
        f_negs = family_negatives(tree)
        all_results["family"] = run_domain_test(
            "Family Tree (canonical baseline)",
            f_base, f_derived, f_negs,
            NEW_FAMILY_BASE, NEW_ENTITY_CHECKS,
            llm_client, n_runs=args.runs)

    elapsed = time.perf_counter() - t_total

    # ── Comparison ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  COMPARISON: Canonical vs Novel Domain")
    print(f"{'='*72}")

    print(f"\n  {'Domain':<25} {'Condition':<12} {'Acc':>7} "
          f"{'Recall':>7} {'Rules':>6} {'Comp':>7}")
    print(f"  {'-'*25} {'-'*12} {'-'*7} {'-'*7} {'-'*6} {'-'*7}")

    for domain, results in all_results.items():
        for cond in ["no_dream", "symbolic", "full", "raw_llm"]:
            if cond not in results:
                continue
            r = results[cond]
            rules = r["rules"]
            std_str = ""
            if "std_recall" in r:
                std_str = f" +/-{r['std_recall']:.0%}"
            print(f"  {domain:<25} {cond:<12} {r['accuracy']:>6.1%} "
                  f"{r['recall']:>5.1%}{std_str:<8} {rules:>5.0f} "
                  f"{r.get('compression', 1.0):>6.3f}")
        print()

    if llm_client:
        u = llm_client.usage
        c = llm_client.estimated_cost()
        print(f"  LLM: {u.calls} calls, "
              f"{u.input_tokens:,} in / {u.output_tokens:,} out "
              f"(${c:.4f})")

    print(f"  Time: {elapsed:.1f}s")

    # Save a committed artifact for provenance (like EX27/EX28); the previous
    # /tmp path was ephemeral and left the 5-run figures unauditable.
    import subprocess as _sp
    _gp = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                  capture_output=True, text=True)
    git_sha = (_gp.stdout.strip() if _gp.returncode == 0 else "") or "unknown"

    def _clean(d):
        return {k: v for k, v in d.items() if k not in ("detail", "rule_list")}

    results_path = Path("experiments/data/ex25b/results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save = {"git_sha": git_sha,
            "model": (llm_client.model if llm_client else None),
            "runs_per_condition": args.runs, "ts": time.time(),
            "domains": {}}
    for domain, results in all_results.items():
        save["domains"][domain] = {}
        for cond, r in results.items():
            cr = _clean(r)
            if "runs" in cr:
                cr["runs"] = [_clean(x) for x in cr["runs"]]
            save["domains"][domain][cond] = cr
    results_path.write_text(json.dumps(save, indent=2))
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    run_experiment()
