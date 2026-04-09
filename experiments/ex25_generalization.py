#!/usr/bin/env python3
"""
EX25: Compression Predicts Generalization.

Tests the core theoretical claim of the paper: if compression = learning
(Solomonoff induction), then a compressed KB should derive facts it never saw.

Four components:
  1. Generalization — dream on full KB, add unseen entities with only base
     facts, check if derived relationships are derivable via discovered rules.
     This is zero-shot concept transfer to new individuals.
  2. Holdout — stratified 80/20 split of derived facts, dream on training
     set, measure recovery rate on held-out facts.
  3. Ablation — compare no-dream / symbolic-only / full pipeline across
     multiple random splits.
  4. Transfer — dream on family + org combined, check if isomorphic
     structure leads to shared abstractions or higher generalization.

KBs:
  - Family tree: 4 generations, 29 people, ~96 base facts, ~160 derived
    facts across 7 predicates (father, mother, grandparent, grandfather,
    grandmother, great_grandparent, ancestor).
  - Org chart: 3-level hierarchy, 15 employees, isomorphic to a family
    (manages≈parent, technical≈male). Derived: tech_lead, people_manager,
    skip_level, chain_of_command.

Usage:
    python experiments/ex25_generalization.py
    python experiments/ex25_generalization.py --splits 5
    python experiments/ex25_generalization.py --no-transfer
"""

import sys
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Compound, Atom
from dreamlog.prefix_parser import parse_s_expression
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient


# ═══════════════════════════════════════════════════════════════════
# PART 1: KB GENERATORS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Person:
    name: str
    gender: str  # "m" or "f"
    parents: List[str] = field(default_factory=list)


# ── Family tree ───────────────────────────────────────────────────

def build_family_tree() -> Dict[str, Person]:
    """4-generation family, 29 people, 2 branches joined by marriage."""
    t = {}

    def add(name, gender, parents=None):
        t[name] = Person(name, gender, parents or [])

    # Gen 0 — great-grandparents
    add("albert", "m");  add("martha", "f")
    add("charles", "m"); add("dorothy", "f")

    # Gen 1 — grandparents (born) + spouses
    add("henry", "m", ["albert", "martha"])
    add("margaret", "f", ["albert", "martha"])
    add("william", "m", ["albert", "martha"])
    add("patricia", "f", ["charles", "dorothy"])
    add("robert", "m", ["charles", "dorothy"])
    add("elizabeth", "f")   # marries henry
    add("catherine", "f")   # marries william
    # margaret marries robert (cross-branch)

    # Gen 2 — parents (born) + spouses
    add("james", "m", ["henry", "elizabeth"])
    add("sarah", "f", ["henry", "elizabeth"])
    add("thomas", "m", ["henry", "elizabeth"])
    add("michael", "m", ["robert", "margaret"])
    add("jennifer", "f", ["robert", "margaret"])
    add("david", "m", ["william", "catherine"])
    add("emily", "f", ["william", "catherine"])
    add("daniel", "m", ["william", "catherine"])
    add("lisa", "f")        # marries james
    add("rachel", "f")      # marries thomas
    add("amanda", "f")      # marries michael
    add("christine", "f")   # marries david

    # Gen 3 — current generation
    add("emma", "f", ["james", "lisa"])
    add("jack", "m", ["james", "lisa"])
    add("oliver", "m", ["thomas", "rachel"])
    add("sophia", "f", ["michael", "amanda"])
    add("noah", "m", ["michael", "amanda"])
    add("lucas", "m", ["david", "christine"])

    return t


def family_base_facts(tree: Dict[str, Person]) -> List[str]:
    facts = []
    for name, p in tree.items():
        facts.append(f"(person {name})")
        facts.append(f"({'male' if p.gender == 'm' else 'female'} {name})")
        for par in p.parents:
            facts.append(f"(parent {par} {name})")
    return facts


def family_derived_facts(tree: Dict[str, Person]) -> Dict[str, List[str]]:
    """Derived facts grouped by predicate, computed from the tree."""
    d: Dict[str, List[str]] = {
        "father": [], "mother": [],
        "grandparent": [], "grandfather": [], "grandmother": [],
        "great_grandparent": [], "ancestor": [],
    }

    for name, p in tree.items():
        for par in p.parents:
            if tree[par].gender == "m":
                d["father"].append(f"(father {par} {name})")
            else:
                d["mother"].append(f"(mother {par} {name})")

        for par in p.parents:
            for gp in tree[par].parents:
                d["grandparent"].append(f"(grandparent {gp} {name})")
                if tree[gp].gender == "m":
                    d["grandfather"].append(f"(grandfather {gp} {name})")
                else:
                    d["grandmother"].append(f"(grandmother {gp} {name})")

        for par in p.parents:
            for gp in tree[par].parents:
                for ggp in tree[gp].parents:
                    d["great_grandparent"].append(
                        f"(great_grandparent {ggp} {name})")

    # Ancestors — transitive closure of parent
    def ancestors(name, visited=None):
        if visited is None:
            visited = set()
        for par in tree[name].parents:
            if par not in visited:
                visited.add(par)
                ancestors(par, visited)
        return visited

    for name in tree:
        for anc in ancestors(name):
            d["ancestor"].append(f"(ancestor {anc} {name})")

    return d


def family_negatives(tree: Dict[str, Person]) -> List[Tuple[str, bool]]:
    """Negative examples: facts that should NOT be derivable."""
    negs = []
    for name, p in tree.items():
        for par in p.parents:
            # Wrong gender role
            if tree[par].gender == "m":
                negs.append(f"(mother {par} {name})")
            else:
                negs.append(f"(father {par} {name})")

    # Reverse direction: child as grandparent of ancestor
    negs.append("(grandparent emma albert)")
    negs.append("(father jack james)")  # jack is child of james, not other way
    negs.append("(grandfather emma henry)")  # emma is female

    return [(n, False) for n in negs[:25]]


# ── New entities for generalization test ──────────────────────────

NEW_FAMILY_BASE = [
    "(person new_gfather)", "(male new_gfather)",
    "(person new_gmother)", "(female new_gmother)",
    "(person new_dad)", "(male new_dad)",
    "(parent new_gfather new_dad)", "(parent new_gmother new_dad)",
    "(person new_mom)", "(female new_mom)",
    "(person new_son)", "(male new_son)",
    "(parent new_dad new_son)", "(parent new_mom new_son)",
    "(person new_daughter)", "(female new_daughter)",
    "(parent new_dad new_daughter)", "(parent new_mom new_daughter)",
]

# (query, expected_derivable, description)
NEW_ENTITY_CHECKS = [
    # Father / Mother
    ("(father new_dad new_son)", True, "father from parent+male"),
    ("(father new_dad new_daughter)", True, "father from parent+male"),
    ("(mother new_mom new_son)", True, "mother from parent+female"),
    ("(mother new_mom new_daughter)", True, "mother from parent+female"),
    ("(father new_mom new_son)", False, "female not father"),
    ("(mother new_dad new_son)", False, "male not mother"),
    # Grandparents
    ("(grandparent new_gfather new_son)", True, "grandparent 2-hop"),
    ("(grandparent new_gmother new_son)", True, "grandparent 2-hop"),
    ("(grandfather new_gfather new_son)", True, "grandfather 2-hop+male"),
    ("(grandmother new_gmother new_son)", True, "grandmother 2-hop+female"),
    ("(grandfather new_gmother new_son)", False, "female not grandfather"),
    ("(grandmother new_gfather new_son)", False, "male not grandmother"),
    # Ancestor (requires transitive or depth-specific rule)
    ("(ancestor new_dad new_son)", True, "ancestor depth-1"),
    ("(ancestor new_gfather new_son)", True, "ancestor depth-2"),
]


# ── Org chart (isomorphic to family) ──────────────────────────────

def build_org_chart() -> Dict[str, Person]:
    """3-level org hierarchy, 15 employees. Isomorphic to family tree."""
    o = {}

    def add(name, emp_type, reports_to=None):
        # Re-use Person: gender="m" → technical, "f" → nontechnical
        o[name] = Person(name, emp_type, reports_to or [])

    # Top level
    add("ceo_pat", "m")

    # VP level
    add("vp_eng", "m", ["ceo_pat"])
    add("vp_ops", "f", ["ceo_pat"])
    add("vp_sales", "f", ["ceo_pat"])

    # Director level
    add("dir_backend", "m", ["vp_eng"])
    add("dir_frontend", "m", ["vp_eng"])
    add("dir_infra", "m", ["vp_eng"])
    add("dir_hr", "f", ["vp_ops"])
    add("dir_finance", "f", ["vp_ops"])
    add("dir_accounts", "f", ["vp_sales"])

    # IC level
    add("eng_a", "m", ["dir_backend"])
    add("eng_b", "m", ["dir_backend"])
    add("eng_c", "m", ["dir_frontend"])
    add("eng_d", "m", ["dir_infra"])
    add("hr_rep", "f", ["dir_hr"])

    return o


def org_base_facts(org: Dict[str, Person]) -> List[str]:
    facts = []
    for name, p in org.items():
        facts.append(f"(employee {name})")
        facts.append(f"({'technical' if p.gender == 'm' else 'nontechnical'} {name})")
        for mgr in p.parents:
            facts.append(f"(manages {mgr} {name})")
    return facts


def org_derived_facts(org: Dict[str, Person]) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {
        "tech_lead": [], "people_manager": [],
        "skip_level": [], "chain_of_command": [],
    }

    for name, p in org.items():
        for mgr in p.parents:
            if org[mgr].gender == "m":
                d["tech_lead"].append(f"(tech_lead {mgr} {name})")
            else:
                d["people_manager"].append(f"(people_manager {mgr} {name})")

        for mgr in p.parents:
            for skip in org[mgr].parents:
                d["skip_level"].append(f"(skip_level {skip} {name})")

    def chain(name, visited=None):
        if visited is None:
            visited = set()
        for mgr in org[name].parents:
            if mgr not in visited:
                visited.add(mgr)
                chain(mgr, visited)
        return visited

    for name in org:
        for mgr in chain(name):
            d["chain_of_command"].append(
                f"(chain_of_command {mgr} {name})")
    return d


# ═══════════════════════════════════════════════════════════════════
# PART 2: HELPERS
# ═══════════════════════════════════════════════════════════════════

def build_kb(fact_strings: List[str]) -> KnowledgeBase:
    kb = KnowledgeBase()
    for s in fact_strings:
        if ":-" in s:
            parts = s.split(":-")
            head = parse_s_expression(parts[0].strip())
            body = [parse_s_expression(b.strip())
                    for b in parts[1].split(",")]
            kb.add_rule(Rule(head, body))
        else:
            kb.add_fact(Fact(parse_s_expression(s)))
    return kb


def is_derivable(kb: KnowledgeBase, query: str,
                 max_calls: int = 10000) -> bool:
    term = parse_s_expression(query)
    ev = PrologEvaluator(kb, max_total_calls=max_calls)
    return ev.has_solution(term)


def dream_kb(kb: KnowledgeBase, llm_client: Optional[LLMClient] = None,
             max_prompt_facts: int = 200,
             open_world: bool = False,
             ) -> Tuple[float, List[str]]:
    """Dream on a KB. Returns (compression_ratio, list of new rule strings)."""
    dreamer = KnowledgeBaseDreamer(llm_client=llm_client,
                                   max_prompt_facts=max_prompt_facts,
                                   open_world=open_world)
    session = dreamer.dream(kb, verify=True)
    rules = []
    for op in session.operations:
        for c in op.new_clauses:
            if isinstance(c, Rule):
                rules.append(str(c))
    return session.compression_ratio, rules


def holdout_split(derived_by_pred: Dict[str, List[str]],
                  ratio: float = 0.2, seed: int = 42
                  ) -> Tuple[List[str], List[str]]:
    """Stratified holdout: hold out `ratio` of each predicate's facts."""
    rng = random.Random(seed)
    train, test = [], []
    for pred, facts in derived_by_pred.items():
        shuffled = list(facts)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * ratio))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return train, test


def get_llm_client(args) -> Optional[LLMClient]:
    import os
    env = args.api_key_env
    if not os.getenv(env):
        print(f"  WARNING: {env} not set, LLM tests will use symbolic only")
        return None
    return LLMClient(provider="anthropic", api_key_env=env,
                     temperature=0.3, max_tokens=1200)


# ═══════════════════════════════════════════════════════════════════
# PART 3: GENERALIZATION TEST (new entities)
# ═══════════════════════════════════════════════════════════════════

def run_generalization_test(llm_client: Optional[LLMClient]):
    """Dream on family KB, add unseen entities, check derived facts."""
    print(f"\n{'─'*72}")
    print(f"  PART A: Zero-Shot Generalization (new entities)")
    print(f"{'─'*72}")

    tree = build_family_tree()
    base = family_base_facts(tree)
    derived = family_derived_facts(tree)
    all_derived = [f for fs in derived.values() for f in fs]

    print(f"  Family: {len(tree)} people, {len(base)} base facts, "
          f"{len(all_derived)} derived facts")
    for pred, fs in derived.items():
        print(f"    {pred}: {len(fs)}")

    results = {}

    for condition, client in [("no_dream", None),
                              ("symbolic", None),
                              ("full", llm_client)]:
        # Build KB with all facts
        kb = build_kb(base + all_derived)
        before = len(kb)

        # Dream (unless baseline)
        rules = []
        ratio = 1.0
        if condition != "no_dream":
            t0 = time.perf_counter()
            ratio, rules = dream_kb(kb, llm_client=client)
            elapsed = time.perf_counter() - t0
            print(f"\n  [{condition}] Dream: {before} → {len(kb)} "
                  f"(ratio {ratio:.3f}) in {elapsed:.1f}s, "
                  f"{len(rules)} rules")
            for r in rules[:5]:
                print(f"    + {r}")
            if len(rules) > 5:
                print(f"    ... +{len(rules)-5} more")

        # Add new entities (base facts only)
        for fact in NEW_FAMILY_BASE:
            kb.add_fact(Fact(parse_s_expression(fact)))

        # Evaluate
        tp, tn, fp, fn = 0, 0, 0, 0
        detail = []
        for query, expected, desc in NEW_ENTITY_CHECKS:
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

        results[condition] = {
            "rules": len(rules),
            "compression": ratio,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "rule_list": rules,
            "detail": detail,
        }

        print(f"\n  [{condition}] New-entity results:")
        print(f"    TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"    Accuracy={accuracy:.1%}  Precision={precision:.1%}  "
              f"Recall={recall:.1%}")

        for status, query, desc in detail:
            if status == "FAIL":
                print(f"    {status}: {query}  [{desc}]")

    # Summary table
    print(f"\n  {'Condition':<15} {'Rules':>5} {'Acc':>7} {'Prec':>7} "
          f"{'Recall':>7} {'Comp':>7}")
    print(f"  {'-'*15} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for cond, r in results.items():
        print(f"  {cond:<15} {r['rules']:>5} {r['accuracy']:>6.1%} "
              f"{r['precision']:>6.1%} {r['recall']:>6.1%} "
              f"{r['compression']:>6.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# PART 4: HOLDOUT TEST WITH ABLATION
# ═══════════════════════════════════════════════════════════════════

def run_holdout_ablation(llm_client: Optional[LLMClient],
                         n_splits: int = 3):
    """Holdout 20% of derived facts, dream, measure recovery."""
    print(f"\n{'─'*72}")
    print(f"  PART B: Holdout Recovery + Ablation ({n_splits} splits)")
    print(f"{'─'*72}")

    tree = build_family_tree()
    base = family_base_facts(tree)
    derived = family_derived_facts(tree)
    negatives = family_negatives(tree)

    all_derived = [f for fs in derived.values() for f in fs]
    print(f"  Base: {len(base)}, Derived: {len(all_derived)}, "
          f"Negatives: {len(negatives)}")

    seeds = [42, 123, 456, 789, 1024][:n_splits]
    conditions = [
        ("no_dream", None),
        ("symbolic", None),
        ("full", llm_client),
    ]

    # condition → list of per-split results
    agg: Dict[str, list] = {c: [] for c, _ in conditions}

    for split_i, seed in enumerate(seeds):
        train_derived, test_derived = holdout_split(derived, 0.2, seed)
        print(f"\n  Split {split_i+1} (seed={seed}): "
              f"train={len(train_derived)}, test={len(test_derived)}")

        for cond_name, client in conditions:
            kb = build_kb(base + train_derived)
            before = len(kb)
            rules = []
            ratio = 1.0

            if cond_name != "no_dream":
                ratio, rules = dream_kb(kb, llm_client=client)
                print(f"    [{cond_name}] {before}→{len(kb)} "
                      f"({ratio:.3f}), {len(rules)} rules")

            # Recovery: how many held-out facts are derivable?
            recovered = 0
            for fact in test_derived:
                if is_derivable(kb, fact):
                    recovered += 1
            recovery_rate = recovered / len(test_derived) if test_derived else 0

            # False positives: how many negatives are incorrectly derivable?
            false_pos = 0
            for neg_query, _ in negatives:
                if is_derivable(kb, neg_query):
                    false_pos += 1
            fp_rate = false_pos / len(negatives) if negatives else 0

            agg[cond_name].append({
                "seed": seed,
                "recovery": recovery_rate,
                "recovered": recovered,
                "test_size": len(test_derived),
                "false_pos": false_pos,
                "fp_rate": fp_rate,
                "compression": ratio,
                "rules": len(rules),
                "rule_list": rules,
            })

            print(f"      Recovered {recovered}/{len(test_derived)} "
                  f"({recovery_rate:.1%}), FP={false_pos}/{len(negatives)} "
                  f"({fp_rate:.1%})")

    # Aggregate
    print(f"\n  Holdout Ablation Summary ({n_splits} splits):")
    print(f"  {'Condition':<15} {'Recovery':>12} {'FP Rate':>12} "
          f"{'Compress':>10} {'Rules':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

    summary = {}
    for cond_name, runs in agg.items():
        recs = [r["recovery"] for r in runs]
        fps = [r["fp_rate"] for r in runs]
        comps = [r["compression"] for r in runs]
        ruls = [r["rules"] for r in runs]

        mean_rec = sum(recs) / len(recs)
        mean_fp = sum(fps) / len(fps)
        mean_comp = sum(comps) / len(comps)
        mean_rules = sum(ruls) / len(ruls)

        # Std dev
        std_rec = (sum((x - mean_rec)**2 for x in recs) / len(recs))**0.5
        std_fp = (sum((x - mean_fp)**2 for x in fps) / len(fps))**0.5

        print(f"  {cond_name:<15} {mean_rec:>5.1%} ± {std_rec:>4.1%} "
              f"{mean_fp:>5.1%} ± {std_fp:>4.1%} "
              f"{mean_comp:>9.3f} {mean_rules:>7.1f}")

        summary[cond_name] = {
            "mean_recovery": mean_rec, "std_recovery": std_rec,
            "mean_fp_rate": mean_fp, "std_fp_rate": std_fp,
            "mean_compression": mean_comp,
            "mean_rules": mean_rules,
            "runs": runs,
        }

    return summary


# ═══════════════════════════════════════════════════════════════════
# PART 5: TRANSFER TEST (family + org combined)
# ═══════════════════════════════════════════════════════════════════

def run_transfer_test(llm_client: Optional[LLMClient]):
    """Compare isolated vs combined dreaming for isomorphic KBs."""
    print(f"\n{'─'*72}")
    print(f"  PART C: Transfer Learning (family + org)")
    print(f"{'─'*72}")

    if not llm_client:
        print("  SKIPPED (no LLM client)")
        return None

    # Build KBs
    family_tree = build_family_tree()
    family_b = family_base_facts(family_tree)
    family_d = family_derived_facts(family_tree)
    family_all = [f for fs in family_d.values() for f in fs]

    org = build_org_chart()
    org_b = org_base_facts(org)
    org_d = org_derived_facts(org)
    org_all = [f for fs in org_d.values() for f in fs]

    print(f"  Family: {len(family_b)} base + {len(family_all)} derived")
    print(f"  Org:    {len(org_b)} base + {len(org_all)} derived")

    results = {}

    # ── Isolated: dream org alone ─────────────────────────────────
    print(f"\n  [org_isolated] Dreaming org alone...")
    kb_org = build_kb(org_b + org_all)
    before = len(kb_org)
    t0 = time.perf_counter()
    ratio_org, rules_org = dream_kb(kb_org, llm_client)
    elapsed = time.perf_counter() - t0
    print(f"    {before}→{len(kb_org)} ({ratio_org:.3f}) in {elapsed:.1f}s, "
          f"{len(rules_org)} rules")
    for r in rules_org:
        print(f"    + {r}")

    # Check org derivations on new employees
    new_org = [
        "(employee new_mgr)", "(technical new_mgr)",
        "(employee new_ic)", "(technical new_ic)",
        "(manages new_mgr new_ic)",
    ]
    for f in new_org:
        kb_org.add_fact(Fact(parse_s_expression(f)))

    org_iso_checks = [
        ("(tech_lead new_mgr new_ic)", True, "tech_lead from manages+technical"),
        ("(people_manager new_mgr new_ic)", False, "technical, not people_manager"),
    ]

    org_iso_pass = 0
    for q, expected, desc in org_iso_checks:
        got = is_derivable(kb_org, q)
        ok = got == expected
        org_iso_pass += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"    {status}: {q}  [{desc}]")

    results["org_isolated"] = {
        "rules": rules_org,
        "compression": ratio_org,
        "generalization_pass": org_iso_pass,
        "generalization_total": len(org_iso_checks),
    }

    # ── Combined: dream family + org together ─────────────────────
    print(f"\n  [combined] Dreaming family + org together...")
    kb_combined = build_kb(family_b + family_all + org_b + org_all)
    before = len(kb_combined)
    t0 = time.perf_counter()
    ratio_comb, rules_comb = dream_kb(kb_combined, llm_client)
    elapsed = time.perf_counter() - t0
    print(f"    {before}→{len(kb_combined)} ({ratio_comb:.3f}) in "
          f"{elapsed:.1f}s, {len(rules_comb)} rules")
    for r in rules_comb[:10]:
        print(f"    + {r}")
    if len(rules_comb) > 10:
        print(f"    ... +{len(rules_comb)-10} more")

    # Check new org employees again
    for f in new_org:
        kb_combined.add_fact(Fact(parse_s_expression(f)))

    comb_pass = 0
    for q, expected, desc in org_iso_checks:
        got = is_derivable(kb_combined, q)
        ok = got == expected
        comb_pass += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"    {status}: {q}  [{desc}]")

    # Also check new family members
    for f in NEW_FAMILY_BASE:
        kb_combined.add_fact(Fact(parse_s_expression(f)))
    family_pass = 0
    family_checks = [c for c in NEW_ENTITY_CHECKS if c[1]]  # positives only
    for q, expected, desc in family_checks[:6]:
        got = is_derivable(kb_combined, q)
        ok = got == expected
        family_pass += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"    {status}: {q}  [{desc}]")

    results["combined"] = {
        "rules": rules_comb,
        "compression": ratio_comb,
        "org_generalization_pass": comb_pass,
        "org_generalization_total": len(org_iso_checks),
        "family_generalization_pass": family_pass,
        "family_generalization_total": len(family_checks[:6]),
    }

    # ── Transfer summary ──────────────────────────────────────────
    print(f"\n  Transfer Summary:")
    print(f"    Org isolated:  {len(rules_org)} rules, "
          f"{org_iso_pass}/{len(org_iso_checks)} generalization")
    print(f"    Combined:      {len(rules_comb)} rules, "
          f"{comb_pass}/{len(org_iso_checks)} org generalization, "
          f"{family_pass}/{len(family_checks[:6])} family generalization")

    # Check for shared abstractions (invented predicates used by both)
    invented = [r for r in rules_comb if "_invented_" in r or "_extracted_" in r]
    if invented:
        print(f"    Shared abstractions: {len(invented)}")
        for r in invented:
            print(f"      {r}")
    else:
        print(f"    No predicate-invention abstractions (rules are domain-specific)")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_experiment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--splits", type=int, default=3,
                        help="Number of random splits for holdout test")
    parser.add_argument("--no-transfer", action="store_true",
                        help="Skip transfer test")
    parser.add_argument("--store", default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    llm_client = get_llm_client(args)

    print(f"{'='*72}")
    print(f"  EX25: Compression Predicts Generalization")
    print(f"  LLM: {'Anthropic Haiku' if llm_client else 'NONE (symbolic only)'}")
    print(f"  Holdout splits: {args.splits}")
    print(f"  Transfer: {'yes' if not args.no_transfer else 'no'}")
    print(f"{'='*72}")

    all_results = {}
    t_total = time.perf_counter()

    # Part A: Generalization
    all_results["generalization"] = run_generalization_test(llm_client)

    # Part B: Holdout + Ablation
    all_results["holdout_ablation"] = run_holdout_ablation(
        llm_client, n_splits=args.splits)

    # Part C: Transfer
    if not args.no_transfer:
        all_results["transfer"] = run_transfer_test(llm_client)

    elapsed_total = time.perf_counter() - t_total

    # ── Final report ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  FINAL REPORT")
    print(f"{'='*72}")

    # Generalization headline
    gen = all_results["generalization"]
    for cond in ["no_dream", "symbolic", "full"]:
        r = gen[cond]
        print(f"  Generalization [{cond}]: "
              f"Acc={r['accuracy']:.0%} "
              f"(TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']})")

    # Holdout headline
    hold = all_results["holdout_ablation"]
    print()
    for cond in ["no_dream", "symbolic", "full"]:
        r = hold[cond]
        print(f"  Holdout [{cond}]: "
              f"Recovery={r['mean_recovery']:.0%} ± {r['std_recovery']:.0%}, "
              f"FP={r['mean_fp_rate']:.0%}")

    # Key result
    print()
    baseline_recall = gen["no_dream"]["recall"]
    full_recall = gen["full"]["recall"]
    baseline_holdout = hold["no_dream"]["mean_recovery"]
    full_holdout = hold["full"]["mean_recovery"]

    print(f"  KEY RESULT: Compression enables generalization")
    print(f"    New-entity recall:  {baseline_recall:.0%} → {full_recall:.0%} "
          f"(+{full_recall - baseline_recall:.0%})")
    print(f"    Holdout recovery:   {baseline_holdout:.0%} → {full_holdout:.0%} "
          f"(+{full_holdout - baseline_holdout:.0%})")

    # Budget
    if llm_client:
        u = llm_client.usage
        c = llm_client.estimated_cost()
        print(f"\n  LLM usage: {u.calls} calls, "
              f"{u.input_tokens:,} in / {u.output_tokens:,} out "
              f"(${c:.4f})")

    print(f"  Total time: {elapsed_total:.1f}s")

    # Save results
    results_path = Path(args.store) if args.store else Path(
        f"/tmp/ex25_results.json")
    # Serialize (strip non-JSON-serializable items)
    save_data = {
        "generalization": {
            cond: {k: v for k, v in r.items()
                   if k not in ("detail", "rule_list")}
            for cond, r in gen.items()
        },
        "holdout_ablation": {
            cond: {k: v for k, v in r.items() if k != "runs"}
            for cond, r in hold.items()
        },
    }
    results_path.write_text(json.dumps(save_data, indent=2))
    print(f"  Results: {results_path}")

    return all_results


if __name__ == "__main__":
    run_experiment()
