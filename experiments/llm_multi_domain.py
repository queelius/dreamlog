#!/usr/bin/env python3
"""
Multi-domain LLM sleep cycle experiments.

Tests LLM-assisted compression across diverse knowledge domains to
understand where it helps and where symbolic methods suffice.

Usage:
    python experiments/llm_multi_domain.py
    python experiments/llm_multi_domain.py --domain geography
    python experiments/llm_multi_domain.py --model qwen3:8b
"""

import sys
import time
import argparse
import json

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient


def domain_biology():
    """Biological taxonomy with class/order/species relationships."""
    kb = KnowledgeBase()

    # Taxonomy
    animals = {
        "dog":    {"class": "mammal", "order": "carnivora", "legs": "4", "domestic": "true"},
        "cat":    {"class": "mammal", "order": "carnivora", "legs": "4", "domestic": "true"},
        "wolf":   {"class": "mammal", "order": "carnivora", "legs": "4", "domestic": "false"},
        "horse":  {"class": "mammal", "order": "perissodactyla", "legs": "4", "domestic": "true"},
        "cow":    {"class": "mammal", "order": "artiodactyla", "legs": "4", "domestic": "true"},
        "eagle":  {"class": "bird", "order": "accipitriformes", "legs": "2", "domestic": "false"},
        "parrot": {"class": "bird", "order": "psittaciformes", "legs": "2", "domestic": "true"},
        "robin":  {"class": "bird", "order": "passeriformes", "legs": "2", "domestic": "false"},
        "snake":  {"class": "reptile", "order": "squamata", "legs": "0", "domestic": "false"},
        "turtle": {"class": "reptile", "order": "testudines", "legs": "4", "domestic": "true"},
        "frog":   {"class": "amphibian", "order": "anura", "legs": "4", "domestic": "false"},
        "salmon": {"class": "fish", "order": "salmoniformes", "legs": "0", "domestic": "false"},
    }

    for name, props in animals.items():
        kb.add_fact(compound("animal", atom(name)))
        for prop, val in props.items():
            kb.add_fact(compound(prop, atom(name), atom(val)))

    # Derivable relationships: warm_blooded = mammal or bird
    for name, props in animals.items():
        if props["class"] in ("mammal", "bird"):
            kb.add_fact(compound("warm_blooded", atom(name)))

    # Derivable: can_fly = bird (except some, but we'll keep it simple)
    for name, props in animals.items():
        if props["class"] == "bird":
            kb.add_fact(compound("can_fly", atom(name)))

    # Derivable: pet = domestic + animal
    for name, props in animals.items():
        if props["domestic"] == "true":
            kb.add_fact(compound("pet", atom(name)))

    # 3 transitive closures for Op D
    for head, base in [("broader_than", "class"), ("related_order", "order")]:
        pass  # These don't quite work as transitive closures

    checks = [
        ("animal(dog)", compound("animal", atom("dog")), True),
        ("warm_blooded(dog)", compound("warm_blooded", atom("dog")), True),
        ("warm_blooded(snake)", compound("warm_blooded", atom("snake")), False),
        ("can_fly(eagle)", compound("can_fly", atom("eagle")), True),
        ("can_fly(dog)", compound("can_fly", atom("dog")), False),
        ("pet(cat)", compound("pet", atom("cat")), True),
        ("pet(wolf)", compound("pet", atom("wolf")), False),
    ]
    return "biology", kb, checks


def domain_geography():
    """Countries, continents, and derivable relationships."""
    kb = KnowledgeBase()

    countries = {
        "usa":       {"continent": "north_america", "hemisphere": "northern", "language": "english"},
        "canada":    {"continent": "north_america", "hemisphere": "northern", "language": "english"},
        "mexico":    {"continent": "north_america", "hemisphere": "northern", "language": "spanish"},
        "brazil":    {"continent": "south_america", "hemisphere": "southern", "language": "portuguese"},
        "argentina": {"continent": "south_america", "hemisphere": "southern", "language": "spanish"},
        "uk":        {"continent": "europe", "hemisphere": "northern", "language": "english"},
        "france":    {"continent": "europe", "hemisphere": "northern", "language": "french"},
        "germany":   {"continent": "europe", "hemisphere": "northern", "language": "german"},
        "japan":     {"continent": "asia", "hemisphere": "northern", "language": "japanese"},
        "australia": {"continent": "oceania", "hemisphere": "southern", "language": "english"},
    }

    for name, props in countries.items():
        kb.add_fact(compound("country", atom(name)))
        for prop, val in props.items():
            kb.add_fact(compound(prop, atom(name), atom(val)))

    # Derivable: english_speaking = language is english
    for name, props in countries.items():
        if props["language"] == "english":
            kb.add_fact(compound("english_speaking", atom(name)))

    # Derivable: northern_country = hemisphere is northern
    for name, props in countries.items():
        if props["hemisphere"] == "northern":
            kb.add_fact(compound("northern_country", atom(name)))

    # Derivable: american_country = continent is north_america or south_america
    for name, props in countries.items():
        if "america" in props["continent"]:
            kb.add_fact(compound("american_country", atom(name)))

    # Neighbors (for potential transitive closure)
    neighbors = [("usa","canada"),("usa","mexico"),("france","germany"),
                 ("brazil","argentina")]
    for a, b in neighbors:
        kb.add_fact(compound("neighbor", atom(a), atom(b)))
        kb.add_fact(compound("neighbor", atom(b), atom(a)))

    checks = [
        ("english_speaking(usa)", compound("english_speaking", atom("usa")), True),
        ("english_speaking(france)", compound("english_speaking", atom("france")), False),
        ("northern_country(japan)", compound("northern_country", atom("japan")), True),
        ("northern_country(brazil)", compound("northern_country", atom("brazil")), False),
        ("american_country(brazil)", compound("american_country", atom("brazil")), True),
        ("american_country(japan)", compound("american_country", atom("japan")), False),
    ]
    return "geography", kb, checks


def domain_university():
    """University courses, students, and derivable relationships."""
    kb = KnowledgeBase()

    students = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry"]
    for s in students:
        kb.add_fact(compound("student", atom(s)))

    courses = ["math", "physics", "chemistry", "cs", "english", "history"]
    for c in courses:
        kb.add_fact(compound("course", atom(c)))

    # Department mapping
    stem = ["math", "physics", "chemistry", "cs"]
    humanities = ["english", "history"]
    for c in stem:
        kb.add_fact(compound("department", atom(c), atom("stem")))
    for c in humanities:
        kb.add_fact(compound("department", atom(c), atom("humanities")))

    # Enrollments
    enrollments = [
        ("alice", "math"), ("alice", "physics"), ("alice", "cs"),
        ("bob", "math"), ("bob", "cs"),
        ("carol", "physics"), ("carol", "chemistry"),
        ("dave", "english"), ("dave", "history"),
        ("eve", "math"), ("eve", "english"),
        ("frank", "cs"), ("frank", "physics"),
        ("grace", "chemistry"), ("grace", "math"),
        ("henry", "english"), ("henry", "history"), ("henry", "math"),
    ]
    for s, c in enrollments:
        kb.add_fact(compound("enrolled", atom(s), atom(c)))

    # Derivable: stem_student = enrolled in a STEM course
    stem_students = set()
    for s, c in enrollments:
        if c in stem:
            stem_students.add(s)
    for s in stem_students:
        kb.add_fact(compound("stem_student", atom(s)))

    # Derivable: classmates = enrolled in same course
    from collections import defaultdict
    course_students = defaultdict(set)
    for s, c in enrollments:
        course_students[c].add(s)
    classmate_pairs = set()
    for c, students_in_course in course_students.items():
        for s1 in students_in_course:
            for s2 in students_in_course:
                if s1 < s2:
                    classmate_pairs.add((s1, s2))
    for s1, s2 in classmate_pairs:
        kb.add_fact(compound("classmates", atom(s1), atom(s2)))

    # Prerequisite chain (for transitive closure)
    prereqs = [("physics", "math"), ("chemistry", "physics"), ("cs", "math")]
    for c1, c2 in prereqs:
        kb.add_fact(compound("prereq", atom(c1), atom(c2)))

    # 3 structurally identical transitive closures (Op D)
    for head, base in [("requires", "prereq"), ("depends_on", "prereq"),
                       ("needs", "prereq")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    checks = [
        ("stem_student(alice)", compound("stem_student", atom("alice")), True),
        ("stem_student(dave)", compound("stem_student", atom("dave")), False),
        ("classmates(alice, bob)", compound("classmates", atom("alice"), atom("bob")), True),
        ("enrolled(alice, math)", compound("enrolled", atom("alice"), atom("math")), True),
        ("requires(chemistry, math)", compound("requires", atom("chemistry"), atom("math")), True),
    ]
    return "university", kb, checks


def domain_cooking():
    """Recipes, ingredients, and derivable dietary properties."""
    kb = KnowledgeBase()

    ingredients = {
        "flour": {"type": "grain", "vegan": "true", "gluten_free": "false"},
        "sugar": {"type": "sweetener", "vegan": "true", "gluten_free": "true"},
        "butter": {"type": "dairy", "vegan": "false", "gluten_free": "true"},
        "egg": {"type": "protein", "vegan": "false", "gluten_free": "true"},
        "milk": {"type": "dairy", "vegan": "false", "gluten_free": "true"},
        "rice": {"type": "grain", "vegan": "true", "gluten_free": "true"},
        "tofu": {"type": "protein", "vegan": "true", "gluten_free": "true"},
        "chicken": {"type": "meat", "vegan": "false", "gluten_free": "true"},
        "olive_oil": {"type": "fat", "vegan": "true", "gluten_free": "true"},
        "tomato": {"type": "vegetable", "vegan": "true", "gluten_free": "true"},
    }

    for name, props in ingredients.items():
        kb.add_fact(compound("ingredient", atom(name)))
        for prop, val in props.items():
            kb.add_fact(compound(prop, atom(name), atom(val)))

    # Recipes with ingredients
    recipes = {
        "cake": ["flour", "sugar", "butter", "egg"],
        "stir_fry": ["rice", "tofu", "olive_oil", "tomato"],
        "omelette": ["egg", "butter", "tomato"],
        "salad": ["tomato", "olive_oil"],
    }

    for recipe, ingrs in recipes.items():
        kb.add_fact(compound("recipe", atom(recipe)))
        for ing in ingrs:
            kb.add_fact(compound("uses", atom(recipe), atom(ing)))

    # Derivable: vegan_recipe = all ingredients are vegan
    for recipe, ingrs in recipes.items():
        if all(ingredients[i]["vegan"] == "true" for i in ingrs):
            kb.add_fact(compound("vegan_recipe", atom(recipe)))

    # Derivable: contains_dairy = uses a dairy ingredient
    for recipe, ingrs in recipes.items():
        if any(ingredients[i]["type"] == "dairy" for i in ingrs):
            kb.add_fact(compound("contains_dairy", atom(recipe)))

    checks = [
        ("vegan_recipe(stir_fry)", compound("vegan_recipe", atom("stir_fry")), True),
        ("vegan_recipe(cake)", compound("vegan_recipe", atom("cake")), False),
        ("contains_dairy(cake)", compound("contains_dairy", atom("cake")), True),
        ("contains_dairy(salad)", compound("contains_dairy", atom("salad")), False),
        ("uses(cake, flour)", compound("uses", atom("cake"), atom("flour")), True),
    ]
    return "cooking", kb, checks


DOMAINS = {
    "biology": domain_biology,
    "geography": domain_geography,
    "university": domain_university,
    "cooking": domain_cooking,
}


def run_experiment(kb, name, client, correctness_checks):
    facts_before = len(kb.facts)
    rules_before = len(kb.rules)
    total_before = facts_before + rules_before

    print(f"\n{'='*70}")
    print(f"  {name}: {total_before} clauses ({facts_before} facts, {rules_before} rules)")
    print(f"{'='*70}")

    # Symbolic baseline
    kb_sym = kb.copy()
    dreamer_sym = KnowledgeBaseDreamer()
    t0 = time.perf_counter()
    session_sym = dreamer_sym.dream(kb_sym, verify=True)
    t_sym = (time.perf_counter() - t0) * 1000
    sym_total = len(kb_sym)

    sym_ops = {}
    for op in session_sym.operations:
        sym_ops[op.operation] = sym_ops.get(op.operation, 0) + 1

    print(f"  Symbolic: {total_before} -> {sym_total} ({t_sym:.0f}ms)", end="")
    if sym_ops:
        print(f" [{', '.join(f'{k}:{v}' for k,v in sym_ops.items())}]")
    else:
        print(" [no ops]")

    # LLM-assisted
    dreamer_llm = KnowledgeBaseDreamer(llm_client=client)
    t0 = time.perf_counter()
    session_llm = dreamer_llm.dream(kb, verify=True)
    t_llm = (time.perf_counter() - t0) * 1000
    llm_total = len(kb)

    llm_ops = {}
    for op in session_llm.operations:
        llm_ops[op.operation] = llm_ops.get(op.operation, 0) + 1

    print(f"  LLM:      {total_before} -> {llm_total} ({t_llm:.0f}ms)", end="")
    if llm_ops:
        print(f" [{', '.join(f'{k}:{v}' for k,v in llm_ops.items())}]")
    else:
        print(" [no ops]")

    # Show LLM-proposed rules
    llm_rules = [op for op in session_llm.operations if op.operation == "llm_compression"]
    if llm_rules:
        print(f"  LLM discovered:")
        for op in llm_rules:
            for clause in op.new_clauses:
                print(f"    {clause}")

    # Correctness
    ev = PrologEvaluator(kb)
    passed = 0
    failed = []
    for label, query, expected in correctness_checks:
        if ev.has_solution(query) == expected:
            passed += 1
        else:
            failed.append(label)

    status = "ALL PASS" if not failed else f"FAILED: {', '.join(failed)}"
    delta = sym_total - llm_total
    improvement = f"+{delta} better" if delta > 0 else "same" if delta == 0 else f"{-delta} worse"
    print(f"  Correctness: {passed}/{len(correctness_checks)} ({status})")
    print(f"  LLM vs symbolic: {improvement}")

    return {
        "domain": name,
        "before": total_before,
        "symbolic": sym_total,
        "llm": llm_total,
        "improvement": delta,
        "correct": passed == len(correctness_checks),
        "llm_rules_found": len(llm_rules),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-domain LLM sleep cycle experiments")
    parser.add_argument("--provider", default="anthropic",
                        help="LLM provider (anthropic, openai, ollama)")
    parser.add_argument("--model", default=None, help="Model (default: per-provider)")
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY",
                        help="Env var containing API key")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (for ollama/custom)")
    parser.add_argument("--api-key", default=None, help="API key (direct)")
    parser.add_argument("--domain", "-d", help="Run single domain")
    args = parser.parse_args()

    client = LLMClient(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=0.3,
        max_tokens=800,
    )
    print(f"Provider: {client.provider}, Model: {client.model}")

    try:
        test = client.complete("Reply with just 'ok'.")
        print(f"Connection: {test.strip()[:20]}")
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    domains = {args.domain: DOMAINS[args.domain]} if args.domain else DOMAINS
    results = []
    for name, fn in domains.items():
        kb = fn()[1]  # (name, kb, checks)
        name, kb, checks = fn()
        results.append(run_experiment(kb, name, client, checks))

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Domain':<15} {'Before':>7} {'Symbolic':>9} {'LLM':>7} {'Delta':>7} {'Rules':>6} {'OK':>4}")
    print(f"  {'-'*15} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*6} {'-'*4}")
    for r in results:
        sign = f"+{r['improvement']}" if r['improvement'] > 0 else str(r['improvement'])
        print(f"  {r['domain']:<15} {r['before']:>7} {r['symbolic']:>9} {r['llm']:>7} {sign:>7} {r['llm_rules_found']:>6} {'Y' if r['correct'] else 'N':>4}")

    total_before = sum(r['before'] for r in results)
    total_sym = sum(r['symbolic'] for r in results)
    total_llm = sum(r['llm'] for r in results)
    total_delta = sum(r['improvement'] for r in results)
    total_rules = sum(r['llm_rules_found'] for r in results)
    all_correct = all(r['correct'] for r in results)
    print(f"  {'-'*15} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*6} {'-'*4}")
    print(f"  {'TOTAL':<15} {total_before:>7} {total_sym:>9} {total_llm:>7} {'+'+str(total_delta) if total_delta>0 else str(total_delta):>7} {total_rules:>6} {'Y' if all_correct else 'N':>4}")
    print(f"  Symbolic ratio: {total_sym/total_before:.2f}, LLM ratio: {total_llm/total_before:.2f}")
    print(f"  LLM usage: {client.usage}")


if __name__ == "__main__":
    main()
