#!/usr/bin/env python3
"""
EX12: Compression on Real-World-Style Knowledge

Port a subset of real-world knowledge into DreamLog and see what
the sleep cycle discovers. Uses:
- A family tree (biblical genealogy style)
- A small geographic ontology
- A course prerequisite graph

Usage:
    python experiments/real_world_kb.py
"""

import sys
import time

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


def build_genealogy_kb():
    """Biblical-style genealogy: 4 generations, 30 people."""
    kb = KnowledgeBase()

    # Generation 1
    gen1 = [("adam", "m"), ("eve", "f")]
    # Generation 2
    gen2 = [("cain", "m"), ("abel", "m"), ("seth", "m")]
    # Generation 3
    gen3 = [("enoch", "m"), ("enos", "m"), ("noam", "f"),
            ("irad", "m"), ("adah", "f")]
    # Generation 4
    gen4 = [("mehujael", "m"), ("methuselah", "m"), ("lamech", "m"),
            ("zillah", "f"), ("naamah", "f")]

    all_people = gen1 + gen2 + gen3 + gen4
    for name, gender in all_people:
        kb.add_fact(compound("person", atom(name)))
        kb.add_fact(compound("male" if gender == "m" else "female", atom(name)))

    # Parent relationships
    parents = [
        # Gen1 -> Gen2
        ("adam", "cain"), ("adam", "abel"), ("adam", "seth"),
        ("eve", "cain"), ("eve", "abel"), ("eve", "seth"),
        # Gen2 -> Gen3
        ("cain", "enoch"), ("seth", "enos"), ("seth", "noam"),
        ("cain", "irad"), ("cain", "adah"),
        # Gen3 -> Gen4
        ("enoch", "mehujael"), ("enos", "methuselah"),
        ("irad", "lamech"), ("irad", "zillah"),
        ("noam", "naamah"),
    ]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))

    # Father/mother (derivable from parent + gender)
    for p, c in parents:
        gender = dict(all_people).get(p, "m")
        pred = "father" if gender == "m" else "mother"
        kb.add_fact(compound(pred, atom(p), atom(c)))

    # Sibling pairs (derivable from shared parent)
    siblings = [("cain","abel"),("cain","seth"),("abel","seth"),
                ("enoch","irad"),("enoch","adah"),("irad","adah"),
                ("enos","noam")]
    for a, b in siblings:
        kb.add_fact(compound("sibling", atom(a), atom(b)))
        kb.add_fact(compound("sibling", atom(b), atom(a)))

    # Ancestor rules
    kb.add_rule(Rule(compound("ancestor", var("X"), var("Y")),
                     [compound("parent", var("X"), var("Y"))]))
    kb.add_rule(Rule(compound("ancestor", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("ancestor", var("Y"), var("Z"))]))

    # Grandparent
    kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z"))]))

    checks = [
        ("ancestor(adam, methuselah)", compound("ancestor", atom("adam"), atom("methuselah")), True),
        ("grandparent(adam, enoch)", compound("grandparent", atom("adam"), atom("enoch")), True),
        ("parent(adam, cain)", compound("parent", atom("adam"), atom("cain")), True),
        ("father(adam, cain)", compound("father", atom("adam"), atom("cain")), True),
        ("ancestor(lamech, adam)", compound("ancestor", atom("lamech"), atom("adam")), False),
    ]
    return "genealogy", kb, checks


def build_geography_kb():
    """Geographic knowledge: countries, capitals, continents, languages."""
    kb = KnowledgeBase()

    data = {
        "usa": {"capital": "washington", "continent": "north_america",
                "language": "english", "hemisphere": "north", "gdp": "high"},
        "canada": {"capital": "ottawa", "continent": "north_america",
                   "language": "english", "hemisphere": "north", "gdp": "high"},
        "mexico": {"capital": "mexico_city", "continent": "north_america",
                   "language": "spanish", "hemisphere": "north", "gdp": "medium"},
        "brazil": {"capital": "brasilia", "continent": "south_america",
                   "language": "portuguese", "hemisphere": "south", "gdp": "medium"},
        "argentina": {"capital": "buenos_aires", "continent": "south_america",
                      "language": "spanish", "hemisphere": "south", "gdp": "medium"},
        "uk": {"capital": "london", "continent": "europe",
               "language": "english", "hemisphere": "north", "gdp": "high"},
        "france": {"capital": "paris", "continent": "europe",
                   "language": "french", "hemisphere": "north", "gdp": "high"},
        "germany": {"capital": "berlin", "continent": "europe",
                    "language": "german", "hemisphere": "north", "gdp": "high"},
        "spain": {"capital": "madrid", "continent": "europe",
                  "language": "spanish", "hemisphere": "north", "gdp": "high"},
        "japan": {"capital": "tokyo", "continent": "asia",
                  "language": "japanese", "hemisphere": "north", "gdp": "high"},
        "china": {"capital": "beijing", "continent": "asia",
                  "language": "mandarin", "hemisphere": "north", "gdp": "high"},
        "india": {"capital": "new_delhi", "continent": "asia",
                  "language": "hindi", "hemisphere": "north", "gdp": "medium"},
        "australia": {"capital": "canberra", "continent": "oceania",
                      "language": "english", "hemisphere": "south", "gdp": "high"},
        "egypt": {"capital": "cairo", "continent": "africa",
                  "language": "arabic", "hemisphere": "north", "gdp": "medium"},
        "south_africa": {"capital": "pretoria", "continent": "africa",
                         "language": "english", "hemisphere": "south", "gdp": "medium"},
    }

    for country, props in data.items():
        kb.add_fact(compound("country", atom(country)))
        for prop, val in props.items():
            kb.add_fact(compound(prop, atom(country), atom(val)))

    # Derivable properties
    for country, props in data.items():
        if props["language"] == "english":
            kb.add_fact(compound("english_speaking", atom(country)))
        if props["gdp"] == "high":
            kb.add_fact(compound("developed", atom(country)))
        if props["hemisphere"] == "north":
            kb.add_fact(compound("northern", atom(country)))

    # Border relationships
    borders = [("usa","canada"),("usa","mexico"),("france","germany"),
               ("france","spain"),("brazil","argentina"),("china","india"),
               ("egypt","south_africa")]  # not real but for testing
    for a, b in borders:
        kb.add_fact(compound("borders", atom(a), atom(b)))
        kb.add_fact(compound("borders", atom(b), atom(a)))

    checks = [
        ("english_speaking(usa)", compound("english_speaking", atom("usa")), True),
        ("english_speaking(france)", compound("english_speaking", atom("france")), False),
        ("developed(japan)", compound("developed", atom("japan")), True),
        ("developed(brazil)", compound("developed", atom("brazil")), False),
        ("northern(egypt)", compound("northern", atom("egypt")), True),
        ("northern(australia)", compound("northern", atom("australia")), False),
    ]
    return "geography", kb, checks


def build_curriculum_kb():
    """University curriculum: courses, prerequisites, departments."""
    kb = KnowledgeBase()

    courses = {
        "calc1": {"dept": "math", "level": "intro", "credits": "4"},
        "calc2": {"dept": "math", "level": "intermediate", "credits": "4"},
        "linear_algebra": {"dept": "math", "level": "intermediate", "credits": "3"},
        "diff_eq": {"dept": "math", "level": "advanced", "credits": "3"},
        "physics1": {"dept": "physics", "level": "intro", "credits": "4"},
        "physics2": {"dept": "physics", "level": "intermediate", "credits": "4"},
        "quantum": {"dept": "physics", "level": "advanced", "credits": "3"},
        "cs101": {"dept": "cs", "level": "intro", "credits": "3"},
        "cs201": {"dept": "cs", "level": "intermediate", "credits": "3"},
        "algorithms": {"dept": "cs", "level": "advanced", "credits": "3"},
        "ml": {"dept": "cs", "level": "advanced", "credits": "3"},
        "chem1": {"dept": "chemistry", "level": "intro", "credits": "4"},
        "chem2": {"dept": "chemistry", "level": "intermediate", "credits": "4"},
        "organic": {"dept": "chemistry", "level": "advanced", "credits": "3"},
    }

    for course, props in courses.items():
        kb.add_fact(compound("course", atom(course)))
        for prop, val in props.items():
            kb.add_fact(compound(prop, atom(course), atom(val)))

    # Prerequisites
    prereqs = [
        ("calc2", "calc1"), ("linear_algebra", "calc1"),
        ("diff_eq", "calc2"), ("diff_eq", "linear_algebra"),
        ("physics1", "calc1"), ("physics2", "physics1"),
        ("physics2", "calc2"), ("quantum", "physics2"),
        ("quantum", "linear_algebra"),
        ("cs201", "cs101"), ("algorithms", "cs201"),
        ("algorithms", "calc2"), ("ml", "algorithms"),
        ("ml", "linear_algebra"), ("chem2", "chem1"),
        ("organic", "chem2"),
    ]
    for course, prereq in prereqs:
        kb.add_fact(compound("prereq", atom(course), atom(prereq)))

    # 3 transitive closures (Op D target)
    for head, base in [("requires", "prereq"), ("depends_on", "prereq"),
                       ("needs_first", "prereq")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    # Derivable: stem_course
    for course, props in courses.items():
        if props["dept"] in ("math", "physics", "cs", "chemistry"):
            kb.add_fact(compound("stem_course", atom(course)))

    # Derivable: advanced_course
    for course, props in courses.items():
        if props["level"] == "advanced":
            kb.add_fact(compound("advanced_course", atom(course)))

    checks = [
        ("requires(quantum, calc1)",
         compound("requires", atom("quantum"), atom("calc1")), True),
        ("requires(ml, cs101)",
         compound("requires", atom("ml"), atom("cs101")), True),
        ("stem_course(calc1)",
         compound("stem_course", atom("calc1")), True),
        ("advanced_course(ml)",
         compound("advanced_course", atom("ml")), True),
        ("requires(calc1, quantum)",
         compound("requires", atom("calc1"), atom("quantum")), False),
    ]
    return "curriculum", kb, checks


def run_experiment(name, kb, checks, llm_client=None):
    initial_facts = len(kb.facts)
    initial_rules = len(kb.rules)
    initial_total = initial_facts + initial_rules

    dreamer = KnowledgeBaseDreamer(llm_client=llm_client)
    t0 = time.perf_counter()
    session = dreamer.dream(kb, verify=True)
    elapsed = (time.perf_counter() - t0) * 1000

    after_total = len(kb)
    ops = {}
    for op in session.operations:
        ops[op.operation] = ops.get(op.operation, 0) + 1

    ev = PrologEvaluator(kb)
    correct = sum(1 for _, q, exp in checks if ev.has_solution(q) == exp)

    print(f"\n  {name}:")
    print(f"    {initial_total} -> {after_total} clauses ({elapsed:.0f}ms)")
    print(f"    Ops: {dict(ops) if ops else 'none'}")
    print(f"    Correctness: {correct}/{len(checks)}")

    # Show what was discovered
    for op in session.operations:
        if op.operation in ("invention", "extraction"):
            for clause in op.new_clauses[:3]:
                if isinstance(clause, Rule):
                    print(f"    Discovered: {clause}")

    return {
        "name": name,
        "before": initial_total,
        "after": after_total,
        "ratio": after_total / initial_total if initial_total else 1.0,
        "ops": ops,
        "correct": correct == len(checks),
    }


def main():
    print(f"{'='*70}")
    print(f"  EX12: REAL-WORLD-STYLE KNOWLEDGE COMPRESSION")
    print(f"{'='*70}")

    domains = [
        build_genealogy_kb(),
        build_geography_kb(),
        build_curriculum_kb(),
    ]

    results = []
    for name, kb, checks in domains:
        results.append(run_experiment(name, kb, checks))

    # Summary
    print(f"\n  {'Domain':<15} {'Before':>7} {'After':>7} {'Ratio':>7} {'OK':>4}")
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*4}")
    for r in results:
        print(f"  {r['name']:<15} {r['before']:>7} {r['after']:>7} {r['ratio']:>7.2f} "
              f"{'Y' if r['correct'] else 'N':>4}")

    total_before = sum(r["before"] for r in results)
    total_after = sum(r["after"] for r in results)
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*4}")
    print(f"  {'TOTAL':<15} {total_before:>7} {total_after:>7} "
          f"{total_after/total_before:.2f}")


if __name__ == "__main__":
    main()
