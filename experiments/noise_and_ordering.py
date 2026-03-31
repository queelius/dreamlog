#!/usr/bin/env python3
"""
EX11: Noise tolerance - does the sleep cycle amplify or suppress errors?
EX13: Operation ordering - does the order A-H matter?

Usage:
    python experiments/noise_and_ordering.py
    python experiments/noise_and_ordering.py --experiment noise
    python experiments/noise_and_ordering.py --experiment ordering
"""

import sys
import time
import random
import argparse
from itertools import permutations
from collections import defaultdict

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


# ============================================================
# EX11: Noise Tolerance
# ============================================================

def build_clean_kb():
    """A clean, well-structured family KB."""
    kb = KnowledgeBase()
    people = ["alice", "bob", "carol", "dave", "eve", "frank",
              "grace", "henry", "iris", "jack"]
    for n in people:
        kb.add_fact(compound("person", atom(n)))
    for n in ["bob", "dave", "frank", "henry", "jack"]:
        kb.add_fact(compound("male", atom(n)))
    for n in ["alice", "carol", "eve", "grace", "iris"]:
        kb.add_fact(compound("female", atom(n)))

    parents = [("alice","bob"),("alice","carol"),("bob","dave"),("bob","eve"),
               ("carol","frank"),("carol","grace"),("dave","henry"),("eve","iris")]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))

    # Likes: most like chocolate
    for n in people[:8]:
        kb.add_fact(compound("likes", atom(n), atom("chocolate")))

    return kb, people


def inject_noise(kb, people, noise_pct, seed=42):
    """Inject random incorrect facts into the KB."""
    rng = random.Random(seed)
    n_facts = len(kb.facts)
    n_noise = max(1, int(n_facts * noise_pct / 100))

    functors = ["parent", "male", "female", "likes"]
    injected = []

    for _ in range(n_noise):
        f = rng.choice(functors)
        if f in ("male", "female"):
            # Wrong gender
            p = rng.choice(people)
            term = compound(f, atom(p))
        elif f == "parent":
            # Impossible parent (child is parent of grandparent)
            a, b = rng.sample(people, 2)
            term = compound("parent", atom(a), atom(b))
        elif f == "likes":
            # Random person likes random thing
            p = rng.choice(people)
            thing = rng.choice(["spinach", "broccoli", "tofu"])
            term = compound("likes", atom(p), atom(thing))

        try:
            kb.add_fact(term)
            injected.append(term)
        except ValueError:
            pass  # Duplicate or invalid

    return injected


def run_noise_experiment():
    print(f"\n{'='*70}")
    print(f"  EX11: NOISE TOLERANCE")
    print(f"  Does the sleep cycle amplify or suppress errors?")
    print(f"{'='*70}")

    noise_levels = [0, 5, 10, 20, 30, 50]

    print(f"\n  {'Noise%':>6} {'Initial':>8} {'After':>7} {'Ratio':>7} {'Ops':>5} "
          f"{'Noisy gen?':>10} {'Correct':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*5} {'-'*10} {'-'*8}")

    for pct in noise_levels:
        kb, people = build_clean_kb()
        initial_clean = len(kb)

        if pct > 0:
            injected = inject_noise(kb, people, pct)
        else:
            injected = []

        initial = len(kb)

        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)

        after = len(kb)
        ratio = after / initial if initial > 0 else 1.0
        ops = len(session.operations)

        # Check: did any noisy fact get generalized into a rule?
        # (This would be BAD - amplifying noise)
        noisy_generalized = False
        for op in session.operations:
            if op.operation == "generalization":
                for clause in op.original_clauses:
                    if isinstance(clause, Fact) and clause.term in injected:
                        noisy_generalized = True

        # Correctness: check known-true and known-false queries
        ev = PrologEvaluator(kb)
        correct = 0
        total = 0
        # Known true
        for p, c in [("alice","bob"),("bob","dave"),("carol","frank")]:
            total += 1
            if ev.has_solution(compound("parent", atom(p), atom(c))):
                correct += 1
        # Known false
        for p, c in [("dave","alice"),("henry","bob")]:
            total += 1
            if not ev.has_solution(compound("parent", atom(p), atom(c))):
                correct += 1

        gen_str = "YES (bad!)" if noisy_generalized else "no"
        correct_str = f"{correct}/{total}"
        print(f"  {pct:>5}% {initial:>8} {after:>7} {ratio:>7.2f} {ops:>5} "
              f"{gen_str:>10} {correct_str:>8}")

    print(f"\n  'Noisy gen?' = did noise get incorporated into a generalized rule")
    print(f"  A 'YES' means the system amplified errors (bad)")
    print(f"  A 'no' means noise was left as isolated facts (good)")


# ============================================================
# EX13: Operation Ordering
# ============================================================

def run_ordering_experiment():
    print(f"\n{'='*70}")
    print(f"  EX13: OPERATION ORDERING SENSITIVITY")
    print(f"  Does the order of operations matter?")
    print(f"{'='*70}")

    # Build a KB that exercises C, D, and E
    from experiments.ablation_and_scale import generate_scalable_kb
    kb_template, _ = generate_scalable_kb(
        n_entities=20, n_relations=3, n_transitive=4, n_body_pattern_groups=2)

    # The variable operations are C, D, E (the creative ones).
    # A, B are safe cleanup that should go first.
    # F, H are post-processing that should go last.
    # Question: does C-D-E vs D-C-E vs E-C-D matter?

    orderings = [
        ("C->D->E (default)", ["C", "D", "E"]),
        ("D->C->E", ["D", "C", "E"]),
        ("E->C->D", ["E", "C", "D"]),
        ("D->E->C", ["D", "E", "C"]),
        ("E->D->C", ["E", "D", "C"]),
        ("C->E->D", ["C", "E", "D"]),
    ]

    print(f"\n  Initial KB: {len(kb_template)} clauses")
    print(f"\n  {'Ordering':<20} {'After':>7} {'Ratio':>7} {'Ops':>5}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*5}")

    results = {}
    for label, order in orderings:
        kb = kb_template.copy()
        dreamer = KnowledgeBaseDreamer()

        # Monkey-patch dream() to run operations in specified order
        # We override the dream method to control operation order
        ops_all = []

        # Always run A and B first (cleanup)
        ops_all.extend(dreamer._eliminate_subsumed(kb))
        ops_all.extend(dreamer._prune_redundant_facts(kb))

        # Run C, D, E in specified order
        suite = None  # skip verification for ordering test
        for op_code in order:
            if op_code == "C":
                ops_all.extend(dreamer._generalize_facts(kb, suite))
            elif op_code == "D":
                ops_all.extend(dreamer._invent_predicates(kb, suite))
            elif op_code == "E":
                ops_all.extend(dreamer._extract_body_patterns(kb, suite))

        after = len(kb)
        ratio = after / len(kb_template) if len(kb_template) > 0 else 1.0
        ops_count = len(ops_all)
        results[label] = after

        print(f"  {label:<20} {after:>7} {ratio:>7.2f} {ops_count:>5}")

    # Analysis
    values = list(results.values())
    if len(set(values)) == 1:
        print(f"\n  Result: Order DOES NOT MATTER (all orderings produce {values[0]} clauses)")
    else:
        best = min(results.items(), key=lambda x: x[1])
        worst = max(results.items(), key=lambda x: x[1])
        print(f"\n  Result: Order MATTERS")
        print(f"    Best:  {best[0]} -> {best[1]} clauses")
        print(f"    Worst: {worst[0]} -> {worst[1]} clauses")
        print(f"    Difference: {worst[1] - best[1]} clauses")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e",
                        choices=["noise", "ordering", "all"],
                        default="all")
    args = parser.parse_args()

    if args.experiment in ("noise", "all"):
        run_noise_experiment()
    if args.experiment in ("ordering", "all"):
        run_ordering_experiment()


if __name__ == "__main__":
    main()
