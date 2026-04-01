#!/usr/bin/env python3
"""
EX20: Multi-cycle LLM-symbolic cascade.

Tests whether LLM discoveries in cycle 1 create patterns that symbolic
operations discover in cycle 2. The hypothesis: iterative dreaming with
LLM finds deeper abstractions than a single cycle.

Usage:
    python experiments/ex20_multi_cycle_llm.py
"""

import sys
import time
import json

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient


def build_cascade_kb():
    """KB designed so LLM cycle 1 discoveries enable symbolic cycle 2 finds.

    Structure:
    - Base: person, male, female, parent facts
    - Derivable via LLM (cross-functor): father, mother rules
    - After father/mother rules + pruning, the remaining KB has 3 structurally
      identical rule pairs (father, mother, plus uncle/aunt if LLM finds them)
      → Op D should discover a parameterized abstraction.
    """
    kb = KnowledgeBase()

    people = {
        "john": "male", "mary": "female", "bob": "male", "alice": "female",
        "carol": "female", "dave": "male", "eve": "female", "frank": "male",
        "grace": "female", "henry": "male",
    }
    for name, gender in people.items():
        kb.add_fact(compound("person", atom(name)))
        kb.add_fact(compound(gender, atom(name)))

    parents = [
        ("john", "bob"), ("john", "alice"), ("mary", "bob"), ("mary", "alice"),
        ("bob", "carol"), ("bob", "dave"), ("alice", "eve"), ("alice", "frank"),
        ("carol", "grace"), ("dave", "henry"),
    ]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))

    # Father/mother facts (derivable from parent + gender)
    for p, c in parents:
        if people[p] == "male":
            kb.add_fact(compound("father", atom(p), atom(c)))
        else:
            kb.add_fact(compound("mother", atom(p), atom(c)))

    # Sibling facts (derivable from shared parents)
    sibling_pairs = [("bob", "alice"), ("carol", "dave"), ("eve", "frank")]
    for a, b in sibling_pairs:
        kb.add_fact(compound("sibling", atom(a), atom(b)))
        kb.add_fact(compound("sibling", atom(b), atom(a)))

    # Uncle/aunt facts (derivable from parent's sibling + gender)
    # uncle(X, Y) :- sibling(X, Z), parent(Z, Y), male(X)
    # aunt(X, Y) :- sibling(X, Z), parent(Z, Y), female(X)
    uncles_aunts = [
        # bob's sibling is alice; alice's children are eve, frank
        ("bob", "eve", "male"), ("bob", "frank", "male"),
        # alice's sibling is bob; bob's children are carol, dave
        ("alice", "carol", "female"), ("alice", "dave", "female"),
    ]
    for person, niece_nephew, gender in uncles_aunts:
        pred = "uncle" if gender == "male" else "aunt"
        kb.add_fact(compound(pred, atom(person), atom(niece_nephew)))

    # Grandparent facts (derivable from parent chain)
    gp_pairs = [
        ("john", "carol"), ("john", "dave"), ("john", "eve"), ("john", "frank"),
        ("mary", "carol"), ("mary", "dave"), ("mary", "eve"), ("mary", "frank"),
    ]
    for gp, gc in gp_pairs:
        kb.add_fact(compound("grandparent", atom(gp), atom(gc)))

    return kb, people


def run_multi_cycle(kb, client, max_cycles=5):
    """Run iterative dream cycles, tracking what each cycle discovers."""
    dreamer = KnowledgeBaseDreamer(llm_client=client)
    cycle_results = []

    for cycle in range(1, max_cycles + 1):
        size_before = len(kb)
        t0 = time.perf_counter()
        session = dreamer.dream(kb, verify=True)
        elapsed = time.perf_counter() - t0
        size_after = len(kb)

        ops_summary = {}
        for op in session.operations:
            ops_summary[op.operation] = ops_summary.get(op.operation, 0) + 1

        new_rules = []
        for op in session.operations:
            for c in op.new_clauses:
                if isinstance(c, Rule):
                    new_rules.append(str(c))

        result = {
            "cycle": cycle,
            "before": size_before,
            "after": size_after,
            "removed": size_before - size_after,
            "time": elapsed,
            "ops": ops_summary,
            "new_rules": new_rules,
            "compressed": session.compressed,
        }
        cycle_results.append(result)

        print(f"\n  Cycle {cycle}: {size_before} -> {size_after} "
              f"(-{size_before - size_after}) in {elapsed:.1f}s", flush=True)
        if ops_summary:
            print(f"    Ops: {ops_summary}", flush=True)
        for r in new_rules:
            print(f"    + {r}", flush=True)

        if not session.compressed:
            print(f"    Converged (no more compression possible)", flush=True)
            break

    return cycle_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-cycle LLM experiment")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-cycles", type=int, default=5)
    args = parser.parse_args()

    client = LLMClient(
        provider=args.provider, model=args.model, api_key=args.api_key,
        api_key_env=args.api_key_env, base_url=args.base_url,
        temperature=0.3, max_tokens=800)

    print(f"Provider: {client.provider}, Model: {client.model}")

    # ── Experiment 1: Cascade KB ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EX20a: Multi-cycle on cascade KB (designed for cascade)")
    print(f"{'='*70}")

    kb_cascade, people = build_cascade_kb()
    print(f"  Initial: {len(kb_cascade)} clauses "
          f"({len(kb_cascade.facts)} facts, {len(kb_cascade.rules)} rules)")

    # Correctness checks
    checks_cascade = [
        ("father(john,bob)", compound("father", atom("john"), atom("bob")), True),
        ("mother(mary,bob)", compound("mother", atom("mary"), atom("bob")), True),
        ("grandparent(john,carol)", compound("grandparent", atom("john"), atom("carol")), True),
        ("uncle(bob,eve)", compound("uncle", atom("bob"), atom("eve")), True),
        ("aunt(alice,carol)", compound("aunt", atom("alice"), atom("carol")), True),
        ("father(mary,bob)", compound("father", atom("mary"), atom("bob")), False),
        ("uncle(alice,eve)", compound("uncle", atom("alice"), atom("eve")), False),
    ]

    results_cascade = run_multi_cycle(kb_cascade, client, args.max_cycles)

    # Verify correctness
    ev = PrologEvaluator(kb_cascade)
    print(f"\n  Correctness checks:")
    all_pass = True
    for label, query, expected in checks_cascade:
        result = ev.has_solution(query)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    {status}: {label}")
    print(f"  All pass: {all_pass}")

    # ── Experiment 2: Symbolic-only comparison ────────────────────
    print(f"\n{'='*70}")
    print(f"  EX20b: Symbolic-only comparison (no LLM)")
    print(f"{'='*70}")

    kb_sym, _ = build_cascade_kb()
    dreamer_sym = KnowledgeBaseDreamer()  # no LLM
    results_sym = []
    for cycle in range(1, args.max_cycles + 1):
        size_before = len(kb_sym)
        session = dreamer_sym.dream(kb_sym, verify=True)
        size_after = len(kb_sym)
        print(f"  Cycle {cycle}: {size_before} -> {size_after} "
              f"(-{size_before - size_after})", flush=True)
        results_sym.append({"before": size_before, "after": size_after})
        if not session.compressed:
            print(f"    Converged", flush=True)
            break

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    initial = results_cascade[0]["before"]
    final_llm = results_cascade[-1]["after"]
    final_sym = results_sym[-1]["after"]

    print(f"  Initial KB:     {initial} clauses")
    print(f"  Symbolic-only:  {final_sym} clauses ({len(results_sym)} cycles)")
    print(f"  LLM+symbolic:   {final_llm} clauses ({len(results_cascade)} cycles)")
    print(f"  LLM advantage:  {final_sym - final_llm} additional clauses removed")
    print(f"  Correctness:    {'ALL PASS' if all_pass else 'FAIL'}")
    print(f"  LLM usage:      {client.usage}")

    total_rules = sum(len(r["new_rules"]) for r in results_cascade)
    print(f"  Total rules discovered: {total_rules}")

    # Was there a cascade? (cycle 2+ found something that cycle 1 didn't)
    if len(results_cascade) > 1 and results_cascade[1]["compressed"]:
        print(f"  CASCADE DETECTED: cycle 2 compressed further!")
        print(f"    Cycle 1: -{results_cascade[0]['removed']}")
        print(f"    Cycle 2: -{results_cascade[1]['removed']}")
    else:
        print(f"  No cascade (converged in 1 cycle or cycle 2 found nothing new)")


if __name__ == "__main__":
    main()
