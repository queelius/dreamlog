#!/usr/bin/env python3
"""
Experiment #1: Iterative Dream Cycles + #4: Semantic Drift

Questions:
1. How many dream cycles until convergence (no more compressions)?
2. Does the KB behavior change on held-out queries across cycles?
3. Does iterated dreaming discover things single-pass misses?

Method:
- Build a KB, generate a held-out query set with expected answers
- Run dream() repeatedly, measuring after each cycle:
  - Clause count and body goal count
  - Which operations fired
  - Semantic drift: how many held-out queries changed answer

Usage:
    python experiments/iterative_dreaming.py
    python experiments/iterative_dreaming.py --max-cycles 20
    python experiments/iterative_dreaming.py --with-llm --model phi4-mini:latest
"""

import sys
import time
import argparse
import json
from collections import defaultdict

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


def generate_held_out_queries(kb, sample_size=200):
    """Generate ground queries with expected True/False answers from current KB.

    These are the 'behavioral fingerprint' of the KB. Any change in answers
    across dream cycles = semantic drift.
    """
    ev = PrologEvaluator(kb)

    # Collect atom pool
    atoms = set()
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            for arg in fact.term.args:
                if isinstance(arg, Atom):
                    atoms.add(arg.value)
    atoms = sorted(atoms)

    # Collect functors with their arities
    functors = {}
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            functors[(fact.term.functor, fact.term.arity)] = True
    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            functors[(rule.head.functor, rule.head.arity)] = True

    # Generate queries
    queries = []
    for (functor, arity) in functors:
        if functor.startswith("_") or functor.startswith("exception_"):
            continue
        if arity == 1:
            for a in atoms[:15]:
                term = Compound(functor, [Atom(a)])
                expected = ev.has_solution(term)
                queries.append((term, expected))
        elif arity == 2:
            for a in atoms[:10]:
                for b in atoms[:10]:
                    if a != b:
                        term = Compound(functor, [Atom(a), Atom(b)])
                        expected = ev.has_solution(term)
                        queries.append((term, expected))
        if len(queries) >= sample_size:
            break

    return queries[:sample_size]


def measure_drift(kb, held_out_queries):
    """Count how many held-out queries changed answer."""
    ev = PrologEvaluator(kb)
    changed = 0
    false_negatives = 0  # was True, now False
    false_positives = 0  # was False, now True
    for term, original_answer in held_out_queries:
        current = ev.has_solution(term)
        if current != original_answer:
            changed += 1
            if original_answer and not current:
                false_negatives += 1
            else:
                false_positives += 1
    return {
        "total_queries": len(held_out_queries),
        "changed": changed,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "drift_rate": changed / len(held_out_queries) if held_out_queries else 0,
    }


def build_rich_kb():
    """A KB with enough structure for interesting multi-cycle behavior."""
    kb = KnowledgeBase()

    # People
    people = ["alice", "bob", "carol", "dave", "eve", "frank", "grace",
              "henry", "iris", "jack"]
    for name in people:
        kb.add_fact(compound("person", atom(name)))

    # Gender
    for name in ["alice", "carol", "eve", "grace", "iris"]:
        kb.add_fact(compound("female", atom(name)))
    for name in ["bob", "dave", "frank", "henry", "jack"]:
        kb.add_fact(compound("male", atom(name)))

    # Parent chain
    parents = [
        ("alice", "bob"), ("alice", "carol"),
        ("bob", "dave"), ("bob", "eve"),
        ("carol", "frank"), ("carol", "grace"),
        ("dave", "henry"), ("eve", "iris"),
        ("frank", "jack"),
    ]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))

    # Father/mother facts (derivable from parent + gender)
    for p, c in parents:
        if p in ["bob", "dave", "frank", "henry", "jack"]:
            kb.add_fact(compound("father", atom(p), atom(c)))
        else:
            kb.add_fact(compound("mother", atom(p), atom(c)))

    # Likes (most like chocolate, guard for Op C)
    for name in people[:8]:
        kb.add_fact(compound("likes", atom(name), atom("chocolate")))

    # Sibling facts (derivable: same parent)
    kb.add_fact(compound("sibling", atom("bob"), atom("carol")))
    kb.add_fact(compound("sibling", atom("carol"), atom("bob")))
    kb.add_fact(compound("sibling", atom("dave"), atom("eve")))
    kb.add_fact(compound("sibling", atom("eve"), atom("dave")))
    kb.add_fact(compound("sibling", atom("frank"), atom("grace")))
    kb.add_fact(compound("sibling", atom("grace"), atom("frank")))

    # 3 transitive closures with different base relations (Op D)
    edges = [("alice", "bob"), ("bob", "dave"), ("dave", "henry")]
    for a, b in edges:
        kb.add_fact(compound("follows", atom(a), atom(b)))
    links = [("alice", "carol"), ("carol", "frank")]
    for a, b in links:
        kb.add_fact(compound("manages", atom(a), atom(b)))

    for head, base in [("ancestor", "parent"), ("reachable", "follows"),
                       ("connected", "manages")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    # Rules with shared body prefix (Op E)
    kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z"))]))
    kb.add_rule(Rule(compound("great_grandparent", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("parent", var("Z"), var("W"))]))
    kb.add_rule(Rule(compound("uncle_or_aunt", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("sibling", var("Z"), var("W"))]))

    # Redundant facts (Op B targets)
    kb.add_fact(compound("ancestor", atom("alice"), atom("bob")))
    kb.add_fact(compound("ancestor", atom("alice"), atom("dave")))

    return kb


def run_iterative_experiment(max_cycles=10, llm_client=None):
    kb = build_rich_kb()

    initial_clauses = len(kb)
    initial_facts = len(kb.facts)
    initial_rules = len(kb.rules)
    initial_body_goals = sum(len(r.body) for r in kb.rules)

    print(f"Initial KB: {initial_clauses} clauses ({initial_facts} facts, "
          f"{initial_rules} rules, {initial_body_goals} body goals)")

    # Generate held-out queries BEFORE any dreaming
    held_out = generate_held_out_queries(kb)
    print(f"Held-out queries: {len(held_out)} "
          f"({sum(1 for _,a in held_out if a)} true, "
          f"{sum(1 for _,a in held_out if not a)} false)")

    # Simulate wake phase for Op F
    ev = PrologEvaluator(kb)
    for _ in range(5):
        for (functor, _), _ in list(set(
            ((f.term.functor, f.term.arity), True)
            for f in kb.facts if isinstance(f.term, Compound)
        ))[:10]:
            try:
                list(ev.query([kb.facts[0].term]))
            except Exception:
                pass

    # Iterative dreaming
    print(f"\n{'Cycle':>5} {'Clauses':>8} {'Facts':>6} {'Rules':>6} {'BodyG':>6} "
          f"{'Ops':>4} {'Drift':>6} {'FN':>4} {'FP':>4} {'Time':>7}")
    print(f"{'-'*5:>5} {'-'*8:>8} {'-'*6:>6} {'-'*6:>6} {'-'*6:>6} "
          f"{'-'*4:>4} {'-'*6:>6} {'-'*4:>4} {'-'*4:>4} {'-'*7:>7}")

    cycle_data = []
    for cycle in range(1, max_cycles + 1):
        dreamer = KnowledgeBaseDreamer(llm_client=llm_client)
        t0 = time.perf_counter()
        session = dreamer.dream(kb, verify=True)
        elapsed = (time.perf_counter() - t0) * 1000

        clauses = len(kb)
        facts = len(kb.facts)
        rules = len(kb.rules)
        body_goals = sum(len(r.body) for r in kb.rules)
        ops = len(session.operations)

        drift = measure_drift(kb, held_out)

        op_summary = defaultdict(int)
        for op in session.operations:
            op_summary[op.operation] += 1

        print(f"{cycle:>5} {clauses:>8} {facts:>6} {rules:>6} {body_goals:>6} "
              f"{ops:>4} {drift['drift_rate']:>5.1%} {drift['false_negatives']:>4} "
              f"{drift['false_positives']:>4} {elapsed:>6.0f}ms"
              + (f"  [{', '.join(f'{k}:{v}' for k,v in op_summary.items())}]" if ops else ""))

        cycle_data.append({
            "cycle": cycle,
            "clauses": clauses,
            "facts": facts,
            "rules": rules,
            "body_goals": body_goals,
            "operations": ops,
            "drift": drift,
            "op_summary": dict(op_summary),
            "compressed": session.compressed,
        })

        if not session.compressed:
            print(f"\n  Converged at cycle {cycle} (no more compressions)")
            break

    # Summary
    print(f"\n{'='*70}")
    print(f"  ITERATIVE DREAMING SUMMARY")
    print(f"{'='*70}")
    print(f"  Initial:     {initial_clauses} clauses, {initial_body_goals} body goals")
    print(f"  Final:       {len(kb)} clauses, {sum(len(r.body) for r in kb.rules)} body goals")
    print(f"  Cycles:      {len(cycle_data)}")
    print(f"  Converged:   {'Yes' if not cycle_data[-1]['compressed'] else 'No (hit max)'}")

    total_drift = cycle_data[-1]['drift']
    print(f"  Final drift: {total_drift['changed']}/{total_drift['total_queries']} "
          f"queries ({total_drift['drift_rate']:.1%})")
    if total_drift['false_negatives']:
        print(f"    False negatives (lost): {total_drift['false_negatives']}")
    if total_drift['false_positives']:
        print(f"    False positives (gained): {total_drift['false_positives']}")

    # Compression trajectory
    if len(cycle_data) > 1:
        print(f"\n  Compression trajectory:")
        for d in cycle_data:
            bar = "#" * max(0, initial_clauses - d['clauses'])
            print(f"    Cycle {d['cycle']}: {d['clauses']:>3} clauses {bar}")

    return cycle_data


def main():
    parser = argparse.ArgumentParser(description="Iterative Dream Cycles + Semantic Drift")
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--base-url", default="http://192.168.0.225:11434/v1")
    parser.add_argument("--model", default="phi4-mini:latest")
    args = parser.parse_args()

    llm_client = None
    if args.with_llm:
        from dreamlog.llm_client import LLMClient
        llm_client = LLMClient(
            base_url=args.base_url, model=args.model,
            temperature=0.3, max_tokens=800)
        print(f"LLM: {args.model} @ {args.base_url}")

    run_iterative_experiment(max_cycles=args.max_cycles, llm_client=llm_client)


if __name__ == "__main__":
    main()
