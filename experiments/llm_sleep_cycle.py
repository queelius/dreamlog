#!/usr/bin/env python3
"""
LLM-Assisted Sleep Cycle Experiment

Tests the sleep cycle with an actual LLM to see:
1. Can it name invented predicates meaningfully?
2. Can it discover cross-functor rules the symbolic methods miss?

Usage:
    python experiments/llm_sleep_cycle.py
    python experiments/llm_sleep_cycle.py --model phi4-mini:latest
    python experiments/llm_sleep_cycle.py --base-url http://localhost:11434/v1
"""

import sys
import time
import argparse

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient


def build_family_kb():
    """Family KB with cross-functor relationships for LLM to discover."""
    kb = KnowledgeBase()

    # People
    people = ["john", "mary", "bob", "alice", "carol", "dave", "eve", "frank"]
    for name in people:
        kb.add_fact(compound("person", atom(name)))

    # Gender
    for name in ["john", "bob", "dave", "frank"]:
        kb.add_fact(compound("male", atom(name)))
    for name in ["mary", "alice", "carol", "eve"]:
        kb.add_fact(compound("female", atom(name)))

    # Parent relationships
    parents = [
        ("john", "bob"), ("john", "alice"),
        ("mary", "bob"), ("mary", "alice"),
        ("bob", "carol"), ("bob", "dave"),
        ("alice", "eve"), ("alice", "frank"),
    ]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))

    # Father facts (derivable from parent + male)
    for p, c in parents:
        if p in ["john", "bob", "dave", "frank"]:
            kb.add_fact(compound("father", atom(p), atom(c)))

    # Mother facts (derivable from parent + female)
    for p, c in parents:
        if p in ["mary", "alice", "carol", "eve"]:
            kb.add_fact(compound("mother", atom(p), atom(c)))

    # Ancestor rules (3 transitive closures with different bases for Op D)
    for head, base in [("ancestor", "parent"), ("descendant_of", "parent")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    # Sibling facts
    kb.add_fact(compound("sibling", atom("bob"), atom("alice")))
    kb.add_fact(compound("sibling", atom("alice"), atom("bob")))
    kb.add_fact(compound("sibling", atom("carol"), atom("dave")))
    kb.add_fact(compound("sibling", atom("dave"), atom("carol")))
    kb.add_fact(compound("sibling", atom("eve"), atom("frank")))
    kb.add_fact(compound("sibling", atom("frank"), atom("eve")))

    return kb


def build_graph_kb():
    """Graph KB with patterns for the LLM to discover."""
    kb = KnowledgeBase()

    # Nodes with types
    for n in ["a", "b", "c", "d", "e", "f"]:
        kb.add_fact(compound("node", atom(n)))
    for n in ["a", "b", "c"]:
        kb.add_fact(compound("source_node", atom(n)))
    for n in ["d", "e", "f"]:
        kb.add_fact(compound("sink_node", atom(n)))

    # Edges
    edges = [("a","b"), ("b","c"), ("c","d"), ("a","d"), ("b","e"), ("c","f")]
    for a, b in edges:
        kb.add_fact(compound("edge", atom(a), atom(b)))

    # Weighted edges (same topology, different predicate)
    for a, b in edges:
        kb.add_fact(compound("weighted_edge", atom(a), atom(b), atom("1")))

    # Path facts (derivable from edges transitively)
    kb.add_fact(compound("path", atom("a"), atom("d")))
    kb.add_fact(compound("path", atom("b"), atom("e")))
    kb.add_fact(compound("path", atom("c"), atom("f")))

    # 3 transitive closures (Op D target)
    for head, base in [("reachable","edge"), ("connected","edge"), ("linked","edge")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    return kb


def run_experiment(kb, name, client, correctness_checks):
    print(f"\n{'='*70}")
    print(f"  Experiment: {name}")
    print(f"{'='*70}")

    facts_before = len(kb.facts)
    rules_before = len(kb.rules)
    total_before = facts_before + rules_before
    print(f"\n  Before: {total_before} clauses ({facts_before} facts, {rules_before} rules)")

    # Phase 1: Symbolic only
    kb_symbolic = kb.copy()
    dreamer_sym = KnowledgeBaseDreamer()
    t0 = time.perf_counter()
    session_sym = dreamer_sym.dream(kb_symbolic, verify=True)
    t_sym = (time.perf_counter() - t0) * 1000

    sym_total = len(kb_symbolic)
    print(f"\n  Symbolic only: {total_before} -> {sym_total} clauses ({t_sym:.0f}ms)")
    if session_sym.operations:
        for op_name, count in _op_summary(session_sym).items():
            print(f"    {op_name}: {count}x")

    # Phase 2: LLM-assisted
    dreamer_llm = KnowledgeBaseDreamer(llm_client=client)
    t0 = time.perf_counter()
    session_llm = dreamer_llm.dream(kb, verify=True)
    t_llm = (time.perf_counter() - t0) * 1000

    llm_total = len(kb)
    print(f"\n  LLM-assisted: {total_before} -> {llm_total} clauses ({t_llm:.0f}ms)")
    if session_llm.operations:
        for op_name, count in _op_summary(session_llm).items():
            print(f"    {op_name}: {count}x")

    # Show what the LLM did
    llm_ops = [op for op in session_llm.operations
               if op.operation in ("llm_compression",)]
    if llm_ops:
        print(f"\n  LLM-proposed rules:")
        for op in llm_ops:
            for clause in op.new_clauses:
                print(f"    {clause}")

    # Check for renamed predicates
    for r in kb.rules:
        if isinstance(r.head, Compound):
            f = r.head.functor
            if not f.startswith("_") and f not in _original_functors(session_llm):
                pass  # named predicate

    # Show invented predicate names
    invented = set()
    for r in kb.rules:
        if isinstance(r.head, Compound):
            f = r.head.functor
            if any(f.startswith(p) for p in ("_invented_", "_extracted_", "transitive", "closure")):
                invented.add(f)
            elif f not in _get_all_head_functors(kb_symbolic):
                invented.add(f)
    if invented:
        print(f"\n  Named/invented predicates: {', '.join(sorted(invented))}")

    # Correctness
    print(f"\n  Correctness checks:")
    ev = PrologEvaluator(kb)
    all_pass = True
    for label, query, expected in correctness_checks:
        result = ev.has_solution(query)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    {status}: {label}")

    print(f"\n  Summary: symbolic {session_sym.compression_ratio:.2f} vs LLM {session_llm.compression_ratio:.2f}")
    if llm_total < sym_total:
        print(f"  LLM improved compression by {sym_total - llm_total} clauses")
    elif llm_total == sym_total:
        print(f"  LLM matched symbolic compression")
    else:
        print(f"  LLM added {llm_total - sym_total} clauses (naming overhead)")

    return all_pass


def _op_summary(session):
    summary = {}
    for op in session.operations:
        summary[op.operation] = summary.get(op.operation, 0) + 1
    return summary


def _original_functors(session):
    functors = set()
    for op in session.operations:
        for clause in op.original_clauses:
            if isinstance(clause, Rule) and isinstance(clause.head, Compound):
                functors.add(clause.head.functor)
            elif isinstance(clause, Fact) and isinstance(clause.term, Compound):
                functors.add(clause.term.functor)
    return functors


def _get_all_head_functors(kb):
    functors = set()
    for r in kb.rules:
        if isinstance(r.head, Compound):
            functors.add(r.head.functor)
    return functors


def main():
    parser = argparse.ArgumentParser(description="LLM-Assisted Sleep Cycle Experiment")
    parser.add_argument("--base-url", default="http://localhost:11434/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default="phi4-mini:latest",
                        help="Model to use")
    parser.add_argument("--api-key", default="ollama")
    args = parser.parse_args()

    print(f"Connecting to {args.base_url} with model {args.model}")

    client = LLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=0.3,
        max_tokens=500,
    )

    # Quick connectivity test
    try:
        test = client.complete("Reply with just the word 'ok'.")
        print(f"LLM test: {test.strip()[:50]}")
    except Exception as e:
        print(f"LLM connection failed: {e}")
        sys.exit(1)

    # Experiment 1: Family KB
    family_kb = build_family_kb()
    family_checks = [
        ("ancestor(john, carol)", compound("ancestor", atom("john"), atom("carol")), True),
        ("ancestor(mary, dave)", compound("ancestor", atom("mary"), atom("dave")), True),
        ("parent(john, bob)", compound("parent", atom("john"), atom("bob")), True),
        ("ancestor(carol, john)", compound("ancestor", atom("carol"), atom("john")), False),
    ]
    run_experiment(family_kb, "Family KB", client, family_checks)

    # Experiment 2: Graph KB
    graph_kb = build_graph_kb()
    graph_checks = [
        ("reachable(a, d)", compound("reachable", atom("a"), atom("d")), True),
        ("reachable(a, f)", compound("reachable", atom("a"), atom("f")), True),
        ("edge(a, b)", compound("edge", atom("a"), atom("b")), True),
        ("reachable(f, a)", compound("reachable", atom("f"), atom("a")), False),
    ]
    run_experiment(graph_kb, "Graph KB", client, graph_checks)

    print(f"\n{'='*70}")
    print(f"  Experiments complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
