#!/usr/bin/env python3
"""
Sleep cycle benchmarks for DreamLog.

Measures compression ratio, operations fired, correctness, and timing
across KB scenarios of varying size and structure. Establishes a baseline
for comparing future improvements (LLM-assisted compression, etc.).

Run:
    python benchmarks/sleep_cycle_bench.py
    python benchmarks/sleep_cycle_bench.py --verbose
    python benchmarks/sleep_cycle_bench.py --scenario family
"""

import time
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer, DreamSession


@dataclass
class BenchmarkResult:
    name: str
    clauses_before: int
    clauses_after: int
    facts_before: int
    facts_after: int
    rules_before: int
    rules_after: int
    compression_ratio: float
    operations: List[Tuple[str, int]]  # (name, mdl_delta)
    verification_queries: int
    verification_passed: bool
    correctness_checks_passed: int
    correctness_checks_total: int
    body_goals_before: int
    body_goals_after: int
    elapsed_ms: float


def scenario_family_small() -> Tuple[str, KnowledgeBase, List[Tuple[str, Compound, bool]]]:
    """Small family KB: 11 parent facts, 1 grandparent rule."""
    kb = KnowledgeBase()
    parents = [
        ("john", "mary"), ("john", "tom"), ("mary", "alice"),
        ("tom", "bob"), ("alice", "charlie"), ("john", "sam"),
        ("sam", "sarah"), ("sarah", "ben"), ("sarah", "jessica"),
        ("sam", "will"), ("jessica", "alice"),
    ]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z"))]))

    checks = [
        ("parent(john, mary)", compound("parent", atom("john"), atom("mary")), True),
        ("grandparent(john, alice)", compound("grandparent", atom("john"), atom("alice")), True),
        ("grandparent(alice, john)", compound("grandparent", atom("alice"), atom("john")), False),
    ]
    return "family_small", kb, checks


def scenario_family_with_guards() -> Tuple[str, KnowledgeBase, List[Tuple[str, Compound, bool]]]:
    """Family KB with gender facts as guards for Operation C."""
    kb = KnowledgeBase()
    people = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "henry"]
    for name in people:
        kb.add_fact(compound("person", atom(name)))
    for name in ["alice", "carol", "eve", "grace"]:
        kb.add_fact(compound("female", atom(name)))
    for name in ["bob", "dave", "frank", "henry"]:
        kb.add_fact(compound("male", atom(name)))

    # Most people like chocolate (Op C target)
    for name in ["alice", "bob", "carol", "dave", "eve", "frank"]:
        kb.add_fact(compound("likes", atom(name), atom("chocolate")))

    # Parents
    for p, c in [("alice", "bob"), ("alice", "carol"), ("bob", "dave"),
                 ("bob", "eve"), ("carol", "frank"), ("carol", "grace")]:
        kb.add_fact(compound("parent", atom(p), atom(c)))

    # Different base relations for each transitive closure (Op D needs different PARAM_0)
    for a, b in [("alice", "bob"), ("bob", "dave"), ("dave", "henry")]:
        kb.add_fact(compound("follows", atom(a), atom(b)))
    for a, b in [("alice", "carol"), ("carol", "frank")]:
        kb.add_fact(compound("manages", atom(a), atom(b)))

    # Three transitive closures with DIFFERENT base relations (Op D target)
    for head, base in [("ancestor", "parent"), ("reachable", "follows"), ("connected", "manages")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    # Grandparent + great-grandparent + great-uncle (Op E target)
    kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z"))]))
    kb.add_rule(Rule(compound("great_grandparent", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("parent", var("Z"), var("W"))]))
    kb.add_rule(Rule(compound("great_uncle", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("male", var("Z")),
                      compound("parent", var("Z"), var("W"))]))

    # Redundant fact (Op B target)
    kb.add_fact(compound("ancestor", atom("alice"), atom("bob")))

    checks = [
        ("ancestor(alice, dave)", compound("ancestor", atom("alice"), atom("dave")), True),
        ("ancestor(alice, grace)", compound("ancestor", atom("alice"), atom("grace")), True),
        ("grandparent(alice, dave)", compound("grandparent", atom("alice"), atom("dave")), True),
        ("likes(alice, chocolate)", compound("likes", atom("alice"), atom("chocolate")), True),
        ("likes(henry, chocolate)", compound("likes", atom("henry"), atom("chocolate")), False),
        ("ancestor(henry, alice)", compound("ancestor", atom("henry"), atom("alice")), False),
    ]
    return "family_with_guards", kb, checks


def scenario_transitive_closures() -> Tuple[str, KnowledgeBase, List[Tuple[str, Compound, bool]]]:
    """Multiple structurally identical transitive closure predicates (Op D target)."""
    kb = KnowledgeBase()

    preds = [
        ("ancestor", "parent", [("a","b"),("b","c"),("c","d"),("d","e")]),
        ("reachable", "edge", [("x","y"),("y","z"),("z","w")]),
        ("connected", "link", [("p","q"),("q","r"),("r","s"),("s","t")]),
        ("above", "over", [("1","2"),("2","3"),("3","4")]),
    ]
    for head, base, facts in preds:
        for a, b in facts:
            kb.add_fact(compound(base, atom(a), atom(b)))
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    checks = [
        ("ancestor(a, e)", compound("ancestor", atom("a"), atom("e")), True),
        ("reachable(x, w)", compound("reachable", atom("x"), atom("w")), True),
        ("connected(p, t)", compound("connected", atom("p"), atom("t")), True),
        ("above(1, 4)", compound("above", atom("1"), atom("4")), True),
        ("ancestor(e, a)", compound("ancestor", atom("e"), atom("a")), False),
    ]
    return "transitive_closures", kb, checks


def scenario_body_patterns() -> Tuple[str, KnowledgeBase, List[Tuple[str, Compound, bool]]]:
    """Rules sharing body sub-sequences (Op E target)."""
    kb = KnowledgeBase()

    for p, c in [("a","b"),("b","c"),("c","d"),("d","e"),("e","f")]:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    kb.add_fact(compound("brother", atom("d"), atom("x")))
    kb.add_fact(compound("sister", atom("d"), atom("y")))

    # All share parent(X,Y), parent(Y,Z) prefix
    kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z"))]))
    kb.add_rule(Rule(compound("great_gp", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("parent", var("Z"), var("W"))]))
    kb.add_rule(Rule(compound("great_uncle", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("brother", var("Z"), var("W"))]))
    kb.add_rule(Rule(compound("great_aunt", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("sister", var("Z"), var("W"))]))

    checks = [
        ("grandparent(a, c)", compound("grandparent", atom("a"), atom("c")), True),
        ("great_gp(a, d)", compound("great_gp", atom("a"), atom("d")), True),
        ("great_uncle(b, x)", compound("great_uncle", atom("b"), atom("x")), True),
        ("great_aunt(b, y)", compound("great_aunt", atom("b"), atom("y")), True),
        ("grandparent(c, a)", compound("grandparent", atom("c"), atom("a")), False),
    ]
    return "body_patterns", kb, checks


def scenario_dead_clauses() -> Tuple[str, KnowledgeBase, List[Tuple[str, Compound, bool]]]:
    """KB with unused clauses after wake-phase queries (Op F target)."""
    kb = KnowledgeBase()

    # Used facts
    for p, c in [("a","b"),("b","c"),("c","d")]:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                     [compound("parent", var("X"), var("Y"))]))
    kb.add_rule(Rule(compound("anc", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("anc", var("Y"), var("Z"))]))

    # Dead facts (never queried)
    for v in ["stale1", "stale2", "stale3", "stale4", "stale5"]:
        kb.add_fact(compound("unused", atom(v)))

    # Dead rules (never fire)
    kb.add_rule(Rule(compound("dead_pred", var("X")),
                     [compound("nonexistent", var("X"))]))
    kb.add_rule(Rule(compound("also_dead", var("X")),
                     [compound("nope", var("X"))]))

    # Simulate wake phase: exercise used predicates broadly
    ev = PrologEvaluator(kb)
    for _ in range(5):
        list(ev.query([compound("anc", atom("a"), atom("d"))]))
        list(ev.query([compound("parent", atom("a"), atom("b"))]))
        list(ev.query([compound("parent", atom("b"), atom("c"))]))
        list(ev.query([compound("parent", atom("c"), atom("d"))]))
        # Do NOT query unused/dead_pred/also_dead

    checks = [
        ("anc(a, d)", compound("anc", atom("a"), atom("d")), True),
        ("parent(a, b)", compound("parent", atom("a"), atom("b")), True),
    ]
    return "dead_clauses", kb, checks


def scenario_mixed_large() -> Tuple[str, KnowledgeBase, List[Tuple[str, Compound, bool]]]:
    """Larger mixed KB exercising all operations."""
    kb = KnowledgeBase()

    # 20 person facts
    names = [f"person_{i}" for i in range(20)]
    for name in names:
        kb.add_fact(compound("person", atom(name)))

    # 15 of them like chocolate (Op C)
    for name in names[:15]:
        kb.add_fact(compound("likes", atom(name), atom("chocolate")))

    # Some like vanilla too (mixed subgroups)
    for name in names[10:18]:
        kb.add_fact(compound("likes", atom(name), atom("vanilla")))

    # Parent chain
    for i in range(len(names) - 1):
        kb.add_fact(compound("parent", atom(names[i]), atom(names[i+1])))

    # 3 transitive closures (Op D)
    for head, base in [("ancestor", "parent"), ("reaches", "parent"), ("above", "parent")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    # 3 rules with shared body prefix (Op E)
    kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z"))]))
    kb.add_rule(Rule(compound("great_gp", var("X"), var("W")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("parent", var("Z"), var("W"))]))
    kb.add_rule(Rule(compound("great_great_gp", var("X"), var("V")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z")),
                      compound("parent", var("Z"), var("W")),
                      compound("parent", var("W"), var("V"))]))

    # Redundant facts (Op B)
    kb.add_fact(compound("ancestor", atom(names[0]), atom(names[1])))

    # Simulate wake phase: exercise the full KB broadly for Op F
    ev = PrologEvaluator(kb)
    for i in range(min(5, len(names) - 5)):
        list(ev.query([compound("ancestor", atom(names[i]), atom(names[i+3]))]))
        list(ev.query([compound("reaches", atom(names[i]), atom(names[i+2]))]))
        list(ev.query([compound("above", atom(names[i]), atom(names[i+1]))]))
        list(ev.query([compound("grandparent", atom(names[i]), atom(names[i+2]))]))
        list(ev.query([compound("great_gp", atom(names[i]), atom(names[i+3]))]))
        list(ev.query([compound("likes", atom(names[i]), atom("chocolate"))]))
        list(ev.query([compound("likes", atom(names[i+10]), atom("vanilla"))]))
        list(ev.query([compound("person", atom(names[i]))]))

    checks = [
        (f"ancestor({names[0]}, {names[5]})",
         compound("ancestor", atom(names[0]), atom(names[5])), True),
        (f"likes({names[0]}, chocolate)",
         compound("likes", atom(names[0]), atom("chocolate")), True),
        (f"likes({names[19]}, chocolate)",
         compound("likes", atom(names[19]), atom("chocolate")), False),
        (f"grandparent({names[0]}, {names[2]})",
         compound("grandparent", atom(names[0]), atom(names[2])), True),
    ]
    return "mixed_large", kb, checks


SCENARIOS = {
    "family_small": scenario_family_small,
    "family_with_guards": scenario_family_with_guards,
    "transitive_closures": scenario_transitive_closures,
    "body_patterns": scenario_body_patterns,
    "dead_clauses": scenario_dead_clauses,
    "mixed_large": scenario_mixed_large,
}


def run_benchmark(scenario_fn, verbose=False) -> BenchmarkResult:
    name, kb, correctness_checks = scenario_fn()

    facts_before = len(kb.facts)
    rules_before = len(kb.rules)
    clauses_before = facts_before + rules_before

    body_goals_before = sum(len(r.body) for r in kb.rules)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {name}: {clauses_before} clauses ({facts_before} facts, {rules_before} rules)")
        print(f"{'='*60}")

    start = time.perf_counter()
    dreamer = KnowledgeBaseDreamer(min_group_size=3)
    session = dreamer.dream(kb, verify=True)
    elapsed = (time.perf_counter() - start) * 1000

    facts_after = len(kb.facts)
    rules_after = len(kb.rules)
    clauses_after = facts_after + rules_after

    ops = [(op.operation, op.mdl_delta) for op in session.operations]

    if verbose and ops:
        print(f"  Operations:")
        for op_name, delta in ops:
            print(f"    {op_name}: {delta:+d}")

    # Correctness checks
    ev = PrologEvaluator(kb)
    passed = 0
    total = len(correctness_checks)
    for label, query, expected in correctness_checks:
        result = ev.has_solution(query)
        if result == expected:
            passed += 1
        elif verbose:
            print(f"  FAIL: {label} = {result}, expected {expected}")

    body_goals_after = sum(len(r.body) for r in kb.rules)

    verification_count = 0
    if session.verification:
        verification_count = session.verification.positive_count + session.verification.negative_count

    result = BenchmarkResult(
        name=name,
        clauses_before=clauses_before,
        clauses_after=clauses_after,
        facts_before=facts_before,
        facts_after=facts_after,
        rules_before=rules_before,
        rules_after=rules_after,
        compression_ratio=session.compression_ratio,
        operations=ops,
        verification_queries=verification_count,
        verification_passed=session.verification.passed if session.verification else True,
        correctness_checks_passed=passed,
        correctness_checks_total=total,
        body_goals_before=body_goals_before,
        body_goals_after=body_goals_after,
        elapsed_ms=elapsed,
    )

    if verbose:
        print(f"  {clauses_before} -> {clauses_after} (ratio: {session.compression_ratio:.2f}, {elapsed:.1f}ms)")
        print(f"  Correctness: {passed}/{total}")

    return result


def print_summary(results: List[BenchmarkResult]):
    print(f"\n{'='*80}")
    print(f"  SLEEP CYCLE BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"  {'Scenario':<25} {'Clause':>12} {'Body goals':>12} {'Ops':>5} {'Correct':>9} {'Time':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*5} {'-'*9} {'-'*8}")

    all_correct = True
    for r in results:
        correct_str = f"{r.correctness_checks_passed}/{r.correctness_checks_total}"
        ops_count = len(r.operations)
        clause_str = f"{r.clauses_before}->{r.clauses_after}"
        body_str = f"{r.body_goals_before}->{r.body_goals_after}"
        status = "" if r.correctness_checks_passed == r.correctness_checks_total else " FAIL"
        if status:
            all_correct = False
        print(f"  {r.name:<25} {clause_str:>12} {body_str:>12} {ops_count:>5} {correct_str:>9} {r.elapsed_ms:>7.1f}ms{status}")

    # Operation breakdown
    op_counts = {}
    for r in results:
        for op_name, delta in r.operations:
            op_counts.setdefault(op_name, {"count": 0, "delta": 0})
            op_counts[op_name]["count"] += 1
            op_counts[op_name]["delta"] += delta

    print(f"\n  Operation totals across all scenarios:")
    for op_name, stats in sorted(op_counts.items()):
        print(f"    {op_name:<25} fired {stats['count']:>3}x, total delta: {stats['delta']:+d}")

    total_clauses_before = sum(r.clauses_before for r in results)
    total_clauses_after = sum(r.clauses_after for r in results)
    total_bg_before = sum(r.body_goals_before for r in results)
    total_bg_after = sum(r.body_goals_after for r in results)
    total_time = sum(r.elapsed_ms for r in results)
    print(f"\n  Clauses:    {total_clauses_before} -> {total_clauses_after} ({total_clauses_after/total_clauses_before:.2f})")
    print(f"  Body goals: {total_bg_before} -> {total_bg_after} ({total_bg_after/total_bg_before:.2f})" if total_bg_before > 0 else "")
    print(f"  Time: {total_time:.1f}ms")
    print(f"  Correctness: {'ALL PASSED' if all_correct else 'SOME FAILED'}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="DreamLog sleep cycle benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed output per scenario")
    parser.add_argument("--scenario", "-s", help="Run a single scenario")
    args = parser.parse_args()

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        results = [run_benchmark(SCENARIOS[args.scenario], verbose=args.verbose)]
    else:
        results = [run_benchmark(fn, verbose=args.verbose) for fn in SCENARIOS.values()]

    print_summary(results)


if __name__ == "__main__":
    main()
