#!/usr/bin/env python3
"""
Sleep cycle benchmarks for DreamLog.

Measures compression ratio, operations fired, correctness, and timing
across KB scenarios of varying size and structure. Establishes a baseline
for comparing future improvements (LLM-assisted compression, etc.).

Run:
    python benchmarks/sleep_cycle_bench.py
    python benchmarks/sleep_cycle_bench.py --verbose
    python benchmarks/sleep_cycle_bench.py --scenario family_with_guards
    python benchmarks/sleep_cycle_bench.py --json > baseline.json
"""

import json
import math
import time
import sys
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict

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
    body_goals_before: int
    body_goals_after: int
    compression_ratio: float
    op_summary: Dict[str, int]  # operation -> count
    correctness_passed: int
    correctness_total: int
    elapsed_ms: float


# ============================================================
# Scenarios
# ============================================================

def scenario_family_small():
    """Small family KB. Too small for any operation to fire. Baseline."""
    kb = KnowledgeBase()
    for p, c in [("john","mary"),("john","tom"),("mary","alice"),
                 ("tom","bob"),("alice","charlie"),("john","sam"),
                 ("sam","sarah"),("sarah","ben"),("sarah","jessica"),
                 ("sam","will"),("jessica","alice")]:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("parent", var("Y"), var("Z"))]))
    checks = [
        ("parent(john,mary)", compound("parent", atom("john"), atom("mary")), True),
        ("grandparent(john,alice)", compound("grandparent", atom("john"), atom("alice")), True),
        ("grandparent(alice,john)", compound("grandparent", atom("alice"), atom("john")), False),
    ]
    return "family_small", kb, checks


def scenario_family_with_guards():
    """Exercises Ops B, C, D, E together. 8 people, gender guards, 3 transitive
    closures with different base relations, shared body prefixes."""
    kb = KnowledgeBase()
    people = ["alice","bob","carol","dave","eve","frank","grace","henry"]
    for name in people:
        kb.add_fact(compound("person", atom(name)))
    for name in ["alice","carol","eve","grace"]:
        kb.add_fact(compound("female", atom(name)))
    for name in ["bob","dave","frank","henry"]:
        kb.add_fact(compound("male", atom(name)))
    for name in ["alice","bob","carol","dave","eve","frank"]:
        kb.add_fact(compound("likes", atom(name), atom("chocolate")))
    for p, c in [("alice","bob"),("alice","carol"),("bob","dave"),
                 ("bob","eve"),("carol","frank"),("carol","grace")]:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    for a, b in [("alice","bob"),("bob","dave"),("dave","henry")]:
        kb.add_fact(compound("follows", atom(a), atom(b)))
    for a, b in [("alice","carol"),("carol","frank")]:
        kb.add_fact(compound("manages", atom(a), atom(b)))

    # 3 transitive closures with DIFFERENT base relations (Op D)
    for head, base in [("ancestor","parent"),("reachable","follows"),("connected","manages")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))
    # Shared body prefix rules (Op E)
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
    # Redundant fact (Op B)
    kb.add_fact(compound("ancestor", atom("alice"), atom("bob")))
    checks = [
        ("ancestor(alice,dave)", compound("ancestor", atom("alice"), atom("dave")), True),
        ("grandparent(alice,dave)", compound("grandparent", atom("alice"), atom("dave")), True),
        ("likes(alice,chocolate)", compound("likes", atom("alice"), atom("chocolate")), True),
        ("likes(henry,chocolate)", compound("likes", atom("henry"), atom("chocolate")), False),
    ]
    return "family_with_guards", kb, checks


def scenario_transitive_closures():
    """4 structurally identical transitive closure predicates (Op D target)."""
    kb = KnowledgeBase()
    preds = [
        ("ancestor","parent",[("a","b"),("b","c"),("c","d"),("d","e")]),
        ("reachable","edge",[("x","y"),("y","z"),("z","w")]),
        ("connected","link",[("p","q"),("q","r"),("r","s"),("s","t")]),
        ("above","over",[("1","2"),("2","3"),("3","4")]),
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
        ("ancestor(a,e)", compound("ancestor", atom("a"), atom("e")), True),
        ("reachable(x,w)", compound("reachable", atom("x"), atom("w")), True),
        ("connected(p,t)", compound("connected", atom("p"), atom("t")), True),
        ("above(1,4)", compound("above", atom("1"), atom("4")), True),
        ("ancestor(e,a)", compound("ancestor", atom("e"), atom("a")), False),
    ]
    return "transitive_closures", kb, checks


def scenario_body_patterns():
    """4 rules sharing body sub-sequence parent(X,Y),parent(Y,Z) (Op E target)."""
    kb = KnowledgeBase()
    for p, c in [("a","b"),("b","c"),("c","d"),("d","e"),("e","f")]:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    kb.add_fact(compound("brother", atom("d"), atom("x")))
    kb.add_fact(compound("sister", atom("d"), atom("y")))
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
        ("grandparent(a,c)", compound("grandparent", atom("a"), atom("c")), True),
        ("great_gp(a,d)", compound("great_gp", atom("a"), atom("d")), True),
        ("great_uncle(b,x)", compound("great_uncle", atom("b"), atom("x")), True),
        ("great_aunt(b,y)", compound("great_aunt", atom("b"), atom("y")), True),
        ("grandparent(c,a)", compound("grandparent", atom("c"), atom("a")), False),
    ]
    return "body_patterns", kb, checks


def scenario_dead_clauses():
    """KB with dead clauses after broad wake phase (Op F target).
    Many alive predicates + some dead ones. Coverage > 50% so Op F fires."""
    kb = KnowledgeBase()
    # Alive: parent, anc, person, likes (4 used functors)
    for p, c in [("a","b"),("b","c"),("c","d")]:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    for n in ["a","b","c","d"]:
        kb.add_fact(compound("person", atom(n)))
    for n in ["a","b","c"]:
        kb.add_fact(compound("likes", atom(n), atom("chocolate")))
    kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                     [compound("parent", var("X"), var("Y"))]))
    kb.add_rule(Rule(compound("anc", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("anc", var("Y"), var("Z"))]))
    # Dead: unused, dead_pred (2 dead functors)
    # Total: 6 functors, 4 used = 67% coverage > 50%
    for v in ["s1","s2","s3"]:
        kb.add_fact(compound("unused", atom(v)))
    kb.add_rule(Rule(compound("dead_pred", var("X")),
                     [compound("nonexistent", var("X"))]))

    # Wake phase: exercise all alive predicates
    ev = PrologEvaluator(kb)
    for _ in range(5):
        list(ev.query([compound("anc", atom("a"), atom("d"))]))
        list(ev.query([compound("parent", atom("a"), atom("b"))]))
        list(ev.query([compound("person", atom("a"))]))
        list(ev.query([compound("likes", atom("a"), atom("chocolate"))]))

    checks = [
        ("anc(a,d)", compound("anc", atom("a"), atom("d")), True),
        ("parent(a,b)", compound("parent", atom("a"), atom("b")), True),
        ("likes(a,chocolate)", compound("likes", atom("a"), atom("chocolate")), True),
    ]
    return "dead_clauses", kb, checks


def scenario_cascading():
    """Designed so multiple operations interact. Op B prunes a redundant fact,
    Op C generalizes remaining facts, Op D invents from rule sets,
    Op E extracts body patterns. All in one dream cycle."""
    kb = KnowledgeBase()
    # Guard predicate
    for n in ["a","b","c","d","e"]:
        kb.add_fact(compound("node", atom(n)))
    # Facts for Op C: most nodes are active
    for n in ["a","b","c","d"]:
        kb.add_fact(compound("active", atom(n), atom("true")))
    # 3 transitive closures with different base relations (Op D)
    for a, b in [("a","b"),("b","c"),("c","d")]:
        kb.add_fact(compound("edge", atom(a), atom(b)))
    for a, b in [("a","c"),("c","e")]:
        kb.add_fact(compound("link", atom(a), atom(b)))
    for a, b in [("a","d"),("d","e")]:
        kb.add_fact(compound("path", atom(a), atom(b)))
    for head, base in [("reach_e","edge"),("reach_l","link"),("reach_p","path")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))
    # Rules with shared body prefix (Op E)
    kb.add_rule(Rule(compound("two_hop", var("X"), var("Z")),
                     [compound("edge", var("X"), var("Y")),
                      compound("edge", var("Y"), var("Z"))]))
    kb.add_rule(Rule(compound("three_hop", var("X"), var("W")),
                     [compound("edge", var("X"), var("Y")),
                      compound("edge", var("Y"), var("Z")),
                      compound("edge", var("Z"), var("W"))]))
    kb.add_rule(Rule(compound("hop_then_link", var("X"), var("W")),
                     [compound("edge", var("X"), var("Y")),
                      compound("edge", var("Y"), var("Z")),
                      compound("link", var("Z"), var("W"))]))
    # Redundant fact (Op B)
    kb.add_fact(compound("reach_e", atom("a"), atom("b")))

    checks = [
        ("reach_e(a,d)", compound("reach_e", atom("a"), atom("d")), True),
        ("reach_l(a,e)", compound("reach_l", atom("a"), atom("e")), True),
        ("two_hop(a,c)", compound("two_hop", atom("a"), atom("c")), True),
        ("three_hop(a,d)", compound("three_hop", atom("a"), atom("d")), True),
        ("active(a,true)", compound("active", atom("a"), atom("true")), True),
        ("active(e,true)", compound("active", atom("e"), atom("true")), False),
    ]
    return "cascading", kb, checks


def scenario_stress():
    """Larger KB: 50 entities, 100+ facts, multiple rule patterns."""
    kb = KnowledgeBase()
    n = 50
    names = [f"e{i}" for i in range(n)]
    # Type facts
    for name in names:
        kb.add_fact(compound("entity", atom(name)))
    # Category: most are type_a, some type_b (Op C)
    for name in names[:40]:
        kb.add_fact(compound("category", atom(name), atom("type_a")))
    for name in names[40:]:
        kb.add_fact(compound("category", atom(name), atom("type_b")))
    # Chain relations
    for i in range(n - 1):
        kb.add_fact(compound("next", atom(names[i]), atom(names[i+1])))
    for i in range(0, n - 1, 2):
        kb.add_fact(compound("skip", atom(names[i]), atom(names[min(i+2, n-1)])))
    # 4 transitive closures (Op D)
    for head, base in [("chain","next"),("skip_chain","skip"),
                       ("forward","next"),("hop","skip")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))
    # 3 rules with shared body prefix (Op E)
    kb.add_rule(Rule(compound("two_next", var("X"), var("Z")),
                     [compound("next", var("X"), var("Y")),
                      compound("next", var("Y"), var("Z"))]))
    kb.add_rule(Rule(compound("three_next", var("X"), var("W")),
                     [compound("next", var("X"), var("Y")),
                      compound("next", var("Y"), var("Z")),
                      compound("next", var("Z"), var("W"))]))
    kb.add_rule(Rule(compound("next_then_skip", var("X"), var("W")),
                     [compound("next", var("X"), var("Y")),
                      compound("next", var("Y"), var("Z")),
                      compound("skip", var("Z"), var("W"))]))

    checks = [
        (f"chain(e0,e10)", compound("chain", atom("e0"), atom("e10")), True),
        (f"two_next(e0,e2)", compound("two_next", atom("e0"), atom("e2")), True),
        (f"category(e0,type_a)", compound("category", atom("e0"), atom("type_a")), True),
        (f"category(e0,type_b)", compound("category", atom("e0"), atom("type_b")), False),
        (f"chain(e10,e0)", compound("chain", atom("e10"), atom("e0")), False),
    ]
    return "stress", kb, checks


def scenario_subsumption_edge():
    """Tests cross-body variable binding in clause subsumption (Op A).

    A general rule f(X) :- g(X,Y), h(Y) should subsume f(a) :- g(a,b), h(b)
    (consistent Y=b) but NOT f(a) :- g(a,b), h(c) (inconsistent Y=b vs Y=c).

    Without the binding propagation fix in clause_subsumes, the inconsistent
    rule would be incorrectly removed, changing query behavior.
    """
    kb = KnowledgeBase()

    # General rule: f(X) :- g(X, Y), h(Y)
    kb.add_rule(Rule(compound("derive", var("X"), var("Z")),
                     [compound("step1", var("X"), var("Y")),
                      compound("step2", var("Y"), var("Z"))]))

    # Consistent specialization (Y=b in both): SHOULD be subsumed
    kb.add_rule(Rule(compound("derive", atom("a"), atom("c")),
                     [compound("step1", atom("a"), atom("b")),
                      compound("step2", atom("b"), atom("c"))]))

    # Inconsistent specialization (Y=b in step1, Y=d in step2): MUST be kept
    kb.add_rule(Rule(compound("derive", atom("a"), atom("e")),
                     [compound("step1", atom("a"), atom("b")),
                      compound("step2", atom("d"), atom("e"))]))

    # Another inconsistent one
    kb.add_rule(Rule(compound("derive", atom("x"), atom("z")),
                     [compound("step1", atom("x"), atom("y1")),
                      compound("step2", atom("y2"), atom("z"))]))

    # Facts for verification
    kb.add_fact(compound("step1", atom("a"), atom("b")))
    kb.add_fact(compound("step1", atom("x"), atom("y1")))
    kb.add_fact(compound("step2", atom("b"), atom("c")))
    kb.add_fact(compound("step2", atom("d"), atom("e")))
    kb.add_fact(compound("step2", atom("y2"), atom("z")))

    checks = [
        # derive(a, c) works via general rule (step1(a,b), step2(b,c))
        ("derive(a, c)", compound("derive", atom("a"), atom("c")), True),
        # derive(a, e) only works via the inconsistent specific rule
        # (step1(a,b) gives Y=b, but step2(d,e) needs Y=d, so general rule fails)
        # The specific rule hardcodes the right path
        ("derive(a, e)", compound("derive", atom("a"), atom("e")), True),
        # derive(x, z) only works via the inconsistent specific rule
        ("derive(x, z)", compound("derive", atom("x"), atom("z")), True),
        # derive(a, z) should not work
        ("derive(a, z)", compound("derive", atom("a"), atom("z")), False),
    ]
    return "subsumption_edge", kb, checks


SCENARIOS = {
    "family_small": scenario_family_small,
    "family_with_guards": scenario_family_with_guards,
    "transitive_closures": scenario_transitive_closures,
    "body_patterns": scenario_body_patterns,
    "dead_clauses": scenario_dead_clauses,
    "cascading": scenario_cascading,
    "stress": scenario_stress,
    "subsumption_edge": scenario_subsumption_edge,
}


# ============================================================
# Runner
# ============================================================

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
    body_goals_after = sum(len(r.body) for r in kb.rules)

    # Aggregate operations by type
    op_summary: Dict[str, int] = {}
    for op in session.operations:
        op_summary[op.operation] = op_summary.get(op.operation, 0) + 1

    if verbose and op_summary:
        print(f"  Operations:")
        for op_name, count in sorted(op_summary.items()):
            print(f"    {op_name}: {count}x")

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

    result = BenchmarkResult(
        name=name,
        clauses_before=clauses_before,
        clauses_after=clauses_after,
        facts_before=facts_before,
        facts_after=facts_after,
        rules_before=rules_before,
        rules_after=rules_after,
        body_goals_before=body_goals_before,
        body_goals_after=body_goals_after,
        compression_ratio=session.compression_ratio,
        op_summary=op_summary,
        correctness_passed=passed,
        correctness_total=total,
        elapsed_ms=elapsed,
    )

    if verbose:
        print(f"  {clauses_before} -> {clauses_after} clauses, {body_goals_before} -> {body_goals_after} body goals")
        print(f"  Ratio: {session.compression_ratio:.2f}, Time: {elapsed:.1f}ms, Correct: {passed}/{total}")

    return result


def print_summary(results: List[BenchmarkResult]):
    print(f"\n{'='*80}")
    print(f"  SLEEP CYCLE BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"  {'Scenario':<22} {'Clauses':>11} {'Body goals':>11} {'Ops':>5} {'OK':>6} {'Time':>8}")
    print(f"  {'-'*22} {'-'*11} {'-'*11} {'-'*5} {'-'*6} {'-'*8}")

    all_correct = True
    for r in results:
        clause_str = f"{r.clauses_before}->{r.clauses_after}"
        body_str = f"{r.body_goals_before}->{r.body_goals_after}"
        ops_count = sum(r.op_summary.values())
        ok_str = f"{r.correctness_passed}/{r.correctness_total}"
        status = "" if r.correctness_passed == r.correctness_total else " FAIL"
        if status:
            all_correct = False
        print(f"  {r.name:<22} {clause_str:>11} {body_str:>11} {ops_count:>5} {ok_str:>6} {r.elapsed_ms:>7.1f}ms{status}")

    # Operation breakdown
    op_totals: Dict[str, int] = {}
    for r in results:
        for op_name, count in r.op_summary.items():
            op_totals[op_name] = op_totals.get(op_name, 0) + count

    if op_totals:
        print(f"\n  Operations across all scenarios:")
        for op_name, count in sorted(op_totals.items()):
            print(f"    {op_name:<25} {count:>3}x")

    tc = sum(r.clauses_before for r in results)
    ta = sum(r.clauses_after for r in results)
    tb = sum(r.body_goals_before for r in results)
    tba = sum(r.body_goals_after for r in results)
    tt = sum(r.elapsed_ms for r in results)
    print(f"\n  Totals:")
    print(f"    Clauses:    {tc} -> {ta} ({ta/tc:.2f})")
    if tb > 0:
        print(f"    Body goals: {tb} -> {tba} ({tba/tb:.2f})")
    print(f"    Time:       {tt:.0f}ms")
    print(f"    Correct:    {'ALL PASSED' if all_correct else 'SOME FAILED'}")
    print(f"{'='*80}")


def results_to_json(results: List[BenchmarkResult]) -> str:
    data = {
        "scenarios": [asdict(r) for r in results],
        "totals": {
            "clauses_before": sum(r.clauses_before for r in results),
            "clauses_after": sum(r.clauses_after for r in results),
            "body_goals_before": sum(r.body_goals_before for r in results),
            "body_goals_after": sum(r.body_goals_after for r in results),
            "all_correct": all(r.correctness_passed == r.correctness_total for r in results),
        }
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(description="DreamLog sleep cycle benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--scenario", "-s", help="Run a single scenario")
    parser.add_argument("--json", action="store_true", help="Output JSON for comparison")
    args = parser.parse_args()

    if args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown: {args.scenario}. Available: {', '.join(SCENARIOS.keys())}")
            sys.exit(1)
        results = [run_benchmark(SCENARIOS[args.scenario], verbose=args.verbose)]
    else:
        results = [run_benchmark(fn, verbose=args.verbose) for fn in SCENARIOS.values()]

    if args.json:
        print(results_to_json(results))
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
