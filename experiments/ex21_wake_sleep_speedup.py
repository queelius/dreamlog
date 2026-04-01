#!/usr/bin/env python3
"""
EX21: Wake-sleep query speedup with LLM-assisted dreaming.

Tests the thesis: LLM-discovered rules don't just compress the KB, they
change the resolution topology, making queries faster.

Compares three conditions:
  1. No dreaming (baseline)
  2. Symbolic-only dreaming (Op A-F, H)
  3. LLM-assisted dreaming (Op A-H including Op G)

Measures query time before and after dreaming.

Usage:
    python experiments/ex21_wake_sleep_speedup.py
"""

import sys
import time

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient


def build_kb():
    """Family KB where LLM rules create query shortcuts.

    Key insight: without father/mother rules, queries like
    "is john the father of bob?" require the USER to know to check
    parent(john, bob) AND male(john) separately. With the rule
    father(X,Y) :- parent(X,Y), male(X), a single father(john, bob)
    query resolves in one rule application.

    Also includes grandparent queries that require multi-step resolution
    through parent chains, and ancestor queries with transitive closure.
    """
    kb = KnowledgeBase()

    people = {
        "john": "male", "mary": "female",
        "bob": "male", "alice": "female",
        "carol": "female", "dave": "male",
        "eve": "female", "frank": "male",
        "grace": "female", "henry": "male",
        "iris": "female", "jack": "male",
    }
    for name, gender in people.items():
        kb.add_fact(compound("person", atom(name)))
        kb.add_fact(compound(gender, atom(name)))

    parents = [
        ("john", "bob"), ("john", "alice"),
        ("mary", "bob"), ("mary", "alice"),
        ("bob", "carol"), ("bob", "dave"),
        ("alice", "eve"), ("alice", "frank"),
        ("carol", "grace"), ("carol", "henry"),
        ("dave", "iris"), ("dave", "jack"),
    ]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))

    # Father/mother facts (LLM should discover these as derivable)
    for p, c in parents:
        if people[p] == "male":
            kb.add_fact(compound("father", atom(p), atom(c)))
        else:
            kb.add_fact(compound("mother", atom(p), atom(c)))

    # Grandparent facts (derivable from parent chain)
    for gp in people:
        for mid in people:
            for gc in people:
                if any(p == gp and c == mid for p, c in parents) and \
                   any(p == mid and c == gc for p, c in parents):
                    kb.add_fact(compound("grandparent", atom(gp), atom(gc)))

    # Ancestor rules (transitive closure)
    kb.add_rule(Rule(compound("ancestor", var("X"), var("Y")),
                     [compound("parent", var("X"), var("Y"))]))
    kb.add_rule(Rule(compound("ancestor", var("X"), var("Z")),
                     [compound("parent", var("X"), var("Y")),
                      compound("ancestor", var("Y"), var("Z"))]))

    return kb, people, parents


def build_test_queries(people, parents):
    """Build a diverse set of queries that exercise different resolution depths."""
    queries = []

    # Depth-1: direct fact lookups (father, mother)
    for p, c in parents:
        queries.append(("father" if people[p] == "male" else "mother",
                        compound("father" if people[p] == "male" else "mother",
                                 atom(p), atom(c))))

    # Depth-2: grandparent queries
    for gp in ["john", "mary"]:
        for gc in ["carol", "dave", "eve", "frank"]:
            queries.append(("grandparent",
                            compound("grandparent", atom(gp), atom(gc))))

    # Depth-3+: ancestor queries (require transitive resolution)
    for anc, desc in [("john", "grace"), ("john", "henry"),
                      ("mary", "iris"), ("mary", "jack"),
                      ("john", "iris"), ("mary", "grace")]:
        queries.append(("ancestor",
                        compound("ancestor", atom(anc), atom(desc))))

    # Negative queries (should not be derivable)
    for a, b in [("carol", "john"), ("dave", "mary"), ("grace", "bob")]:
        queries.append(("ancestor_neg",
                        compound("ancestor", atom(a), atom(b))))

    # Guard predicate queries (ensure male/female/person aren't pruned by Op F)
    for name, gender in people.items():
        queries.append(("guard",
                        compound(gender, atom(name))))

    return queries


def benchmark_queries(kb, queries, n_iterations=10):
    """Run queries n times and return total time + per-query stats."""
    ev = PrologEvaluator(kb)
    results_by_type = {}

    # Warm up
    for _, q in queries:
        ev.has_solution(q)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        for _, q in queries:
            ev.has_solution(q)
    total_ms = (time.perf_counter() - t0) * 1000

    # Per-type timing
    for qtype, q in queries:
        ev2 = PrologEvaluator(kb)
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            ev2.has_solution(q)
        elapsed = (time.perf_counter() - t0) * 1000 / n_iterations
        results_by_type.setdefault(qtype, []).append(elapsed)

    type_avgs = {k: sum(v) / len(v) for k, v in results_by_type.items()}
    return total_ms, type_avgs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--iterations", type=int, default=20,
                        help="Query iterations for timing")
    args = parser.parse_args()

    client = LLMClient(
        provider=args.provider, model=args.model, api_key=args.api_key,
        api_key_env=args.api_key_env, base_url=args.base_url,
        temperature=0.3, max_tokens=800)

    print(f"Provider: {client.provider}, Model: {client.model}")
    n_iter = args.iterations

    kb_orig, people, parents = build_kb()
    queries = build_test_queries(people, parents)
    print(f"KB: {len(kb_orig)} clauses, {len(queries)} test queries, {n_iter} iterations")

    # ── Condition 1: No dreaming (baseline) ───────────────────────
    print(f"\n{'='*70}")
    print(f"  Condition 1: No dreaming (baseline)")
    print(f"{'='*70}")

    kb_baseline = kb_orig.copy()
    time_baseline, types_baseline = benchmark_queries(
        kb_baseline, queries, n_iter)
    print(f"  KB: {len(kb_baseline)} clauses")
    print(f"  Total query time: {time_baseline:.1f}ms ({n_iter} iterations)")
    for qtype, avg in sorted(types_baseline.items()):
        print(f"    {qtype:<15} {avg:.3f}ms/query")

    # ── Condition 2: Symbolic-only dreaming ───────────────────────
    print(f"\n{'='*70}")
    print(f"  Condition 2: Symbolic-only dreaming")
    print(f"{'='*70}")

    kb_sym = kb_orig.copy()
    # Wake phase: run queries to build usage data
    kb_sym.enable_derivation_tracking()
    ev_wake = PrologEvaluator(kb_sym)
    for _ in range(5):
        for _, q in queries:
            list(ev_wake.query([q]))

    dreamer_sym = KnowledgeBaseDreamer()
    t0 = time.perf_counter()
    session_sym = dreamer_sym.dream(kb_sym, verify=True)
    dream_time_sym = (time.perf_counter() - t0) * 1000

    sym_ops = {}
    for op in session_sym.operations:
        sym_ops[op.operation] = sym_ops.get(op.operation, 0) + 1

    time_sym, types_sym = benchmark_queries(kb_sym, queries, n_iter)
    speedup_sym = time_baseline / time_sym if time_sym > 0 else float('inf')

    print(f"  Dream: {dream_time_sym:.0f}ms, ops: {sym_ops}")
    print(f"  KB: {len(kb_sym)} clauses")
    print(f"  Total query time: {time_sym:.1f}ms")
    print(f"  Speedup: {speedup_sym:.1f}x")
    for qtype in sorted(types_baseline):
        before = types_baseline[qtype]
        after = types_sym.get(qtype, before)
        sp = before / after if after > 0 else float('inf')
        print(f"    {qtype:<15} {before:.3f} -> {after:.3f}ms ({sp:.1f}x)")

    # ── Condition 3: LLM-assisted dreaming ────────────────────────
    print(f"\n{'='*70}")
    print(f"  Condition 3: LLM-assisted dreaming")
    print(f"{'='*70}")

    kb_llm = kb_orig.copy()
    # Wake phase
    kb_llm.enable_derivation_tracking()
    ev_wake_llm = PrologEvaluator(kb_llm)
    for _ in range(5):
        for _, q in queries:
            list(ev_wake_llm.query([q]))

    dreamer_llm = KnowledgeBaseDreamer(llm_client=client)
    t0 = time.perf_counter()
    session_llm = dreamer_llm.dream(kb_llm, verify=True)
    dream_time_llm = (time.perf_counter() - t0) * 1000

    llm_ops = {}
    for op in session_llm.operations:
        llm_ops[op.operation] = llm_ops.get(op.operation, 0) + 1

    llm_rules = []
    for op in session_llm.operations:
        if op.operation == "llm_compression":
            for c in op.new_clauses:
                llm_rules.append(str(c))

    time_llm, types_llm = benchmark_queries(kb_llm, queries, n_iter)
    speedup_llm = time_baseline / time_llm if time_llm > 0 else float('inf')

    print(f"  Dream: {dream_time_llm:.0f}ms, ops: {llm_ops}")
    print(f"  KB: {len(kb_llm)} clauses")
    print(f"  LLM rules discovered:")
    for r in llm_rules:
        print(f"    + {r}")
    print(f"  Total query time: {time_llm:.1f}ms")
    print(f"  Speedup: {speedup_llm:.1f}x")
    for qtype in sorted(types_baseline):
        before = types_baseline[qtype]
        after = types_llm.get(qtype, before)
        sp = before / after if after > 0 else float('inf')
        print(f"    {qtype:<15} {before:.3f} -> {after:.3f}ms ({sp:.1f}x)")

    # ── Verify correctness ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Correctness verification")
    print(f"{'='*70}")

    ev_check = PrologEvaluator(kb_llm)
    ev_base = PrologEvaluator(kb_baseline)
    mismatches = 0
    for qtype, q in queries:
        r_base = ev_base.has_solution(q)
        r_llm = ev_check.has_solution(q)
        if r_base != r_llm:
            mismatches += 1
            print(f"  MISMATCH: {q} baseline={r_base} llm={r_llm}")
    print(f"  {len(queries) - mismatches}/{len(queries)} match "
          f"({'ALL PASS' if mismatches == 0 else f'{mismatches} MISMATCHES'})")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':<25} {'Clauses':>8} {'Query ms':>10} {'Speedup':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8}")
    print(f"  {'No dreaming':<25} {len(kb_baseline):>8} {time_baseline:>10.1f} {'1.0x':>8}")
    print(f"  {'Symbolic dream':<25} {len(kb_sym):>8} {time_sym:>10.1f} {speedup_sym:>7.1f}x")
    print(f"  {'LLM dream':<25} {len(kb_llm):>8} {time_llm:>10.1f} {speedup_llm:>7.1f}x")
    print(f"  LLM usage: {client.usage}")


if __name__ == "__main__":
    main()
