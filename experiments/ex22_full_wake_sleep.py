#!/usr/bin/env python3
"""
EX22: Full wake-sleep cycle (LLM wake + dream compression).

The complete DreamCoder-style loop:
  1. Seed a KB with ground facts
  2. Wake: query undefined predicates → LLM generates rules
  3. Dream: compress the KB (symbolic + LLM ops)
  4. Repeat: query again, see if compressed KB + new rules enable more

This tests the actual product use case: a user populates facts, asks
questions the system can't answer, and the LLM fills in the gaps.

Usage:
    python experiments/ex22_full_wake_sleep.py
"""

import sys
import time

sys.path.insert(0, ".")

from dreamlog.pythonic import DreamLog
from dreamlog.knowledge import Fact, Rule
from dreamlog.factories import atom, var, compound
from dreamlog.terms import Compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.llm_client import LLMClient


def run_experiment():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  EX22: Full Wake-Sleep Cycle")
    print(f"{'='*70}")

    # ── Phase 0: Seed KB with ground facts ────────────────────────
    print(f"\n  Phase 0: Seed KB", flush=True)

    dl = DreamLog(
        llm_provider="anthropic",
        api_key_env=args.api_key_env,
        model=args.model,
        temperature=0.3,
        max_tokens=500,
    )

    # Family facts — no rules, just raw data
    dl.fact("parent", "john", "bob")
    dl.fact("parent", "john", "alice")
    dl.fact("parent", "mary", "bob")
    dl.fact("parent", "mary", "alice")
    dl.fact("parent", "bob", "carol")
    dl.fact("parent", "bob", "dave")
    dl.fact("parent", "alice", "eve")

    dl.fact("male", "john")
    dl.fact("male", "bob")
    dl.fact("male", "dave")
    dl.fact("female", "mary")
    dl.fact("female", "alice")
    dl.fact("female", "carol")
    dl.fact("female", "eve")

    kb = dl.engine.kb
    print(f"  Seeded: {len(kb)} clauses ({len(kb.facts)} facts, {len(kb.rules)} rules)")

    # ── Phase 1: Wake — query undefined predicates ────────────────
    print(f"\n  Phase 1: Wake (query undefined predicates)", flush=True)

    # These predicates don't exist yet — the LLM hook should generate rules
    wake_queries = [
        ("grandparent", "john", "carol"),
        ("grandparent", "john", "eve"),
        ("ancestor", "john", "carol"),
    ]

    for functor, *args_list in wake_queries:
        print(f"\n    Query: ({functor} {' '.join(args_list)})", flush=True)
        t0 = time.perf_counter()
        results = list(dl.query(functor, *args_list))
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"    Results: {len(results)} solutions ({elapsed:.0f}ms)", flush=True)
        for r in results[:3]:
            print(f"      {r}", flush=True)

    print(f"\n  After wake: {len(kb)} clauses ({len(kb.facts)} facts, {len(kb.rules)} rules)")
    print(f"  Rules generated:")
    for rule in kb.rules:
        print(f"    {rule}")

    # ── Phase 2: Dream — compress the KB ──────────────────────────
    print(f"\n  Phase 2: Dream (compress)", flush=True)

    # Get the LLM client from the provider chain for Op G
    provider = dl.engine.llm_hook.provider if dl.engine.llm_hook else None
    # Unwrap retry wrapper to get the base LLMClient
    base_provider = provider
    while hasattr(base_provider, '_base_provider'):
        base_provider = base_provider._base_provider
    if not hasattr(base_provider, 'usage'):
        base_provider = None

    dreamer = KnowledgeBaseDreamer(llm_client=base_provider)
    before = len(kb)
    t0 = time.perf_counter()
    session = dreamer.dream(kb, verify=True)
    dream_time = time.perf_counter() - t0

    print(f"  Dream: {before} -> {len(kb)} clauses in {dream_time:.1f}s")
    for op in session.operations:
        if op.new_clauses:
            for c in op.new_clauses:
                print(f"    {op.operation}: + {c}")

    # ── Phase 3: Verify — do queries still work? ──────────────────
    print(f"\n  Phase 3: Verify", flush=True)

    ev = PrologEvaluator(kb)
    checks = [
        ("parent(john, bob)", compound("parent", atom("john"), atom("bob")), True),
        ("parent(bob, carol)", compound("parent", atom("bob"), atom("carol")), True),
        ("parent(carol, john)", compound("parent", atom("carol"), atom("john")), False),
    ]

    # Also check the LLM-generated predicates
    for functor, *args_list in wake_queries:
        term = compound(functor, *[atom(a) for a in args_list])
        checks.append((f"{functor}({', '.join(args_list)})", term, True))

    all_pass = True
    for label, query, expected in checks:
        result = ev.has_solution(query)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"    {status}: {label}")

    # ── Phase 4: Second wake — can the compressed KB answer more? ──
    print(f"\n  Phase 4: Second wake (new queries on compressed KB)", flush=True)

    new_queries = [
        ("grandparent", "mary", "carol"),
        ("grandparent", "mary", "dave"),
        ("ancestor", "mary", "eve"),
        ("ancestor", "john", "eve"),
    ]

    for functor, *args_list in new_queries:
        term = compound(functor, *[atom(a) for a in args_list])
        result = ev.has_solution(term)
        print(f"    ({functor} {' '.join(args_list)}) = {result}", flush=True)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Seed facts:        {14}")
    print(f"  After wake:        {before} clauses (LLM generated {before - 14} rules)")
    print(f"  After dream:       {len(kb)} clauses")
    print(f"  Correctness:       {'ALL PASS' if all_pass else 'FAIL'}")
    print(f"  Final KB:")
    print(f"    Facts:  {len(kb.facts)}")
    print(f"    Rules:  {len(kb.rules)}")
    for rule in kb.rules:
        print(f"      {rule}")

    if base_provider and hasattr(base_provider, 'usage'):
        print(f"  LLM usage: {base_provider.usage}")


if __name__ == "__main__":
    run_experiment()
