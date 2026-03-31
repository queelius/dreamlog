#!/usr/bin/env python3
"""
Ablation study + scaling experiments.

1. Ablation: disable each operation individually, measure impact
2. Scale: generate KBs of increasing size, measure compression and time
3. Operation ordering: does the order A-H matter?
4. Wake-sleep loop: query -> dream -> query, measure performance improvement

Usage:
    python experiments/ablation_and_scale.py
    python experiments/ablation_and_scale.py --experiment ablation
    python experiments/ablation_and_scale.py --experiment scale
    python experiments/ablation_and_scale.py --experiment wake_sleep
"""

import sys
import time
import argparse
import random

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


# ============================================================
# KB Generators
# ============================================================

def generate_scalable_kb(n_entities, n_relations=3, n_transitive=3,
                         n_body_pattern_groups=1):
    """Generate a KB of controllable size.

    Args:
        n_entities: number of entities (scales fact count)
        n_relations: number of binary relation types
        n_transitive: number of transitive closure predicates
        n_body_pattern_groups: number of groups sharing body patterns
    """
    kb = KnowledgeBase()
    entities = [f"e{i}" for i in range(n_entities)]

    # Type facts (guard for Op C)
    for e in entities:
        kb.add_fact(compound("entity", atom(e)))

    # Category: most are type_a (Op C target)
    for e in entities[:int(n_entities * 0.8)]:
        kb.add_fact(compound("category", atom(e), atom("type_a")))
    for e in entities[int(n_entities * 0.8):]:
        kb.add_fact(compound("category", atom(e), atom("type_b")))

    # Binary relations (chain structure for transitive closure)
    relation_names = [f"rel{i}" for i in range(n_relations)]
    for rel in relation_names:
        for i in range(n_entities - 1):
            kb.add_fact(compound(rel, atom(entities[i]), atom(entities[i + 1])))

    # Transitive closures with different bases (Op D target)
    tc_names = [(f"tc{i}", relation_names[i % len(relation_names)])
                for i in range(n_transitive)]
    for head, base in tc_names:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))

    # Rules with shared body prefix (Op E target)
    for g in range(n_body_pattern_groups):
        base_rel = relation_names[g % len(relation_names)]
        for i in range(3):
            suffix_rel = f"suffix_{g}_{i}"
            kb.add_fact(compound(suffix_rel, atom(entities[0]), atom(entities[1])))
            kb.add_rule(Rule(compound(f"combined_{g}_{i}", var("X"), var("W")),
                             [compound(base_rel, var("X"), var("Y")),
                              compound(base_rel, var("Y"), var("Z")),
                              compound(suffix_rel, var("Z"), var("W"))]))

    # Redundant facts (Op B targets)
    for head, base in tc_names[:2]:
        kb.add_fact(compound(head, atom(entities[0]), atom(entities[1])))

    return kb, entities


# ============================================================
# Experiment 1: Ablation Study
# ============================================================

def run_ablation():
    """Disable each operation individually, measure impact."""
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY")
    print(f"  Which operations matter most?")
    print(f"{'='*70}")

    kb_template, entities = generate_scalable_kb(
        n_entities=20, n_relations=3, n_transitive=4, n_body_pattern_groups=2)

    # Full run (baseline)
    kb = kb_template.copy()
    dreamer = KnowledgeBaseDreamer()
    session = dreamer.dream(kb, verify=True)
    baseline_clauses = len(kb)
    baseline_bg = sum(len(r.body) for r in kb.rules)
    baseline_ops = {}
    for op in session.operations:
        baseline_ops[op.operation] = baseline_ops.get(op.operation, 0) + 1

    initial_clauses = len(kb_template)
    print(f"\n  Initial KB: {initial_clauses} clauses")
    print(f"  Full dream: {initial_clauses} -> {baseline_clauses} "
          f"(ops: {dict(baseline_ops)})")

    # Disable each operation by monkey-patching
    operations = {
        "Op A (subsumption)": "_eliminate_subsumed",
        "Op B (pruning)": "_prune_redundant_facts",
        "Op C (generalization)": "_generalize_facts",
        "Op D (invention)": "_invent_predicates",
        "Op E (extraction)": "_extract_body_patterns",
        "Op F (dead clauses)": "_prune_dead_clauses",
        "Op H (lemmas)": "_cache_lemmas",
    }

    print(f"\n  {'Operation disabled':<25} {'Clauses':>8} {'Delta':>7} {'Impact':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*8}")

    for label, method_name in operations.items():
        kb = kb_template.copy()
        dreamer = KnowledgeBaseDreamer()

        # Monkey-patch: replace the method with a no-op
        original = getattr(dreamer, method_name)
        if method_name in ("_generalize_facts", "_invent_predicates",
                           "_extract_body_patterns", "_llm_compress"):
            setattr(dreamer, method_name, lambda kb, suite=None: [])
        else:
            setattr(dreamer, method_name, lambda kb, **kw: [])

        session = dreamer.dream(kb, verify=True)
        clauses = len(kb)
        delta = clauses - baseline_clauses
        impact = f"+{delta}" if delta > 0 else str(delta)
        pct = f"({delta/(initial_clauses-baseline_clauses)*100:+.0f}%)" if initial_clauses != baseline_clauses else ""
        print(f"  {label:<25} {clauses:>8} {impact:>7} {pct:>8}")


# ============================================================
# Experiment 2: Scaling
# ============================================================

def run_scale():
    """Measure how compression scales with KB size."""
    print(f"\n{'='*70}")
    print(f"  SCALING EXPERIMENT")
    print(f"  How does compression quality and time scale with KB size?")
    print(f"{'='*70}")

    sizes = [10, 20, 50, 100, 200]

    print(f"\n  {'N':>5} {'Initial':>8} {'After':>7} {'Ratio':>7} {'Ops':>5} "
          f"{'Time':>8} {'Rate':>10}")
    print(f"  {'-'*5} {'-'*8} {'-'*7} {'-'*7} {'-'*5} {'-'*8} {'-'*10}")

    for n in sizes:
        kb, _ = generate_scalable_kb(n_entities=n, n_relations=3,
                                      n_transitive=4, n_body_pattern_groups=2)
        initial = len(kb)

        t0 = time.perf_counter()
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)
        elapsed = (time.perf_counter() - t0) * 1000

        after = len(kb)
        ops = len(session.operations)
        ratio = after / initial if initial > 0 else 1.0
        rate = initial / (elapsed / 1000) if elapsed > 0 else 0

        print(f"  {n:>5} {initial:>8} {after:>7} {ratio:>7.2f} {ops:>5} "
              f"{elapsed:>7.0f}ms {rate:>8.0f} cl/s")


# ============================================================
# Experiment 3: Wake-Sleep Loop
# ============================================================

def run_wake_sleep():
    """Full wake-sleep loop: query -> dream -> query, measure improvement."""
    print(f"\n{'='*70}")
    print(f"  WAKE-SLEEP LOOP")
    print(f"  Does dreaming improve query performance?")
    print(f"{'='*70}")

    kb, entities = generate_scalable_kb(n_entities=30, n_relations=2,
                                         n_transitive=3)

    # Generate test queries
    test_queries = []
    for head, _ in [("tc0", "rel0"), ("tc1", "rel1"), ("tc2", "rel0")]:
        for i in range(0, min(10, len(entities) - 5), 2):
            test_queries.append(
                compound(head, atom(entities[i]), atom(entities[i + 3])))

    def benchmark_queries(kb, queries, label):
        ev = PrologEvaluator(kb)
        t0 = time.perf_counter()
        results = 0
        for q in queries:
            if ev.has_solution(q):
                results += 1
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed, results

    # Phase 1: Wake (query without dreaming)
    elapsed_before, results_before = benchmark_queries(kb, test_queries, "before")
    print(f"\n  Before dreaming:")
    print(f"    KB: {len(kb)} clauses")
    print(f"    Queries: {len(test_queries)}, {results_before} succeeded")
    print(f"    Time: {elapsed_before:.1f}ms")

    # Phase 2: Enable derivation tracking and re-query (builds up usage data)
    kb.enable_derivation_tracking()
    ev = PrologEvaluator(kb)
    for _ in range(5):
        for q in test_queries:
            list(ev.query([q]))

    # Phase 3: Dream
    dreamer = KnowledgeBaseDreamer()
    dream_t0 = time.perf_counter()
    session = dreamer.dream(kb, verify=True)
    dream_elapsed = (time.perf_counter() - dream_t0) * 1000

    ops = {}
    for op in session.operations:
        ops[op.operation] = ops.get(op.operation, 0) + 1

    print(f"\n  Dream cycle ({dream_elapsed:.0f}ms):")
    print(f"    KB: {len(kb)} clauses")
    print(f"    Operations: {dict(ops)}")

    # Phase 4: Re-query after dreaming
    elapsed_after, results_after = benchmark_queries(kb, test_queries, "after")
    print(f"\n  After dreaming:")
    print(f"    KB: {len(kb)} clauses")
    print(f"    Queries: {len(test_queries)}, {results_after} succeeded")
    print(f"    Time: {elapsed_after:.1f}ms")

    # Compare
    speedup = elapsed_before / elapsed_after if elapsed_after > 0 else float('inf')
    print(f"\n  Query speedup: {speedup:.2f}x")
    print(f"  Correctness preserved: {results_before == results_after}")

    # Phase 5: Second dream cycle
    session2 = dreamer.dream(kb, verify=True)
    print(f"  Second dream: {'converged' if not session2.compressed else 'more compression'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-e",
                        choices=["ablation", "scale", "wake_sleep", "all"],
                        default="all")
    args = parser.parse_args()

    if args.experiment in ("ablation", "all"):
        run_ablation()
    if args.experiment in ("scale", "all"):
        run_scale()
    if args.experiment in ("wake_sleep", "all"):
        run_wake_sleep()


if __name__ == "__main__":
    main()
