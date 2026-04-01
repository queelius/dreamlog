#!/usr/bin/env python3
"""
EX16 + EX17: Prompt directionality variants + structural cycle filter.

Tests whether prompt changes eliminate bidirectional rules, and whether
a dependency-graph cycle filter catches any that slip through.

Usage:
    python experiments/ex16_prompt_and_filter.py
"""

import sys
import time
import json
from collections import defaultdict

sys.path.insert(0, ".")

from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import (
    KnowledgeBaseDreamer, build_verification_suite,
    extend_verification_for_rules, _strip_llm_noise, _collect_user_functors,
)
from dreamlog.llm_client import LLMClient


# ── KB builders ──────────────────────────────────────────────────────

def build_family_kb():
    kb = KnowledgeBase()
    for name in ["john", "mary", "bob", "alice", "carol", "dave", "eve", "frank"]:
        kb.add_fact(compound("person", atom(name)))
    for name in ["john", "bob", "dave", "frank"]:
        kb.add_fact(compound("male", atom(name)))
    for name in ["mary", "alice", "carol", "eve"]:
        kb.add_fact(compound("female", atom(name)))
    parents = [("john","bob"),("john","alice"),("mary","bob"),("mary","alice"),
               ("bob","carol"),("bob","dave"),("alice","eve"),("alice","frank")]
    for p, c in parents:
        kb.add_fact(compound("parent", atom(p), atom(c)))
    for p, c in parents:
        if p in ["john", "bob", "dave", "frank"]:
            kb.add_fact(compound("father", atom(p), atom(c)))
    for p, c in parents:
        if p in ["mary", "alice", "carol", "eve"]:
            kb.add_fact(compound("mother", atom(p), atom(c)))
    for head, base in [("ancestor","parent"),("descendant_of","parent")]:
        kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                         [compound(base, var("X"), var("Y"))]))
        kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                         [compound(base, var("X"), var("Y")),
                          compound(head, var("Y"), var("Z"))]))
    for a, b in [("bob","alice"),("alice","bob"),("carol","dave"),
                 ("dave","carol"),("eve","frank"),("frank","eve")]:
        kb.add_fact(compound("sibling", atom(a), atom(b)))
    return kb


def kb_to_fact_lines(kb, limit=50):
    lines = []
    for fact in kb.facts[:limit]:
        term = fact.term
        if isinstance(term, Compound):
            args = ", ".join(
                a.value if isinstance(a, Atom) else str(a) for a in term.args)
            lines.append(f"{term.functor}({args}).")
    return lines


# ── Prompt variants ──────────────────────────────────────────────────

def prompt_baseline(fact_lines):
    return (
        "Given these facts from a knowledge base:\n\n"
        + "\n".join(fact_lines)
        + "\n\nPropose rules that derive some facts from others. "
        "Use variables (X, Y, Z) for the general pattern. "
        "Use MULTIPLE rules for the same head when the pattern involves "
        "OR conditions (disjunction).\n\n"
        "Example format:\n"
        '  ["rule", ["father", "X", "Y"], [["parent", "X", "Y"], ["male", "X"]]]\n'
        '  ["rule", ["warm_blooded", "X"], [["class", "X", "mammal"]]]\n'
        '  ["rule", ["warm_blooded", "X"], [["class", "X", "bird"]]]\n\n'
        "Reply with ONLY a JSON array of rules. "
        "No explanation, no markdown.\n\n"
        "Rules:"
    )


def prompt_anti_bidirectional(fact_lines):
    return (
        "Given these facts from a knowledge base:\n\n"
        + "\n".join(fact_lines)
        + "\n\nPropose rules that derive SPECIFIC predicates from MORE GENERAL ones. "
        "A rule should EXPLAIN why a fact is true using simpler building blocks.\n\n"
        "IMPORTANT constraints:\n"
        "- Rules must go in ONE direction only: specific <- general.\n"
        "- NEVER propose reverse/inverse rules (if father <- parent+male, "
        "do NOT also propose parent <- father).\n"
        "- Body predicates should have MORE facts than the head predicate.\n"
        "- Each rule must derive at least 2 existing facts.\n\n"
        "Example format:\n"
        '  ["rule", ["father", "X", "Y"], [["parent", "X", "Y"], ["male", "X"]]]\n\n'
        "Reply with ONLY a JSON array of rules. "
        "No explanation, no markdown.\n\n"
        "Rules:"
    )


def prompt_hierarchy(fact_lines, predicate_counts):
    count_lines = "\n".join(
        f"  {fn}: {cnt} facts" for fn, cnt in
        sorted(predicate_counts.items(), key=lambda x: -x[1]))
    return (
        "Given these facts from a knowledge base:\n\n"
        + "\n".join(fact_lines)
        + f"\n\nPredicate fact counts (base predicates have more facts):\n{count_lines}\n\n"
        "Propose rules that derive predicates with FEWER facts from predicates "
        "with MORE facts. Rules should flow from general -> specific.\n\n"
        "NEVER propose rules that derive a base predicate (many facts) from "
        "a derived predicate (fewer facts). That reverses the information flow.\n\n"
        "Example format:\n"
        '  ["rule", ["father", "X", "Y"], [["parent", "X", "Y"], ["male", "X"]]]\n\n'
        "Reply with ONLY a JSON array of rules. "
        "No explanation, no markdown.\n\n"
        "Rules:"
    )


# ── Cycle detection filter (EX17) ───────────────────────────────────

def detect_cycles(rules):
    """Build a dependency graph from proposed rules and find cycles.

    Returns (clean_rules, rejected_rules) where rejected are those whose
    addition would create a cycle in the head->body functor graph.
    Self-recursive rules (head functor in own body) are allowed.
    """
    # Build graph incrementally, accepting rules that don't create cycles
    graph = defaultdict(set)  # head_functor -> set of body_functors
    clean = []
    rejected = []

    for rule in rules:
        head_fn = rule.head.functor
        body_fns = {g.functor for g in rule.body if isinstance(g, Compound)}
        # Self-recursion is fine (e.g., ancestor :- parent, ancestor)
        body_fns_no_self = body_fns - {head_fn}

        # Temporarily add edges and check for cycles
        old_edges = graph[head_fn].copy()
        graph[head_fn] |= body_fns_no_self

        if _has_cycle(graph):
            graph[head_fn] = old_edges  # rollback
            rejected.append(rule)
        else:
            clean.append(rule)

    return clean, rejected


def _has_cycle(graph):
    """DFS-based cycle detection on a directed graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)

    def dfs(node):
        color[node] = GRAY
        for neighbor in graph.get(node, set()):
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    for node in list(graph.keys()):
        if color[node] == WHITE:
            if dfs(node):
                return True
    return False


# ── Rule parsing helper ──────────────────────────────────────────────

def parse_rules_from_response(response):
    """Parse LLM response into Rule objects."""
    text = _strip_llm_noise(response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON array
        import re
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []

    rules = []
    for item in data:
        try:
            if not isinstance(item, list) or len(item) < 3:
                continue
            head_data = item[1]
            body_data = item[2]

            def make_term(d):
                fn = d[0]
                args = [Variable(a) if a[0].isupper() else Atom(a) for a in d[1:]]
                return Compound(fn, args)

            head = make_term(head_data)
            body = [make_term(b) for b in body_data]
            if body:
                rules.append(Rule(head, body))
        except Exception:
            continue
    return rules


# ── Analysis helpers ─────────────────────────────────────────────────

def classify_rules(rules):
    """Classify rules into forward, reverse pairs, and novel."""
    head_to_body = defaultdict(set)
    for rule in rules:
        head_fn = rule.head.functor
        body_fns = frozenset(g.functor for g in rule.body)
        head_to_body[head_fn].add(body_fns)

    bidirectional_pairs = []
    for fn_a, body_sets_a in head_to_body.items():
        for body_fns in body_sets_a:
            for fn_b in body_fns:
                if fn_b in head_to_body:
                    for body_fns_b in head_to_body[fn_b]:
                        if fn_a in body_fns_b:
                            bidirectional_pairs.append((fn_a, fn_b))

    return {
        "total": len(rules),
        "bidirectional_pairs": list(set(bidirectional_pairs)),
        "unique_heads": list(head_to_body.keys()),
    }


def test_rules_on_kb(rules, kb, max_calls=500):
    """Test how many rules derive 2+ facts and measure verification time."""
    kb_functors = _collect_user_functors(kb)
    accepted = []
    t0 = time.perf_counter()

    for rule in rules:
        body_fns = [g.functor for g in rule.body if isinstance(g, Compound)]
        if any(fn not in kb_functors for fn in body_fns):
            continue

        test_kb = kb.copy()
        test_kb.add_rule(rule)
        ev = PrologEvaluator(test_kb, max_total_calls=max_calls)

        derivable = 0
        try:
            for fact in kb.facts:
                if (isinstance(fact.term, Compound)
                        and fact.term.functor == rule.head.functor):
                    if ev.has_solution(fact.term):
                        derivable += 1
        except RecursionError:
            continue

        if derivable >= 2:
            accepted.append((rule, derivable))

    elapsed = time.perf_counter() - t0
    return accepted, elapsed


# ── Main experiment ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    client = LLMClient(
        provider=args.provider, model=args.model, api_key=args.api_key,
        api_key_env=args.api_key_env, base_url=args.base_url,
        temperature=0.3, max_tokens=800)

    print(f"Provider: {client.provider}, Model: {client.model}")
    test = client.complete("Reply with just 'ok'.")
    print(f"Connection: {_strip_llm_noise(test).strip()}")

    kb = build_family_kb()
    fact_lines = kb_to_fact_lines(kb)

    # Predicate fact counts for hierarchy prompt
    pred_counts = defaultdict(int)
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            pred_counts[fact.term.functor] += 1

    prompts = {
        "baseline": prompt_baseline(fact_lines),
        "anti_bidirectional": prompt_anti_bidirectional(fact_lines),
        "hierarchy": prompt_hierarchy(fact_lines, pred_counts),
    }

    print(f"\nKB: {len(kb)} clauses ({len(kb.facts)} facts)")
    print(f"Predicate counts: {dict(pred_counts)}")

    # ── EX16: Test each prompt variant ────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EX16: Prompt Variant Comparison")
    print(f"{'='*70}")

    results = {}
    for name, prompt in prompts.items():
        print(f"\n  --- {name} ---")
        t0 = time.perf_counter()
        response = client.complete(prompt)
        api_time = time.perf_counter() - t0

        rules = parse_rules_from_response(response)
        classification = classify_rules(rules)

        accepted, eval_time = test_rules_on_kb(rules, kb)

        results[name] = {
            "rules_proposed": len(rules),
            "rules_accepted": len(accepted),
            "bidirectional_pairs": classification["bidirectional_pairs"],
            "api_time": api_time,
            "eval_time": eval_time,
        }

        print(f"    Proposed: {len(rules)}, Accepted: {len(accepted)}")
        print(f"    Bidirectional pairs: {classification['bidirectional_pairs']}")
        print(f"    API: {api_time:.1f}s, Eval: {eval_time:.3f}s")

        for rule, deriv in accepted:
            print(f"      {rule}  (derives {deriv})")

    # ── EX17: Cycle filter on baseline results ────────────────────
    print(f"\n{'='*70}")
    print(f"  EX17: Structural Cycle Filter")
    print(f"{'='*70}")

    # Re-run baseline to get rules for filtering
    response = client.complete(prompts["baseline"])
    all_rules = parse_rules_from_response(response)
    print(f"\n  Baseline proposed {len(all_rules)} rules")

    clean, rejected = detect_cycles(all_rules)
    print(f"  After cycle filter: {len(clean)} clean, {len(rejected)} rejected")

    if rejected:
        print(f"  Rejected rules:")
        for r in rejected:
            print(f"    {r}")

    # Test the clean rules
    accepted_clean, eval_time_clean = test_rules_on_kb(clean, kb)
    print(f"  Clean rules accepted: {len(accepted_clean)}, eval: {eval_time_clean:.3f}s")
    for rule, deriv in accepted_clean:
        print(f"    {rule}  (derives {deriv})")

    # Full dream with clean rules vs all rules
    print(f"\n  --- Full dream comparison ---")

    # Dream with all rules (original behavior)
    kb_all = build_family_kb()
    dreamer_all = KnowledgeBaseDreamer(llm_client=client)
    t0 = time.perf_counter()
    session_all = dreamer_all.dream(kb_all, verify=True)
    time_all = time.perf_counter() - t0
    print(f"  All rules:    {42} -> {len(kb_all)} in {time_all:.1f}s "
          f"({len(session_all.operations)} ops)")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Variant':<25} {'Proposed':>8} {'Accepted':>9} {'Bidir':>6} {'Eval(s)':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*6} {'-'*8}")
    for name, r in results.items():
        n_bidir = len(r["bidirectional_pairs"])
        print(f"  {name:<25} {r['rules_proposed']:>8} {r['rules_accepted']:>9} "
              f"{n_bidir:>6} {r['eval_time']:>8.3f}")
    print(f"  {'baseline+cycle_filter':<25} {len(all_rules):>8} {len(accepted_clean):>9} "
          f"{'0':>6} {eval_time_clean:>8.3f}")

    # ── Recommendation ────────────────────────────────────────────
    best = min(results.items(),
               key=lambda x: (len(x[1]["bidirectional_pairs"]), -x[1]["rules_accepted"]))
    print(f"\n  Best prompt: {best[0]} "
          f"({best[1]['rules_accepted']} rules, "
          f"{len(best[1]['bidirectional_pairs'])} bidir pairs)")
    print(f"  Cycle filter: eliminates {len(rejected)} bidirectional rules from baseline")


if __name__ == "__main__":
    main()
