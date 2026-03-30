#!/usr/bin/env python3
"""
Experiment #3: DreamCoder Benchmark Comparison

DreamCoder discovers library primitives (map, fold, filter) by compressing
lambda calculus programs. Can DreamLog's sleep cycle discover equivalent
abstractions from logic programs?

Method:
- Encode DreamCoder's list domain tasks as DreamLog rules
- Lists represented as cons(Head, Tail) / nil
- Each "task" is a set of rules defining a list operation
- Run the sleep cycle and see what abstractions emerge
- Compare with DreamCoder's discovered primitives

DreamCoder's key discoveries in the list domain:
- map: applying a function to each element
- fold: reducing a list with an accumulator
- filter: selecting elements matching a predicate
- Common recursive patterns (base case + recursive step on cons)

Usage:
    python experiments/dreamcoder_benchmark.py
    python experiments/dreamcoder_benchmark.py --with-llm --model qwen3:14b
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


def build_list_kb():
    """Build a KB with multiple list-processing rule sets.

    Each operation follows the same recursive pattern:
      base case: op(nil, ...) :- ...
      recursive: op(cons(X, Xs), ...) :- ..., op(Xs, ...).

    DreamCoder would discover that these share the recursive traversal pattern
    and extract it as a library primitive (map, fold, etc).
    """
    kb = KnowledgeBase()

    # Helper: build list terms
    def make_list(*items):
        result = atom("nil")
        for item in reversed(items):
            result = compound("cons", atom(str(item)), result)
        return result

    # ================================================================
    # List facts for testing
    # ================================================================
    # Some small lists as facts for verification
    list_123 = make_list("1", "2", "3")
    list_456 = make_list("4", "5", "6")
    list_12 = make_list("1", "2")

    # ================================================================
    # Task 1: member/2 - check if element is in list
    # member(X, cons(X, _)).
    # member(X, cons(_, Xs)) :- member(X, Xs).
    # ================================================================
    kb.add_rule(Rule(
        compound("member", var("X"), compound("cons", var("X"), var("_Tail"))),
        []))  # base case: X is the head
    kb.add_rule(Rule(
        compound("member", var("X"), compound("cons", var("_Head"), var("Xs"))),
        [compound("member", var("X"), var("Xs"))]))

    # ================================================================
    # Task 2: append/3 - concatenate two lists
    # append(nil, Ys, Ys).
    # append(cons(X, Xs), Ys, cons(X, Zs)) :- append(Xs, Ys, Zs).
    # ================================================================
    kb.add_rule(Rule(
        compound("append", atom("nil"), var("Ys"), var("Ys")),
        []))
    kb.add_rule(Rule(
        compound("append", compound("cons", var("X"), var("Xs")), var("Ys"),
                 compound("cons", var("X"), var("Zs"))),
        [compound("append", var("Xs"), var("Ys"), var("Zs"))]))

    # ================================================================
    # Task 3: reverse/2 - reverse a list
    # reverse(nil, nil).
    # reverse(cons(X, Xs), Reversed) :- reverse(Xs, RevTail), append(RevTail, cons(X, nil), Reversed).
    # ================================================================
    kb.add_rule(Rule(
        compound("reverse", atom("nil"), atom("nil")),
        []))
    kb.add_rule(Rule(
        compound("reverse", compound("cons", var("X"), var("Xs")), var("Reversed")),
        [compound("reverse", var("Xs"), var("RevTail")),
         compound("append", var("RevTail"), compound("cons", var("X"), atom("nil")),
                  var("Reversed"))]))

    # ================================================================
    # Task 4: length/2 - count elements
    # length(nil, 0).
    # length(cons(_, Xs), s(N)) :- length(Xs, N).
    # (using successor notation: s(s(s(0))) = 3)
    # ================================================================
    kb.add_rule(Rule(
        compound("length", atom("nil"), atom("0")),
        []))
    kb.add_rule(Rule(
        compound("length", compound("cons", var("_Head"), var("Xs")),
                 compound("s", var("N"))),
        [compound("length", var("Xs"), var("N"))]))

    # ================================================================
    # Task 5: last/2 - get last element
    # last(cons(X, nil), X).
    # last(cons(_, Xs), X) :- last(Xs, X).
    # ================================================================
    kb.add_rule(Rule(
        compound("last", compound("cons", var("X"), atom("nil")), var("X")),
        []))
    kb.add_rule(Rule(
        compound("last", compound("cons", var("_Head"), var("Xs")), var("X")),
        [compound("last", var("Xs"), var("X"))]))

    # ================================================================
    # Task 6: sum_list/2 - sum elements (successor arithmetic)
    # sum_list(nil, 0).
    # sum_list(cons(X, Xs), Total) :- sum_list(Xs, SubTotal), add(X, SubTotal, Total).
    # ================================================================
    kb.add_rule(Rule(
        compound("sum_list", atom("nil"), atom("0")),
        []))
    kb.add_rule(Rule(
        compound("sum_list", compound("cons", var("X"), var("Xs")), var("Total")),
        [compound("sum_list", var("Xs"), var("SubTotal")),
         compound("add", var("X"), var("SubTotal"), var("Total"))]))

    # ================================================================
    # Task 7: map_succ/2 - increment each element
    # map_succ(nil, nil).
    # map_succ(cons(X, Xs), cons(s(X), Ys)) :- map_succ(Xs, Ys).
    # ================================================================
    kb.add_rule(Rule(
        compound("map_succ", atom("nil"), atom("nil")),
        []))
    kb.add_rule(Rule(
        compound("map_succ", compound("cons", var("X"), var("Xs")),
                 compound("cons", compound("s", var("X")), var("Ys"))),
        [compound("map_succ", var("Xs"), var("Ys"))]))

    # ================================================================
    # Task 8: map_double/2 - double each element
    # map_double(nil, nil).
    # map_double(cons(X, Xs), cons(Y, Ys)) :- double(X, Y), map_double(Xs, Ys).
    # ================================================================
    kb.add_rule(Rule(
        compound("map_double", atom("nil"), atom("nil")),
        []))
    kb.add_rule(Rule(
        compound("map_double", compound("cons", var("X"), var("Xs")),
                 compound("cons", var("Y"), var("Ys"))),
        [compound("double", var("X"), var("Y")),
         compound("map_double", var("Xs"), var("Ys"))]))

    # ================================================================
    # Task 9: all_positive/1 - check if all elements are positive
    # all_positive(nil).
    # all_positive(cons(X, Xs)) :- positive(X), all_positive(Xs).
    # ================================================================
    kb.add_rule(Rule(
        compound("all_positive", atom("nil")),
        []))
    kb.add_rule(Rule(
        compound("all_positive", compound("cons", var("X"), var("Xs"))),
        [compound("positive", var("X")),
         compound("all_positive", var("Xs"))]))

    # ================================================================
    # Task 10: all_even/1 - check if all elements are even
    # all_even(nil).
    # all_even(cons(X, Xs)) :- even(X), all_even(Xs).
    # ================================================================
    kb.add_rule(Rule(
        compound("all_even", atom("nil")),
        []))
    kb.add_rule(Rule(
        compound("all_even", compound("cons", var("X"), var("Xs"))),
        [compound("even", var("X")),
         compound("all_even", var("Xs"))]))

    # ================================================================
    # Task 11: all_nonzero/1
    # all_nonzero(nil).
    # all_nonzero(cons(X, Xs)) :- nonzero(X), all_nonzero(Xs).
    # ================================================================
    kb.add_rule(Rule(
        compound("all_nonzero", atom("nil")),
        []))
    kb.add_rule(Rule(
        compound("all_nonzero", compound("cons", var("X"), var("Xs"))),
        [compound("nonzero", var("X")),
         compound("all_nonzero", var("Xs"))]))

    # ================================================================
    # Supporting facts for testing
    # ================================================================
    # add/3 facts for small numbers
    kb.add_fact(compound("add", atom("0"), atom("0"), atom("0")))
    kb.add_fact(compound("add", atom("1"), atom("0"), atom("1")))
    kb.add_fact(compound("add", atom("0"), atom("1"), atom("1")))
    kb.add_fact(compound("add", atom("1"), atom("1"), atom("2")))

    # double/2
    kb.add_fact(compound("double", atom("1"), atom("2")))
    kb.add_fact(compound("double", atom("2"), atom("4")))

    # positive/1, even/1, nonzero/1
    for n in ["1", "2", "3"]:
        kb.add_fact(compound("positive", atom(n)))
        kb.add_fact(compound("nonzero", atom(n)))
    for n in ["2", "4"]:
        kb.add_fact(compound("even", atom(n)))

    return kb


def analyze_abstractions(kb):
    """Analyze what abstractions the sleep cycle discovered."""
    discovered = {
        "invented": [],
        "extracted": [],
        "generalized": [],
    }

    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            f = rule.head.functor
            if f.startswith("_invented_") or (not f.startswith("_") and
                    f not in ("member","append","reverse","length","last",
                              "sum_list","map_succ","map_double",
                              "all_positive","all_even","all_nonzero")):
                discovered["invented"].append(rule)
            elif f.startswith("_extracted_"):
                discovered["extracted"].append(rule)

    return discovered


def run_benchmark(llm_client=None):
    kb = build_list_kb()

    initial_rules = len(kb.rules)
    initial_facts = len(kb.facts)
    initial_total = initial_rules + initial_facts
    initial_body_goals = sum(len(r.body) for r in kb.rules)

    print(f"{'='*70}")
    print(f"  DREAMCODER LIST DOMAIN BENCHMARK")
    print(f"{'='*70}")
    print(f"\n  Tasks: 11 list operations (member, append, reverse, length,")
    print(f"         last, sum_list, map_succ, map_double, all_positive,")
    print(f"         all_even, all_nonzero)")
    print(f"  Initial: {initial_total} clauses ({initial_facts} facts, "
          f"{initial_rules} rules, {initial_body_goals} body goals)")

    # Show the rule patterns
    print(f"\n  Rule patterns (DreamCoder would find these):")
    print(f"    - 'all_X' pattern: 3 predicates (all_positive, all_even, all_nonzero)")
    print(f"      all share: SELF(nil). SELF(cons(X,Xs)) :- check(X), SELF(Xs).")
    print(f"    - 'map_X' pattern: 2 predicates (map_succ, map_double)")
    print(f"      both traverse cons and transform head")
    print(f"    - Recursive list traversal: shared across most tasks")

    # Run dream cycle
    dreamer = KnowledgeBaseDreamer(llm_client=llm_client)
    t0 = time.perf_counter()
    session = dreamer.dream(kb, verify=True)
    elapsed = (time.perf_counter() - t0) * 1000

    final_rules = len(kb.rules)
    final_facts = len(kb.facts)
    final_total = final_rules + final_facts
    final_body_goals = sum(len(r.body) for r in kb.rules)

    print(f"\n  After dreaming ({elapsed:.0f}ms):")
    print(f"    {initial_total} -> {final_total} clauses "
          f"({final_facts} facts, {final_rules} rules, {final_body_goals} body goals)")

    # Operation breakdown
    op_summary = {}
    for op in session.operations:
        op_summary[op.operation] = op_summary.get(op.operation, 0) + 1
    if op_summary:
        print(f"    Operations: {', '.join(f'{k}:{v}' for k,v in op_summary.items())}")

    # Analyze abstractions
    discovered = analyze_abstractions(kb)

    print(f"\n  Abstractions discovered:")
    if discovered["invented"]:
        print(f"    Invented predicates ({len(discovered['invented'])} rules):")
        for r in discovered["invented"]:
            print(f"      {r}")
    if discovered["extracted"]:
        print(f"    Extracted patterns ({len(discovered['extracted'])} rules):")
        for r in discovered["extracted"]:
            print(f"      {r}")
    if not any(discovered.values()):
        print(f"    None")

    # Compare with DreamCoder expectations
    print(f"\n  DreamCoder comparison:")

    # Did it find the 'all_X' pattern? (3 predicates with same skeleton)
    all_check_functors = set()
    for r in kb.rules:
        if isinstance(r.head, Compound) and r.head.functor in ("all_positive","all_even","all_nonzero"):
            all_check_functors.add(r.head.functor)
    # If they were compressed, they'd be wrappers around an invented predicate
    invented_functors = {r.head.functor for r in kb.rules
                        if isinstance(r.head, Compound)
                        and (r.head.functor.startswith("_invented_") or
                             r.head.functor not in ("member","append","reverse","length",
                                                     "last","sum_list","map_succ","map_double",
                                                     "all_positive","all_even","all_nonzero") and
                             not r.head.functor.startswith("_extracted_"))}

    if invented_functors:
        print(f"    'all_X' pattern: FOUND (invented: {invented_functors})")
        print(f"    This is analogous to DreamCoder's 'forall' or 'filter' primitive")
    else:
        print(f"    'all_X' pattern: NOT FOUND")
        print(f"    (all_positive, all_even, all_nonzero have same skeleton but")
        print(f"     Op D needs param_count==1; these have 1 varying body functor")
        print(f"     so it should work if the skeleton matches correctly)")

    # Show final rules for analysis
    print(f"\n  Final rules:")
    for r in kb.rules:
        print(f"    {r}")

    # Correctness checks
    print(f"\n  Correctness:")
    ev = PrologEvaluator(kb)

    checks = [
        ("member(2, cons(1,cons(2,cons(3,nil))))",
         compound("member", atom("2"), compound("cons", atom("1"),
                  compound("cons", atom("2"), compound("cons", atom("3"), atom("nil"))))),
         True),
        ("length(cons(1,cons(2,nil)), s(s(0)))",
         compound("length", compound("cons", atom("1"), compound("cons", atom("2"), atom("nil"))),
                  compound("s", compound("s", atom("0")))),
         True),
        ("last(cons(1,cons(2,cons(3,nil))), 3)",
         compound("last", compound("cons", atom("1"),
                  compound("cons", atom("2"), compound("cons", atom("3"), atom("nil")))),
                  atom("3")),
         True),
        ("all_positive(cons(1,cons(2,nil)))",
         compound("all_positive", compound("cons", atom("1"), compound("cons", atom("2"), atom("nil")))),
         True),
    ]

    all_pass = True
    for label, query, expected in checks:
        result = ev.has_solution(query)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"    {status}: {label}")

    return {
        "initial": initial_total,
        "final": final_total,
        "body_goals_initial": initial_body_goals,
        "body_goals_final": final_body_goals,
        "operations": op_summary,
        "invented": len(discovered["invented"]),
        "extracted": len(discovered["extracted"]),
        "correct": all_pass,
    }


def main():
    parser = argparse.ArgumentParser(description="DreamCoder Benchmark")
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--base-url", default="http://192.168.0.225:11434/v1")
    parser.add_argument("--model", default="qwen3:14b")
    args = parser.parse_args()

    llm_client = None
    if args.with_llm:
        from dreamlog.llm_client import LLMClient
        llm_client = LLMClient(
            base_url=args.base_url, model=args.model,
            temperature=0.3, max_tokens=1000)
        print(f"LLM: {args.model}")

    run_benchmark(llm_client=llm_client)


if __name__ == "__main__":
    main()
