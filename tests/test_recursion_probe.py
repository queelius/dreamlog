# tests/test_recursion_probe.py
"""Feasibility probe for recursive rule discovery (go/no-go gate).

Confirms the evaluator terminates on a right-recursive transitive-closure
rule under the bounded evaluator, and that Operation B prunes the closure
facts once the recursive rule is present.
"""
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact, Rule, KnowledgeBase
from dreamlog.evaluator import PrologEvaluator
from dreamlog.kb_dreamer import KnowledgeBaseDreamer


def _chain_kb():
    """parent(a,b), parent(b,c), parent(c,d) + the ancestor closure."""
    kb = KnowledgeBase()
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    for x, y in edges:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    closure = [("a", "b"), ("a", "c"), ("a", "d"),
               ("b", "c"), ("b", "d"), ("c", "d")]
    for x, y in closure:
        kb.add_fact(Fact(compound("ancestor", atom(x), atom(y))))
    return kb


def _ancestor_rules():
    X, Y, Z = var("X"), var("Y"), var("Z")
    base = Rule(compound("ancestor", X, Y), [compound("parent", X, Y)])
    rec = Rule(compound("ancestor", X, Z),
               [compound("parent", X, Y), compound("ancestor", Y, Z)])
    return base, rec


def test_right_recursion_terminates_and_is_sound():
    kb = _chain_kb()
    # Remove the stored ancestor facts so derivation must use the rules
    for f in [f for f in kb.facts if f.term.functor == "ancestor"]:
        kb.remove_fact_by_value(f)
    base, rec = _ancestor_rules()
    kb.add_rule(base)
    kb.add_rule(rec)

    ev = PrologEvaluator(kb, max_total_calls=10000)
    # Reachable pair derivable, no RecursionError
    assert ev.has_solution(compound("ancestor", atom("a"), atom("d"))) is True
    # Non-reachable pair NOT derivable (sound, terminates)
    ev2 = PrologEvaluator(kb, max_total_calls=10000)
    assert ev2.has_solution(compound("ancestor", atom("d"), atom("a"))) is False


def test_op_b_prunes_closure_facts_under_recursion():
    kb = _chain_kb()
    base, rec = _ancestor_rules()
    kb.add_rule(base)
    kb.add_rule(rec)
    n_ancestor_before = sum(1 for f in kb.facts if f.term.functor == "ancestor")
    assert n_ancestor_before == 6

    dreamer = KnowledgeBaseDreamer()
    ops = dreamer._prune_redundant_facts(kb, max_calls=10000)

    n_ancestor_after = sum(1 for f in kb.facts if f.term.functor == "ancestor")
    assert n_ancestor_after == 0, "Op B should prune all ancestor facts"
    # Still derivable via the recursive rule
    ev = PrologEvaluator(kb, max_total_calls=10000)
    assert ev.has_solution(compound("ancestor", atom("a"), atom("d"))) is True
    assert len(ops) >= 1
