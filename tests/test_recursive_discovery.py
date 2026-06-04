# tests/test_recursive_discovery.py
from dreamlog.recursive_discovery import transitive_closure


def test_tc_on_a_chain():
    edges = {("a", "b"), ("b", "c"), ("c", "d")}
    assert transitive_closure(edges) == {
        ("a", "b"), ("a", "c"), ("a", "d"),
        ("b", "c"), ("b", "d"), ("c", "d"),
    }


def test_tc_on_a_branch():
    edges = {("r", "x"), ("r", "y"), ("x", "z")}
    assert transitive_closure(edges) == {
        ("r", "x"), ("r", "y"), ("x", "z"), ("r", "z"),
    }


def test_tc_empty():
    assert transitive_closure(set()) == set()


def test_tc_is_irreflexive_on_a_dag():
    edges = {("a", "b"), ("b", "c")}
    tc = transitive_closure(edges)
    assert not any(x == y for x, y in tc)


# ── Operation I tests ──────────────────────────────────────────────────────

from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact, KnowledgeBase
from dreamlog.kb_dreamer import KnowledgeBaseDreamer, build_verification_suite


def _closure_kb():
    kb = KnowledgeBase()
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    for x, y in edges:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    for x, y in [("a", "b"), ("a", "c"), ("a", "d"),
                 ("b", "c"), ("b", "d"), ("c", "d")]:
        kb.add_fact(Fact(compound("ancestor", atom(x), atom(y))))
    return kb


def test_op_i_discovers_ancestor_closure():
    kb = _closure_kb()
    suite = build_verification_suite(kb)
    dreamer = KnowledgeBaseDreamer(min_base_facts=3)

    ops = dreamer._discover_recursion(kb, suite)

    assert len(ops) == 1
    assert ops[0].operation == "recursion"
    # The 6 ancestor facts were replaced by 2 rules
    assert ops[0].mdl_delta == 2 - 6
    # Two ancestor rules are now in the KB, no ancestor facts remain
    anc_rules = [r for r in kb.rules if r.head.functor == "ancestor"]
    assert len(anc_rules) == 2
    assert not any(f.term.functor == "ancestor" for f in kb.facts)


def test_op_i_no_false_discovery_when_not_a_closure():
    kb = KnowledgeBase()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d")]:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    # "likes" is unrelated, not a closure of parent
    for x, y in [("a", "c"), ("d", "a")]:
        kb.add_fact(Fact(compound("likes", atom(x), atom(y))))
    suite = build_verification_suite(kb)
    dreamer = KnowledgeBaseDreamer(min_base_facts=3)

    ops = dreamer._discover_recursion(kb, suite)
    assert ops == []
