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

from dreamlog.factories import atom, compound
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


def test_op_i_rejects_near_miss_closure():
    # `step` has 3 facts (>= min_base_facts), so its base is NOT gated out.
    # `path` equals the transitive closure of `step` MINUS one reachable pair,
    # a true near-miss. It must be rejected at the math gate (r_ext != TC(b_ext)),
    # which the `likes` case above never reaches because of the size gate.
    kb = KnowledgeBase()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d")]:
        kb.add_fact(Fact(compound("step", atom(x), atom(y))))
    full_closure = [("a", "b"), ("a", "c"), ("a", "d"),
                    ("b", "c"), ("b", "d"), ("c", "d")]
    for x, y in [p for p in full_closure if p != ("a", "d")]:  # closure minus one
        kb.add_fact(Fact(compound("path", atom(x), atom(y))))
    suite = build_verification_suite(kb)
    dreamer = KnowledgeBaseDreamer(min_base_facts=3)

    ops = dreamer._discover_recursion(kb, suite)
    assert ops == []


# ── EX27 domain generator tests ───────────────────────────────────────────────

import importlib.util, pathlib


def _load_ex27():
    path = pathlib.Path(__file__).parent.parent / "experiments" / "ex27_recursion.py"
    spec = importlib.util.spec_from_file_location("ex27_recursion", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_fact_pair(s: str):
    """Extract (arg1, arg2) from a fact string like '(flux_links a b)' or 'flux_links a b'."""
    tokens = s.strip("() ").split()
    return (tokens[1], tokens[2])


def test_flux_domain_reaches_is_closure_of_links():
    ex27 = _load_ex27()
    base, derived = ex27.flux_domain(n_nodes=8, seed=42)
    links = {_parse_fact_pair(s) for s in base}
    reaches = {_parse_fact_pair(s) for s in derived}
    from dreamlog.recursive_discovery import transitive_closure
    assert reaches == transitive_closure(links)
    assert len(links) >= 3
