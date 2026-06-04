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
