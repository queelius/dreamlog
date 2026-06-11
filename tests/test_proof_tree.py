# tests/test_proof_tree.py
"""
Behavior-focused tests for dreamlog/proof_tree.py and the
query_with_proof machinery in dreamlog/evaluator.py.

Covers: ProofNode, ProofLog, and PrologEvaluator.query_with_proof().
No LLM/network calls; purely symbolic.
"""

import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.evaluator import PrologEvaluator
from dreamlog.proof_tree import ProofNode, ProofLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_family_kb():
    """Return a KB with parent facts and a grandparent rule."""
    kb = KnowledgeBase()
    # parent(john, mary), parent(mary, sue)
    kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
    kb.add_fact(Fact(compound("parent", atom("mary"), atom("sue"))))
    # grandparent(X, Z) :- parent(X, Y), parent(Y, Z)
    gp_rule = Rule(
        compound("grandparent", var("X"), var("Z")),
        [compound("parent", var("X"), var("Y")),
         compound("parent", var("Y"), var("Z"))],
    )
    kb.add_rule(gp_rule)
    return kb


def _make_ancestor_kb():
    """Return a KB with parent facts and two-rule ancestor definition."""
    kb = KnowledgeBase()
    # facts: a -> b -> c -> d
    for a, b in [("a", "b"), ("b", "c"), ("c", "d")]:
        kb.add_fact(Fact(compound("parent", atom(a), atom(b))))
    # ancestor(X, Y) :- parent(X, Y).
    kb.add_rule(Rule(
        compound("ancestor", var("X"), var("Y")),
        [compound("parent", var("X"), var("Y"))],
    ))
    # ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
    kb.add_rule(Rule(
        compound("ancestor", var("X"), var("Z")),
        [compound("parent", var("X"), var("Y")),
         compound("ancestor", var("Y"), var("Z"))],
    ))
    return kb


def _make_multi_solution_kb():
    """Return a KB where color(X) has three ground solutions."""
    kb = KnowledgeBase()
    for c in ("red", "green", "blue"):
        kb.add_fact(Fact(compound("color", atom(c))))
    return kb


# ---------------------------------------------------------------------------
# ProofNode unit tests
# ---------------------------------------------------------------------------

class TestProofNodeBasic:
    def test_leaf_node_subtree_size_is_one(self):
        node = ProofNode(goal=atom("foo"), depth=0)
        assert node.subtree_size() == 1

    def test_node_with_children_subtree_size(self):
        child1 = ProofNode(goal=atom("a"), depth=1)
        child2 = ProofNode(goal=atom("b"), depth=1)
        root = ProofNode(goal=atom("root"), children=[child1, child2], depth=0)
        assert root.subtree_size() == 3

    def test_nested_subtree_size(self):
        leaf = ProofNode(goal=atom("leaf"), depth=2)
        mid = ProofNode(goal=atom("mid"), children=[leaf], depth=1)
        root = ProofNode(goal=atom("root"), children=[mid], depth=0)
        assert root.subtree_size() == 3

    def test_clause_sequence_leaf(self):
        f = Fact(compound("foo", atom("x")))
        node = ProofNode(goal=compound("foo", atom("x")), clause=f, depth=0)
        seq = node.clause_sequence()
        assert seq == [f]

    def test_clause_sequence_with_children(self):
        f1 = Fact(compound("a", atom("x")))
        f2 = Fact(compound("b", atom("y")))
        child = ProofNode(goal=compound("b", atom("y")), clause=f2, depth=1)
        root = ProofNode(goal=compound("a", atom("x")), clause=f1,
                         children=[child], depth=0)
        seq = root.clause_sequence()
        assert seq == [f1, f2]

    def test_structural_key_leaf_vs_node(self):
        leaf = ProofNode(goal=compound("p", atom("x")), depth=0)
        key = leaf.structural_key()
        assert key[0] == "leaf"

    def test_structural_key_node_has_node_prefix(self):
        f = Fact(compound("p", atom("x")))
        child = ProofNode(goal=compound("q", atom("y")), depth=1)
        root = ProofNode(goal=compound("p", atom("x")), clause=f,
                         children=[child], depth=0)
        key = root.structural_key()
        assert key[0] == "node"

    def test_repr_leaf(self):
        node = ProofNode(goal=atom("foo"), depth=0)
        text = repr(node)
        assert "ProofNode" in text

    def test_repr_with_children(self):
        child = ProofNode(goal=atom("child"), depth=1)
        root = ProofNode(goal=atom("root"), children=[child], depth=0)
        text = repr(root)
        assert "<-" in text


# ---------------------------------------------------------------------------
# Fact-level proof via query_with_proof
# ---------------------------------------------------------------------------

class TestFactProof:
    def test_ground_fact_returns_one_pair(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("likes", atom("alice"), atom("pizza"))))
        ev = PrologEvaluator(kb)
        goal = compound("likes", atom("alice"), atom("pizza"))
        pairs = list(ev.query_with_proof([goal]))
        assert len(pairs) == 1

    def test_fact_proof_solution_is_empty_bindings(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("likes", atom("alice"), atom("pizza"))))
        ev = PrologEvaluator(kb)
        goal = compound("likes", atom("alice"), atom("pizza"))
        (sol, node), = ev.query_with_proof([goal])
        # ground query - no free variables to bind
        assert isinstance(sol.bindings, dict)

    def test_fact_proof_node_uses_fact_clause(self):
        kb = KnowledgeBase()
        f = Fact(compound("likes", atom("alice"), atom("pizza")))
        kb.add_fact(f)
        ev = PrologEvaluator(kb)
        goal = compound("likes", atom("alice"), atom("pizza"))
        (sol, node), = ev.query_with_proof([goal])
        assert isinstance(node.clause, Fact)

    def test_fact_proof_node_has_no_children(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("foo", atom("a"))))
        ev = PrologEvaluator(kb)
        goal = compound("foo", atom("a"))
        (sol, node), = ev.query_with_proof([goal])
        assert node.children == []

    def test_fact_proof_node_depth_is_zero(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("foo", atom("a"))))
        ev = PrologEvaluator(kb)
        goal = compound("foo", atom("a"))
        (sol, node), = ev.query_with_proof([goal])
        assert node.depth == 0

    def test_fact_proof_subtree_size_is_one(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("foo", atom("a"))))
        ev = PrologEvaluator(kb)
        goal = compound("foo", atom("a"))
        (sol, node), = ev.query_with_proof([goal])
        assert node.subtree_size() == 1


# ---------------------------------------------------------------------------
# Rule-derived proof
# ---------------------------------------------------------------------------

class TestRuleProof:
    def test_grandparent_yields_one_solution(self):
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        pairs = list(ev.query_with_proof([goal]))
        assert len(pairs) == 1

    def test_grandparent_proof_uses_rule_clause(self):
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        (sol, node), = ev.query_with_proof([goal])
        assert isinstance(node.clause, Rule)

    def test_grandparent_proof_has_two_children(self):
        """Grandparent rule has two body goals -> two child proof nodes."""
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        (sol, node), = ev.query_with_proof([goal])
        # Each body goal produces a child node
        assert len(node.children) == 2

    def test_grandparent_children_are_fact_nodes(self):
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        (sol, node), = ev.query_with_proof([goal])
        for child in node.children:
            assert isinstance(child.clause, Fact)

    def test_grandparent_child_depth_incremented(self):
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        (sol, node), = ev.query_with_proof([goal])
        assert node.depth == 0
        for child in node.children:
            assert child.depth > 0

    def test_grandparent_subtree_size_is_three(self):
        """root (rule) + 2 children (facts) = 3 nodes."""
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        (sol, node), = ev.query_with_proof([goal])
        assert node.subtree_size() == 3

    def test_rule_proof_clause_sequence_length(self):
        """clause_sequence should include rule + both fact clauses."""
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        (sol, node), = ev.query_with_proof([goal])
        seq = node.clause_sequence()
        # 1 rule + 2 facts = 3 clauses
        assert len(seq) == 3


# ---------------------------------------------------------------------------
# Recursive (ancestor) proof
# ---------------------------------------------------------------------------

class TestRecursiveProof:
    def test_direct_parent_proof_uses_base_rule(self):
        """ancestor(a, b) via parent(a, b) - uses the base rule."""
        kb = _make_ancestor_kb()
        ev = PrologEvaluator(kb)
        goal = compound("ancestor", atom("a"), atom("b"))
        pairs = list(ev.query_with_proof([goal]))
        assert len(pairs) >= 1
        # At least one proof must exist
        _, node = pairs[0]
        assert isinstance(node.clause, Rule)

    def test_transitive_ancestor_proof_has_nested_depth(self):
        """ancestor(a, c) requires a -> b -> c; proof tree must be deeper than 1."""
        kb = _make_ancestor_kb()
        ev = PrologEvaluator(kb)
        goal = compound("ancestor", atom("a"), atom("c"))
        pairs = list(ev.query_with_proof([goal]))
        assert len(pairs) >= 1
        # Find the proof with greatest depth (the recursive one)
        max_depth = 0
        for _, node in pairs:
            # DFS to find max depth in tree
            def _max_depth(n):
                if not n.children:
                    return n.depth
                return max(n.depth, max(_max_depth(c) for c in n.children))
            max_depth = max(max_depth, _max_depth(node))
        # Two-hop query: depth must exceed 1
        assert max_depth >= 2

    def test_three_hop_ancestor_proof_deeper_than_two(self):
        """ancestor(a, d) via a->b->c->d; proof tree must have depth >= 3."""
        kb = _make_ancestor_kb()
        ev = PrologEvaluator(kb)
        goal = compound("ancestor", atom("a"), atom("d"))
        pairs = list(ev.query_with_proof([goal]))
        assert len(pairs) >= 1
        max_depth = 0
        for _, node in pairs:
            def _max_depth(n):
                if not n.children:
                    return n.depth
                return max(n.depth, max(_max_depth(c) for c in n.children))
            max_depth = max(max_depth, _max_depth(node))
        assert max_depth >= 3

    def test_ancestor_solutions_consistent_with_plain_query(self):
        """Solutions from query_with_proof must match plain query()."""
        kb = _make_ancestor_kb()
        ev_plain = PrologEvaluator(kb)
        ev_proof = PrologEvaluator(kb)
        goal = compound("ancestor", atom("a"), var("Z"))
        plain_sols = list(ev_plain.query([goal]))
        proof_pairs = list(ev_proof.query_with_proof([goal]))
        plain_z = sorted(
            s.bindings.get("Z", var("Z")).substitute(s.bindings).__repr__()
            for s in plain_sols
        )
        proof_z = sorted(
            s.bindings.get("Z", var("Z")).substitute(s.bindings).__repr__()
            for s, _ in proof_pairs
        )
        assert plain_z == proof_z


# ---------------------------------------------------------------------------
# Multiple solutions
# ---------------------------------------------------------------------------

class TestMultipleSolutions:
    def test_three_color_solutions_yields_three_proofs(self):
        kb = _make_multi_solution_kb()
        ev = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        pairs = list(ev.query_with_proof([goal]))
        assert len(pairs) == 3

    def test_each_color_solution_has_distinct_binding(self):
        kb = _make_multi_solution_kb()
        ev = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        pairs = list(ev.query_with_proof([goal]))
        bound_values = set()
        for sol, _ in pairs:
            val = sol.bindings.get("X")
            assert val is not None
            bound_values.add(repr(val))
        assert len(bound_values) == 3

    def test_proof_solutions_match_plain_query_solutions(self):
        kb = _make_multi_solution_kb()
        ev_plain = PrologEvaluator(kb)
        ev_proof = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        plain = sorted(
            repr(s.bindings["X"]) for s in ev_plain.query([goal])
        )
        proof = sorted(
            repr(s.bindings["X"]) for s, _ in ev_proof.query_with_proof([goal])
        )
        assert plain == proof

    def test_each_proof_has_its_own_fact_clause(self):
        kb = _make_multi_solution_kb()
        ev = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        pairs = list(ev.query_with_proof([goal]))
        for sol, node in pairs:
            assert isinstance(node.clause, Fact)

    def test_no_solutions_for_unknown_predicate(self):
        kb = KnowledgeBase()
        ev = PrologEvaluator(kb)
        goal = compound("unknown", atom("x"))
        pairs = list(ev.query_with_proof([goal]))
        assert pairs == []


# ---------------------------------------------------------------------------
# ProofLog
# ---------------------------------------------------------------------------

class TestProofLog:
    def test_empty_log_proof_count_is_zero(self):
        log = ProofLog()
        assert log.proof_count == 0

    def test_add_proof_increments_count(self):
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        assert log.proof_count == 1

    def test_add_multiple_proofs_count_matches(self):
        kb = _make_multi_solution_kb()
        ev = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        assert log.proof_count == 3

    def test_clear_resets_count(self):
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        assert log.proof_count == 1
        log.clear()
        assert log.proof_count == 0

    def test_get_common_subtrees_returns_list(self):
        """get_common_subtrees() must return a list (possibly empty)."""
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        result = log.get_common_subtrees(min_count=1, min_depth=1)
        assert isinstance(result, list)

    def test_get_common_subtrees_tuples_have_count(self):
        """Each entry from get_common_subtrees is (key, count) with count >= min_count."""
        kb = _make_multi_solution_kb()
        ev = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        # All three color proofs share the same structural key (fact with color/1)
        result = log.get_common_subtrees(min_count=2, min_depth=1)
        assert len(result) >= 1
        for key, count in result:
            assert count >= 2

    def test_get_common_subtrees_sorted_descending(self):
        """Results must be sorted by count descending."""
        kb = _make_multi_solution_kb()
        ev = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        result = log.get_common_subtrees(min_count=1, min_depth=1)
        counts = [c for _, c in result]
        assert counts == sorted(counts, reverse=True)

    def test_find_proof_nodes_matching_returns_list(self):
        """find_proof_nodes_matching always returns a list."""
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        log = ProofLog()
        nodes = []
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
            nodes.append(node)
        # look up the key of the root node itself
        key = nodes[0].structural_key()
        matches = log.find_proof_nodes_matching(key)
        assert isinstance(matches, list)
        assert len(matches) >= 1

    def test_find_proof_nodes_matching_finds_root(self):
        """Root node's own structural key should be found."""
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        log = ProofLog()
        root_nodes = []
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
            root_nodes.append(node)
        key = root_nodes[0].structural_key()
        matches = log.find_proof_nodes_matching(key)
        assert root_nodes[0] in matches

    def test_find_proof_nodes_no_match_returns_empty(self):
        """A bogus structural key produces an empty list, not an error."""
        kb = _make_family_kb()
        ev = PrologEvaluator(kb)
        goal = compound("grandparent", atom("john"), atom("sue"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        bogus_key = ("node", ("no_such", 99), ("fact", ("no_such", 99)), ())
        matches = log.find_proof_nodes_matching(bogus_key)
        assert matches == []

    def test_clear_removes_subtree_counts(self):
        """After clear(), get_common_subtrees should return empty."""
        kb = _make_multi_solution_kb()
        ev = PrologEvaluator(kb)
        goal = compound("color", var("X"))
        log = ProofLog()
        for sol, node in ev.query_with_proof([goal]):
            log.add_proof(goal, node)
        log.clear()
        result = log.get_common_subtrees(min_count=1, min_depth=1)
        assert result == []


# ---------------------------------------------------------------------------
# Negation-as-failure via query_with_proof
# ---------------------------------------------------------------------------

class TestNafProof:
    """query_with_proof handles not/1 via a simplified path (non-proof branch
    in _solve_goals_with_proof).  The contract: if NAF succeeds, exactly one
    (Solution, ProofNode) pair is returned and the node wraps the not/1 goal."""

    def test_naf_succeeds_yields_one_pair(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("bird", atom("tweety"))))
        # naf_test(X) :- bird(X), not(penguin(X))
        kb.add_rule(Rule(
            compound("naf_test", var("X")),
            [compound("bird", var("X")), compound("not", compound("penguin", var("X")))],
        ))
        ev = PrologEvaluator(kb)
        goal = compound("naf_test", atom("tweety"))
        pairs = list(ev.query_with_proof([goal]))
        # tweety is a bird and not a penguin -> should succeed
        assert len(pairs) >= 1

    def test_naf_fails_yields_no_pairs(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("bird", atom("tweety"))))
        kb.add_fact(Fact(compound("penguin", atom("tweety"))))
        kb.add_rule(Rule(
            compound("naf_test", var("X")),
            [compound("bird", var("X")), compound("not", compound("penguin", var("X")))],
        ))
        ev = PrologEvaluator(kb)
        goal = compound("naf_test", atom("tweety"))
        pairs = list(ev.query_with_proof([goal]))
        # tweety is a penguin so naf_test should fail
        assert len(pairs) == 0
