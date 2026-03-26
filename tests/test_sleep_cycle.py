# tests/test_sleep_cycle.py
import pytest
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.kb_dreamer import KnowledgeBaseDreamer, DreamSession


class TestUsageTracking:
    def test_record_and_get_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        assert kb.get_usage(f) == 0
        kb.record_usage(f)
        assert kb.get_usage(f) == 1
        kb.record_usage(f)
        assert kb.get_usage(f) == 2

    def test_get_usage_unknown_clause(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        assert kb.get_usage(f) == 0

    def test_reset_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.record_usage(f)
        kb.record_usage(f)
        assert kb.get_usage(f) == 2
        kb.reset_usage()
        assert kb.get_usage(f) == 0

    def test_total_queries_tracked(self):
        kb = KnowledgeBase()
        f1 = Fact(compound("a", atom("x")))
        f2 = Fact(compound("b", atom("y")))
        kb.add_fact(f1)
        kb.add_fact(f2)
        kb.record_usage(f1)
        kb.record_usage(f1)
        kb.record_usage(f2)
        assert kb.total_queries_tracked() == 3

    def test_copy_preserves_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.record_usage(f)
        kb.record_usage(f)
        copy = kb.copy()
        assert copy.get_usage(f) == 2

    def test_restore_from_restores_usage(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.record_usage(f)
        snapshot = kb.copy()
        kb.record_usage(f)
        kb.record_usage(f)
        assert kb.get_usage(f) == 3
        kb.restore_from(snapshot)
        assert kb.get_usage(f) == 1

    def test_rule_usage(self):
        kb = KnowledgeBase()
        r = Rule(compound("a", var("X")), [compound("b", var("X"))])
        kb.add_rule(r)
        kb.record_usage(r)
        assert kb.get_usage(r) == 1


class TestOperationA:
    def test_specific_rule_removed(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                         [compound("par", var("X"), var("Y"))]))
        kb.add_rule(Rule(compound("anc", atom("john"), var("Y")),
                         [compound("par", atom("john"), var("Y"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 1
        assert session.compressed is True

    def test_fact_subsumed_by_bodyless_rule(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("a", var("X")), []))
        kb.add_fact(compound("a", atom("hello")))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 0
        assert len(kb.rules) == 1

    def test_different_body_lengths_kept(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("a", var("X")),
                         [compound("b", var("X"))]))
        kb.add_rule(Rule(compound("a", var("X")),
                         [compound("b", var("X")), compound("c", var("X"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 2

    def test_rules_only_kb(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("a", var("X")), [compound("b", var("X"))]))
        kb.add_rule(Rule(compound("a", atom("x")), [compound("b", atom("x"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 1

    def test_empty_kb(self):
        kb = KnowledgeBase()
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb)
        assert session.compressed is False
        assert session.compression_ratio == 1.0


class TestOperationB:
    def test_derivable_fact_removed(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                         [compound("parent", var("X"), var("Y"))]))
        kb.add_fact(compound("anc", atom("john"), atom("mary")))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        anc_facts = [f for f in kb.facts
                     if hasattr(f.term, 'functor') and f.term.functor == "anc"]
        assert len(anc_facts) == 0

    def test_non_derivable_fact_kept(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 2

    def test_mutual_dependency_fallback(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("x")))
        kb.add_rule(Rule(compound("a", var("X")), [compound("b", var("X"))]))
        kb.add_rule(Rule(compound("b", var("X")), [compound("a", var("X"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) >= 1

    def test_facts_only_kb(self):
        kb = KnowledgeBase()
        for v in ["a", "b", "c"]:
            kb.add_fact(compound("x", atom(v)))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 3


class TestVerification:
    def test_positive_queries_built(self):
        from dreamlog.kb_dreamer import build_verification_suite
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        suite = build_verification_suite(kb)
        assert len(suite.positive_queries) >= 2

    def test_negative_queries_catch_overgeneration(self):
        from dreamlog.kb_dreamer import build_verification_suite
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("a", atom("y")))
        kb.add_fact(compound("b", atom("z")))
        suite = build_verification_suite(kb)
        result = suite.verify(kb, lambda k: PrologEvaluator(k))
        assert result.passed
        bad_kb = KnowledgeBase()
        bad_kb.add_rule(Rule(compound("a", var("X")), []))
        bad_kb.add_fact(compound("b", atom("z")))
        result = suite.verify(bad_kb, lambda k: PrologEvaluator(k))
        assert not result.passed

    def test_rollback_preserves_kb(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        original_size = len(kb)
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)
        assert len(kb) == original_size

    def test_rule_derived_queries_in_suite(self):
        """Verification suite includes queries derived from rules, not just facts."""
        from dreamlog.kb_dreamer import build_verification_suite, extend_verification_for_rules
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_rule(Rule(compound("anc", var("X"), var("Y")),
                         [compound("parent", var("X"), var("Y"))]))
        kb.add_rule(Rule(compound("anc", var("X"), var("Z")),
                         [compound("parent", var("X"), var("Y")),
                          compound("anc", var("Y"), var("Z"))]))
        suite = build_verification_suite(kb)
        extend_verification_for_rules(suite, kb)
        # Should have positive queries for derived facts like anc(john, mary)
        ev = PrologEvaluator(kb)
        assert any(ev.has_solution(q) for q in suite.positive_queries
                   if isinstance(q, Compound) and q.functor == "anc")


class TestOperationC:
    def test_basic_generalization(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        session = dreamer.dream(kb, verify=True)
        assert session.compressed is True
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for name in ["alice", "bob", "carol"]:
            assert ev.has_solution(compound("likes", atom(name), atom("chocolate")))
        assert not ev.has_solution(compound("likes", atom("dave"), atom("chocolate")))

    def test_mdl_rejects_small_group(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("person", atom(name)))
        kb.add_fact(compound("likes", atom("alice"), atom("chocolate")))
        kb.add_fact(compound("likes", atom("bob"), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        likes_facts = [f for f in kb.facts
                       if hasattr(f.term, 'functor') and f.term.functor == "likes"]
        assert len(likes_facts) == 2

    def test_no_shared_constants_skipped(self):
        """Facts with no shared constant in any position can't form subgroups."""
        kb = KnowledgeBase()
        # Each fact differs in BOTH positions, no constant pattern subgroup >= 3
        for a, b in [("a", "1"), ("b", "2"), ("c", "3")]:
            kb.add_fact(compound("f", atom(a), atom(b)))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 3

    def test_no_guard_found_skipped(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        likes_facts = [f for f in kb.facts
                       if hasattr(f.term, 'functor') and f.term.functor == "likes"]
        assert len(likes_facts) == 3

    def test_guard_selects_smallest_extension(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("small_group", atom(name)))
        for name in ["alice", "bob", "carol", "dave", "eve"]:
            kb.add_fact(compound("big_group", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=False)
        exc_facts = [f for f in kb.facts
                     if hasattr(f.term, 'functor')
                     and f.term.functor.startswith("exception_")]
        assert len(exc_facts) == 0

    def test_exception_predicates_excluded(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=True)
        size_after = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_after

    def test_idempotent(self):
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=True)
        size_after_first = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_after_first


class TestSubgroupDiscovery:
    """Tests for argument-position partitioning in Operation C."""

    def test_mixed_values_subgroup_found(self):
        """likes group with chocolate and vanilla: chocolate subgroup compresses."""
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        kb.add_fact(compound("likes", atom("eve"), atom("vanilla")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        session = dreamer.dream(kb, verify=False)
        assert session.compressed is True
        # The vanilla fact should still be there as-is
        vanilla_facts = [f for f in kb.facts
                         if hasattr(f.term, 'functor') and f.term.functor == "likes"]
        assert len(vanilla_facts) == 1
        assert vanilla_facts[0].term.args[1] == atom("vanilla")

    def test_mixed_values_correctness(self):
        """After subgroup compression, queries still work correctly."""
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        kb.add_fact(compound("likes", atom("eve"), atom("vanilla")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=True)
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for name in ["alice", "bob", "carol"]:
            assert ev.has_solution(compound("likes", atom(name), atom("chocolate")))
        assert not ev.has_solution(compound("likes", atom("dave"), atom("chocolate")))
        assert ev.has_solution(compound("likes", atom("eve"), atom("vanilla")))

    def test_higher_arity_subgroup(self):
        """3-arity facts with shared args in positions 0 and 2."""
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        # config(app, <key>, true) - varying position 1
        for key in ["debug", "verbose", "logging"]:
            kb.add_fact(compound("config", atom("app"), atom(key), atom("true")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        session = dreamer.dream(kb, verify=False)
        # Should find subgroup where position 1 varies, positions 0,2 constant
        config_facts = [f for f in kb.facts
                        if hasattr(f.term, 'functor') and f.term.functor == "config"]
        # Original 3 facts should be compressed (if guard found)
        # person/1 covers {alice, bob, carol, dave}, not {debug, verbose, logging}
        # No guard covers the config keys, so this should NOT compress
        assert len(config_facts) == 3

    def test_higher_arity_with_guard(self):
        """3-arity facts compress when a guard exists for the varying values."""
        kb = KnowledgeBase()
        for key in ["debug", "verbose", "logging", "color"]:
            kb.add_fact(compound("setting", atom(key)))
        for key in ["debug", "verbose", "logging"]:
            kb.add_fact(compound("config", atom("app"), atom(key), atom("true")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        session = dreamer.dream(kb, verify=False)
        config_facts = [f for f in kb.facts
                        if hasattr(f.term, 'functor') and f.term.functor == "config"]
        assert len(config_facts) == 0  # compressed into rule
        assert session.compressed is True


class TestOperationD:
    def _make_transitive_closure_kb(self, n_predicates=3):
        """Create KB with N structurally identical transitive closure predicates."""
        kb = KnowledgeBase()
        preds = [("ancestor", "parent"), ("reachable", "edge"), ("connected", "link")]
        for head, base in preds[:n_predicates]:
            kb.add_rule(Rule(
                compound(head, var("X"), var("Y")),
                [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(
                compound(head, var("X"), var("Z")),
                [compound(base, var("X"), var("Y")),
                 compound(head, var("Y"), var("Z"))]))
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_fact(compound("edge", atom("a"), atom("b")))
        kb.add_fact(compound("edge", atom("b"), atom("c")))
        kb.add_fact(compound("link", atom("x"), atom("y")))
        return kb

    def test_three_predicates_compressed(self):
        kb = self._make_transitive_closure_kb(3)
        assert len(kb.rules) == 6
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)
        assert len(kb.rules) == 5
        assert session.compressed is True

    def test_two_predicates_k2_rejected(self):
        kb = self._make_transitive_closure_kb(2)
        assert len(kb.rules) == 4
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 4

    def test_two_predicates_k3_accepted(self):
        kb = KnowledgeBase()
        for head, base in [("f", "g"), ("p", "q")]:
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("X"))]))
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("Y")),
                              compound(head, var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("X")),
                              compound(base, var("Y"))]))
        original_count = len(kb.rules)
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) < original_count

    def test_invented_excluded_from_future(self):
        kb = self._make_transitive_closure_kb(3)
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=True)
        size_after = len(kb.rules)
        dreamer.dream(kb, verify=True)
        assert len(kb.rules) == size_after

    def test_different_skeletons_not_grouped(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("f", var("X")),
                         [compound("g", var("X"))]))
        kb.add_rule(Rule(compound("h", var("X"), var("Y")),
                         [compound("j", var("X"), var("Y"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 2

    def test_single_rule_predicates_skipped(self):
        kb = KnowledgeBase()
        for head, base in [("f", "g"), ("h", "j"), ("k", "m")]:
            kb.add_rule(Rule(compound(head, var("X")),
                             [compound(base, var("X"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        assert len(kb.rules) == 3

    def test_derived_queries_still_work(self):
        kb = self._make_transitive_closure_kb(3)
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=True)
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("ancestor", atom("john"), atom("alice")))
        assert ev.has_solution(compound("reachable", atom("a"), atom("c")))
        assert ev.has_solution(compound("ancestor", atom("john"), atom("mary")))

    def test_facts_preserved(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("ancestor", atom("adam"), atom("eve")))
        for head, base in [("ancestor", "parent"), ("reachable", "edge"),
                           ("connected", "link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        anc_facts = [f for f in kb.facts
                     if isinstance(f.term, Compound) and f.term.functor == "ancestor"]
        assert len(anc_facts) == 1

    def test_idempotent(self):
        kb = self._make_transitive_closure_kb(3)
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=True)
        size_first = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_first


class TestOperationE:
    def test_three_rules_shared_prefix(self):
        """3 rules sharing a 2-goal prefix: extracted (K=2, N=3, savings=1)."""
        kb = KnowledgeBase()
        # grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z"))]))
        # great_gp(X, W) :- parent(X, Y), parent(Y, Z), parent(Z, W).
        kb.add_rule(Rule(compound("great_gp", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("parent", var("Z"), var("W"))]))
        # great_uncle(X, W) :- parent(X, Y), parent(Y, Z), brother(Z, W).
        kb.add_rule(Rule(compound("great_uncle", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("brother", var("Z"), var("W"))]))
        # Add facts for verification
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_fact(compound("parent", atom("alice"), atom("bob")))
        kb.add_fact(compound("brother", atom("alice"), atom("charlie")))

        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=True)

        # Should have extracted the common prefix
        extracted_rules = [r for r in kb.rules
                           if isinstance(r.head, Compound)
                           and r.head.functor.startswith("_extracted_")]
        assert len(extracted_rules) >= 1

        # Verify queries still work
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("grandparent", atom("john"), atom("alice")))
        assert ev.has_solution(compound("great_gp", atom("john"), atom("bob")))
        assert ev.has_solution(compound("great_uncle", atom("john"), atom("charlie")))

    def test_two_rules_k3_extracted(self):
        """2 rules sharing a 3-goal sub-sequence: extracted (K=3, N=2, savings=1)."""
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("f", var("X"), var("W")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z")),
                          compound("c", var("Z"), var("W"))]))
        kb.add_rule(Rule(compound("g", var("X"), var("W")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z")),
                          compound("c", var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) >= 1

    def test_two_rules_k2_rejected(self):
        """2 rules sharing a 2-goal sub-sequence: rejected (K=2, N=2, savings=0)."""
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("f", var("X"), var("Z")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z"))]))
        kb.add_rule(Rule(compound("g", var("X"), var("Z")),
                         [compound("a", var("X"), var("Y")),
                          compound("b", var("Y"), var("Z"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) == 0

    def test_interface_variables(self):
        """Extracted predicate exposes only interface vars, hides internal ones."""
        kb = KnowledgeBase()
        # f(X, W) :- a(X, Y), b(Y, Z), c(Z, W) -- Y and Z are internal to prefix a,b
        # g(X, W) :- a(X, Y), b(Y, Z), d(Z, W)
        # h(X, W) :- a(X, Y), b(Y, Z), e(Z, W)
        for head, tail_f in [("f", "c"), ("g", "d"), ("h", "e")]:
            kb.add_rule(Rule(compound(head, var("X"), var("W")),
                             [compound("a", var("X"), var("Y")),
                              compound("b", var("Y"), var("Z")),
                              compound(tail_f, var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) == 1
        # Interface should be (X, Z) -- X from head, Z connects to remaining body
        # Y is internal to the sub-sequence
        assert extracted[0].head.arity == 2

    def test_subsequence_at_end(self):
        """Sub-sequence at end of body (not just prefix) is detected."""
        kb = KnowledgeBase()
        # f(X, W) :- start(X, Y), common(Y, Z), common2(Z, W)
        # g(X, W) :- other(X, Y), common(Y, Z), common2(Z, W)
        # h(X, W) :- third(X, Y), common(Y, Z), common2(Z, W)
        for head, start_f in [("f", "start"), ("g", "other"), ("h", "third")]:
            kb.add_rule(Rule(compound(head, var("X"), var("W")),
                             [compound(start_f, var("X"), var("Y")),
                              compound("common", var("Y"), var("Z")),
                              compound("common2", var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        extracted = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_extracted_")]
        assert len(extracted) >= 1

    def test_generated_predicates_excluded(self):
        """_extracted_, _invented_, exception_ predicates are skipped."""
        kb = KnowledgeBase()
        for name in ["alice", "bob", "carol", "dave"]:
            kb.add_fact(compound("person", atom(name)))
        for name in ["alice", "bob", "carol"]:
            kb.add_fact(compound("likes", atom(name), atom("chocolate")))
        dreamer = KnowledgeBaseDreamer(min_group_size=3)
        dreamer.dream(kb, verify=True)
        size_after = len(kb)
        dreamer.dream(kb, verify=True)
        assert len(kb) == size_after

    def test_idempotent(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(compound("grandparent", var("X"), var("Z")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z"))]))
        kb.add_rule(Rule(compound("great_gp", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("parent", var("Z"), var("W"))]))
        kb.add_rule(Rule(compound("great_uncle", var("X"), var("W")),
                         [compound("parent", var("X"), var("Y")),
                          compound("parent", var("Y"), var("Z")),
                          compound("brother", var("Z"), var("W"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        size_first = len(kb)
        dreamer.dream(kb, verify=False)
        assert len(kb) == size_first


class TestEvaluatorUsageRecording:
    def test_fact_usage_recorded(self):
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        ev = PrologEvaluator(kb)
        list(ev.query([compound("parent", atom("john"), atom("mary"))]))
        f = kb.facts[0]
        assert kb.get_usage(f) >= 1

    def test_rule_usage_recorded(self):
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        r = Rule(compound("anc", var("X"), var("Y")),
                 [compound("parent", var("X"), var("Y"))])
        kb.add_rule(r)
        ev = PrologEvaluator(kb)
        list(ev.query([compound("anc", atom("john"), atom("mary"))]))
        assert kb.get_usage(r) >= 1

    def test_usage_accumulates_across_queries(self):
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        ev = PrologEvaluator(kb)
        list(ev.query([compound("a", atom("x"))]))
        list(ev.query([compound("a", atom("x"))]))
        list(ev.query([compound("a", atom("x"))]))
        assert kb.get_usage(kb.facts[0]) >= 3

    def test_unused_fact_has_zero_usage(self):
        from dreamlog.evaluator import PrologEvaluator
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        ev = PrologEvaluator(kb)
        list(ev.query([compound("a", atom("x"))]))
        assert kb.get_usage(kb.facts[0]) >= 1  # a(x) used
        assert kb.get_usage(kb.facts[1]) == 0   # b(y) never queried
