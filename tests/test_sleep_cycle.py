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


class TestOperationF:
    def test_dead_derived_fact_removed(self):
        """Derived fact (not seed) with 0 usage is pruned."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        # Simulate wake phase
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(15):
            list(ev.query([compound("a", atom("x"))]))
        # Now add a derived fact AFTER the wake phase (simulates prior dream)
        kb.add_fact(compound("b", atom("y")))
        # b(y) has 0 usage — it's derived, not seed
        dreamer = KnowledgeBaseDreamer()
        seed_terms = {compound("a", atom("x"))}  # only a(x) is seed
        ops = dreamer._prune_dead_clauses(kb, seed_terms=seed_terms)
        remaining = [f.term for f in kb.facts]
        assert compound("a", atom("x")) in remaining
        assert compound("b", atom("y")) not in remaining

    def test_seed_fact_protected(self):
        """Seed fact with 0 usage is NOT pruned."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))  # never queried, but is seed
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(15):
            list(ev.query([compound("a", atom("x"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        remaining = [f.term for f in kb.facts]
        assert compound("a", atom("x")) in remaining
        assert compound("b", atom("y")) in remaining  # protected as seed

    def test_dead_rule_removed(self):
        """Derived rule with 0 usage is pruned."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        r_used = Rule(compound("b", var("X")), [compound("a", var("X"))])
        kb.add_rule(r_used)
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(15):
            list(ev.query([compound("b", atom("x"))]))
        # Add a dead rule after wake (simulates prior dream output)
        r_dead = Rule(compound("c", var("X")), [compound("d", var("X"))])
        kb.add_rule(r_dead)
        dreamer = KnowledgeBaseDreamer()
        seed_rules = {(r_used.head, tuple(r_used.body))}
        ops = dreamer._prune_dead_clauses(kb, seed_rules=seed_rules)
        assert len(kb.rules) == 1  # dead rule removed, used rule kept

    def test_used_clauses_preserved(self):
        """Clauses with usage > 0 are kept."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(10):
            list(ev.query([compound("a", atom("x"))]))
            list(ev.query([compound("b", atom("y"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 2  # both used, both kept

    def test_threshold_prevents_premature_pruning(self):
        """Not enough queries -> skip dead clause pruning."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        list(ev.query([compound("a", atom("x"))]))  # only 1 query
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        assert len(kb.facts) == 2  # not enough data, keep both

    def test_generated_predicates_not_pruned(self):
        """_invented_, _extracted_, exception_ predicates skipped even if 0 usage."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        # Simulate generated predicates
        from dreamlog.terms import Compound, Atom, Variable
        kb.add_rule(Rule(Compound("_invented_0", [Variable("R"), Variable("X")]),
                         [Compound("call", [Variable("R"), Variable("X")])]))
        kb.add_fact(Fact(Compound("exception_likes_person", [Atom("dave")])))
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        for _ in range(15):
            list(ev.query([compound("a", atom("x"))]))
        dreamer = KnowledgeBaseDreamer()
        dreamer.dream(kb, verify=False)
        # Generated predicates should survive even with 0 usage
        inv_rules = [r for r in kb.rules
                     if isinstance(r.head, Compound)
                     and r.head.functor.startswith("_invented_")]
        assert len(inv_rules) == 1
        exc_facts = [f for f in kb.facts
                     if isinstance(f.term, Compound)
                     and f.term.functor.startswith("exception_")]
        assert len(exc_facts) == 1


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


class TestLLMNaming:
    def test_invented_predicate_renamed(self):
        """LLM suggests name for _invented_0, it gets renamed."""
        from tests.mock_provider import MockLLMProvider  # noqa: F811
        kb = KnowledgeBase()
        for head, base in [("ancestor","parent"),("reachable","edge"),("connected","link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        kb.add_fact(compound("parent", atom("a"), atom("b")))
        kb.add_fact(compound("edge", atom("x"), atom("y")))
        kb.add_fact(compound("link", atom("p"), atom("q")))

        mock = MockLLMProvider(responses=["transitive_closure"])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        session = dreamer.dream(kb, verify=True)

        # _invented_0 should be renamed to transitive_closure
        rule_functors = {r.head.functor for r in kb.rules if isinstance(r.head, Compound)}
        assert "transitive_closure" in rule_functors
        assert "_invented_0" not in rule_functors

    def test_no_llm_skips_naming(self):
        """Without LLM client, naming is skipped."""
        kb = KnowledgeBase()
        for head, base in [("ancestor","parent"),("reachable","edge"),("connected","link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        dreamer = KnowledgeBaseDreamer()  # no llm_client
        dreamer.dream(kb, verify=False)
        rule_functors = {r.head.functor for r in kb.rules if isinstance(r.head, Compound)}
        # Should have _invented_0, not a named version
        assert any(f.startswith("_invented_") for f in rule_functors)

    def test_bad_name_keeps_original(self):
        """If LLM suggests invalid name, keep _invented_N."""
        from tests.mock_provider import MockLLMProvider  # noqa: F811
        kb = KnowledgeBase()
        for head, base in [("ancestor","parent"),("reachable","edge"),("connected","link")]:
            kb.add_rule(Rule(compound(head, var("X"), var("Y")),
                             [compound(base, var("X"), var("Y"))]))
            kb.add_rule(Rule(compound(head, var("X"), var("Z")),
                             [compound(base, var("X"), var("Y")),
                              compound(head, var("Y"), var("Z"))]))
        # LLM returns something with spaces/special chars
        mock = MockLLMProvider(responses=["this is not a valid name!!!"])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        dreamer.dream(kb, verify=False)
        rule_functors = {r.head.functor for r in kb.rules if isinstance(r.head, Compound)}
        assert any(f.startswith("_invented_") for f in rule_functors)


class TestOperationG:
    def test_cross_functor_rule_proposed(self):
        """LLM proposes father(X,Y) :- parent(X,Y), male(X)."""
        from tests.mock_provider import MockLLMProvider
        kb = KnowledgeBase()
        for p, c in [("john","mary"),("bob","alice"),("carol","dave")]:
            kb.add_fact(compound("parent", atom(p), atom(c)))
        for n in ["john", "bob"]:
            kb.add_fact(compound("male", atom(n)))
        kb.add_fact(compound("father", atom("john"), atom("mary")))
        kb.add_fact(compound("father", atom("bob"), atom("alice")))

        import json
        rule_json = json.dumps([["rule", ["father", "X", "Y"],
                                [["parent", "X", "Y"], ["male", "X"]]]])
        mock = MockLLMProvider(responses=[rule_json])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        session = dreamer.dream(kb, verify=False)

        # father facts should be removed (derivable from new rule)
        father_facts = [f for f in kb.facts
                        if isinstance(f.term, Compound) and f.term.functor == "father"]
        father_rules = [r for r in kb.rules
                        if isinstance(r.head, Compound) and r.head.functor == "father"]
        # Should have the rule and fewer facts
        assert len(father_rules) >= 1 or len(father_facts) < 2

    def test_no_llm_skips_compression(self):
        """Without LLM client, Op G is skipped."""
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        dreamer = KnowledgeBaseDreamer()
        session = dreamer.dream(kb, verify=False)
        # No LLM operations
        assert all(op.operation != "llm_compression" for op in session.operations)

    def test_naf_rule_parsed(self):
        """LLM proposes helper + not/1 rule (e.g., vegan_recipe)."""
        from tests.mock_provider import MockLLMProvider
        import json
        kb = KnowledgeBase()
        kb.add_fact(compound("recipe", atom("stir_fry")))
        kb.add_fact(compound("recipe", atom("cake")))
        kb.add_fact(compound("uses", atom("cake"), atom("butter")))
        kb.add_fact(compound("uses", atom("stir_fry"), atom("tofu")))
        kb.add_fact(compound("vegan", atom("butter"), atom("false")))
        kb.add_fact(compound("vegan", atom("tofu"), atom("true")))
        kb.add_fact(compound("vegan_recipe", atom("stir_fry")))

        rules_json = json.dumps([
            ["rule", ["has_non_vegan", "X"],
             [["uses", "X", "Y"], ["vegan", "Y", "false"]]],
            ["rule", ["vegan_recipe", "X"],
             [["recipe", "X"], ["not", ["has_non_vegan", "X"]]]],
        ])
        mock = MockLLMProvider(responses=[rules_json])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        dreamer.dream(kb, verify=True)

        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        assert ev.has_solution(compound("vegan_recipe", atom("stir_fry")))
        assert not ev.has_solution(compound("vegan_recipe", atom("cake")))

    def test_cycle_filter_rejects_bidirectional(self):
        """Cycle filter rejects parent<-father when father<-parent+male exists."""
        from dreamlog.kb_dreamer import _filter_cyclic_rules
        r1 = Rule(compound("father", var("X"), var("Y")),
                  [compound("parent", var("X"), var("Y")),
                   compound("male", var("X"))])
        r2 = Rule(compound("parent", var("X"), var("Y")),
                  [compound("father", var("X"), var("Y"))])
        result = _filter_cyclic_rules([r1, r2])
        # r1 accepted, r2 rejected (creates cycle father->parent->father)
        assert len(result) == 1
        assert result[0].head.functor == "father"

    def test_cycle_filter_allows_self_recursion(self):
        """Cycle filter allows ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)."""
        from dreamlog.kb_dreamer import _filter_cyclic_rules
        r = Rule(compound("ancestor", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")),
                  compound("ancestor", var("Y"), var("Z"))])
        result = _filter_cyclic_rules([r])
        assert len(result) == 1

    def test_invalid_rule_rejected(self):
        """LLM proposes rule that over-generates -> rejected."""
        from tests.mock_provider import MockLLMProvider
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        kb.add_fact(compound("b", atom("y")))
        # LLM proposes a(X) :- b(X) which would make a(y) true (wrong)
        import json
        mock = MockLLMProvider(responses=[
            json.dumps([["rule", ["a", "X"], [["b", "X"]]]])])
        dreamer = KnowledgeBaseDreamer(llm_client=mock)
        dreamer.dream(kb, verify=True)
        # a(y) should NOT be derivable
        from dreamlog.evaluator import PrologEvaluator
        ev = PrologEvaluator(kb)
        assert not ev.has_solution(compound("a", atom("y")))


def _ancestor_closure_kb():
    kb = KnowledgeBase()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
        kb.add_fact(Fact(compound("parent", atom(x), atom(y))))
    # full ancestor closure of the 4-edge chain
    nodes = ["a", "b", "c", "d", "e"]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            kb.add_fact(Fact(compound("ancestor", atom(nodes[i]), atom(nodes[j]))))
    return kb


def test_dream_flag_off_does_not_discover_recursion():
    kb = _ancestor_closure_kb()
    dreamer = KnowledgeBaseDreamer(discover_recursion=False)
    dreamer.dream(kb)
    # No ancestor RULE was created (recursion off)
    assert not any(len(r.body) > 0 and r.head.functor == "ancestor"
                   for r in kb.rules)


def test_dream_flag_on_discovers_recursion_and_compresses():
    kb = _ancestor_closure_kb()
    n_before = len(kb)
    dreamer = KnowledgeBaseDreamer(discover_recursion=True, min_base_facts=3)
    session = dreamer.dream(kb)
    # ancestor is now defined by rules, the closure facts are pruned
    anc_rules = [r for r in kb.rules
                 if r.head.functor == "ancestor" and len(r.body) > 0]
    assert len(anc_rules) == 2
    assert not any(f.term.functor == "ancestor" for f in kb.facts)
    assert len(kb) < n_before
