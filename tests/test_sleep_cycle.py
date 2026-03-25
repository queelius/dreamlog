# tests/test_sleep_cycle.py
import pytest
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.kb_dreamer import KnowledgeBaseDreamer, DreamSession


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
