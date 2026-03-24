# tests/test_sleep_cycle.py
import pytest
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
