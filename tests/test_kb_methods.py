# tests/test_kb_methods.py
import pytest
from dreamlog.factories import atom, compound
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Variable


class TestKBCopy:
    def test_copy_preserves_facts(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        copy = kb.copy()
        assert len(copy.facts) == 1
        assert copy.facts[0] == kb.facts[0]

    def test_copy_is_independent(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        copy = kb.copy()
        kb.add_fact(compound("b", atom("y")))
        assert len(copy.facts) == 1
        assert len(kb.facts) == 2


class TestKBRestoreFrom:
    def test_restore_reverts_changes(self):
        kb = KnowledgeBase()
        kb.add_fact(compound("a", atom("x")))
        snapshot = kb.copy()
        kb.add_fact(compound("b", atom("y")))
        kb.add_rule(Rule(compound("c", Variable("Z")), []))
        assert len(kb) == 3
        kb.restore_from(snapshot)
        assert len(kb) == 1


class TestKBRemoveByValue:
    def test_remove_fact_by_value(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        kb.add_fact(f)
        kb.remove_fact_by_value(f)
        assert len(kb.facts) == 0

    def test_remove_fact_not_found_raises(self):
        kb = KnowledgeBase()
        f = Fact(compound("a", atom("x")))
        with pytest.raises(ValueError):
            kb.remove_fact_by_value(f)

    def test_remove_rule_by_value(self):
        kb = KnowledgeBase()
        r = Rule(compound("a", Variable("X")),
                 [compound("b", Variable("X"))])
        kb.add_rule(r)
        kb.remove_rule_by_value(r)
        assert len(kb.rules) == 0


class TestKBReplaceFacts:
    def test_replace_facts_with_rule(self):
        kb = KnowledgeBase()
        f1 = Fact(compound("likes", atom("a"), atom("choc")))
        f2 = Fact(compound("likes", atom("b"), atom("choc")))
        kb.add_fact(f1)
        kb.add_fact(f2)
        new_rule = Rule(compound("likes", Variable("X"), atom("choc")), [])
        kb.replace_facts([f1, f2], [new_rule])
        assert len(kb.facts) == 0
        assert len(kb.rules) == 1
