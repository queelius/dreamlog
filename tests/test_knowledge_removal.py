"""
Tests for fact and rule removal from KnowledgeBase
"""

import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Fact, Rule, KnowledgeBase


class TestFactRemoval:
    """Test removing facts from knowledge base"""

    def test_remove_fact_by_index(self):
        kb = KnowledgeBase()

        # Add some facts
        fact1 = Fact(compound("parent", atom("john"), atom("mary")))
        fact2 = Fact(compound("parent", atom("mary"), atom("alice")))
        fact3 = Fact(compound("parent", atom("alice"), atom("bob")))

        kb.add_fact(fact1)
        kb.add_fact(fact2)
        kb.add_fact(fact3)

        assert len(kb.facts) == 3

        # Remove middle fact
        removed = kb.remove_fact(1)

        assert removed == fact2
        assert len(kb.facts) == 2
        assert kb.facts[0] == fact1
        assert kb.facts[1] == fact3

    def test_remove_fact_index_out_of_range(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))

        # Try to remove invalid indices
        with pytest.raises(IndexError, match="out of range"):
            kb.remove_fact(5)

        with pytest.raises(IndexError, match="out of range"):
            kb.remove_fact(-1)

    def test_remove_fact_updates_index(self):
        """Test that removal updates the internal index correctly"""
        kb = KnowledgeBase()

        # Add facts
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("sibling", atom("alice"), atom("bob"))))
        kb.add_fact(Fact(compound("parent", atom("mary"), atom("alice"))))

        # Remove one parent fact
        kb.remove_fact(0)

        # Query should still work with remaining facts
        matching = list(kb.get_matching_facts(compound("parent", var("X"), var("Y"))))
        assert len(matching) == 1
        assert matching[0].term == compound("parent", atom("mary"), atom("alice"))

    def test_remove_all_facts(self):
        kb = KnowledgeBase()

        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("parent", atom("mary"), atom("alice"))))

        assert len(kb.facts) == 2

        # Remove all facts
        kb.remove_fact(1)
        kb.remove_fact(0)

        assert len(kb.facts) == 0

    def test_remove_fact_from_empty_kb(self):
        kb = KnowledgeBase()

        with pytest.raises(IndexError):
            kb.remove_fact(0)


class TestRuleRemoval:
    """Test removing rules from knowledge base"""

    def test_remove_rule_by_index(self):
        kb = KnowledgeBase()

        # Add some rules
        rule1 = Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        )
        rule2 = Rule(
            compound("sibling", var("X"), var("Y")),
            [compound("parent", var("Z"), var("X")), compound("parent", var("Z"), var("Y"))]
        )
        rule3 = Rule(
            compound("ancestor", var("X"), var("Y")),
            [compound("parent", var("X"), var("Y"))]
        )

        kb.add_rule(rule1)
        kb.add_rule(rule2)
        kb.add_rule(rule3)

        assert len(kb.rules) == 3

        # Remove middle rule
        removed = kb.remove_rule(1)

        assert removed == rule2
        assert len(kb.rules) == 2
        assert kb.rules[0] == rule1
        assert kb.rules[1] == rule3

    def test_remove_rule_index_out_of_range(self):
        kb = KnowledgeBase()
        kb.add_rule(Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        ))

        with pytest.raises(IndexError, match="out of range"):
            kb.remove_rule(5)

        with pytest.raises(IndexError, match="out of range"):
            kb.remove_rule(-1)

    def test_remove_rule_updates_index(self):
        """Test that removal updates the internal index correctly"""
        kb = KnowledgeBase()

        # Add rules with different head functors
        rule1 = Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        )
        rule2 = Rule(
            compound("sibling", var("X"), var("Y")),
            [compound("parent", var("Z"), var("X")), compound("parent", var("Z"), var("Y"))]
        )

        kb.add_rule(rule1)
        kb.add_rule(rule2)

        # Remove grandparent rule
        kb.remove_rule(0)

        # Query should still work with remaining rules
        matching = list(kb.get_matching_rules(compound("sibling", var("A"), var("B"))))
        assert len(matching) == 1
        assert matching[0] == rule2

    def test_remove_all_rules(self):
        kb = KnowledgeBase()

        rule1 = Rule(compound("test", var("X")), [compound("foo", var("X"))])
        rule2 = Rule(compound("test2", var("Y")), [compound("bar", var("Y"))])

        kb.add_rule(rule1)
        kb.add_rule(rule2)

        assert len(kb.rules) == 2

        kb.remove_rule(1)
        kb.remove_rule(0)

        assert len(kb.rules) == 0

    def test_remove_rule_from_empty_kb(self):
        kb = KnowledgeBase()

        with pytest.raises(IndexError):
            kb.remove_rule(0)


class TestMixedRemoval:
    """Test removing both facts and rules"""

    def test_remove_facts_and_rules(self):
        kb = KnowledgeBase()

        # Add facts
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("parent", atom("mary"), atom("alice"))))

        # Add rules
        kb.add_rule(Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        ))

        assert len(kb) == 3  # 2 facts + 1 rule

        # Remove a fact
        kb.remove_fact(0)
        assert len(kb.facts) == 1
        assert len(kb) == 2

        # Remove the rule
        kb.remove_rule(0)
        assert len(kb.rules) == 0
        assert len(kb) == 1

        # Remove remaining fact
        kb.remove_fact(0)
        assert len(kb) == 0

    def test_indices_are_separate(self):
        """Test that fact and rule indices are independent"""
        kb = KnowledgeBase()

        kb.add_fact(Fact(compound("fact1", atom("a"))))
        kb.add_fact(Fact(compound("fact2", atom("b"))))
        kb.add_rule(Rule(compound("rule1", var("X")), [compound("foo", var("X"))]))
        kb.add_rule(Rule(compound("rule2", var("Y")), [compound("bar", var("Y"))]))

        # Removing fact[0] should not affect rules
        kb.remove_fact(0)
        assert len(kb.facts) == 1
        assert len(kb.rules) == 2

        # Removing rule[0] should not affect facts
        kb.remove_rule(0)
        assert len(kb.facts) == 1
        assert len(kb.rules) == 1


class TestRemovalEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_remove_single_fact_multiple_times(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("test", atom("a"))))

        # Remove the only fact
        kb.remove_fact(0)
        assert len(kb.facts) == 0

        # Try to remove again - should fail
        with pytest.raises(IndexError):
            kb.remove_fact(0)

    def test_remove_and_readd_fact(self):
        kb = KnowledgeBase()

        fact = Fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(fact)

        # Remove it
        removed = kb.remove_fact(0)
        assert removed == fact
        assert len(kb.facts) == 0

        # Re-add it
        kb.add_fact(fact)
        assert len(kb.facts) == 1
        assert kb.facts[0] == fact

    def test_remove_updates_length(self):
        kb = KnowledgeBase()

        kb.add_fact(Fact(compound("a", atom("1"))))
        kb.add_fact(Fact(compound("b", atom("2"))))
        kb.add_rule(Rule(compound("c", var("X")), [compound("d", var("X"))]))

        initial_len = len(kb)
        assert initial_len == 3

        kb.remove_fact(0)
        assert len(kb) == initial_len - 1

        kb.remove_rule(0)
        assert len(kb) == initial_len - 2
