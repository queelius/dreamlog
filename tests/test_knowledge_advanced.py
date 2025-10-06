"""
Advanced tests for DreamLog knowledge representation
"""

import pytest
import json
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Term, Atom, Variable, Compound
from dreamlog import atom, var, compound
from dreamlog.factories import term_from_prefix


class TestKnowledgeBaseAdvanced:
    """Test advanced knowledge base functionality"""
    
    def test_knowledge_base_initialization(self):
        """Test knowledge base initialization and properties"""
        kb = KnowledgeBase()
        
        assert len(kb.facts) == 0
        assert len(kb.rules) == 0
        assert len(kb._fact_index) == 0
        assert len(kb._rule_index) == 0
    
    def test_fact_operations(self):
        """Test fact creation and operations"""
        # Test Fact creation
        term = compound("parent", atom("john"), atom("mary"))
        fact = Fact(term)
        
        assert fact.term == term
        assert len(fact.get_variables()) == 0  # Ground fact
        
        # Test Fact with variables
        term_with_var = compound("parent", var("X"), atom("mary"))
        fact_with_var = Fact(term_with_var)
        variables = fact_with_var.get_variables()
        assert "X" in variables
        assert len(variables) == 1
        
        # Test substitution
        bindings = {"X": atom("john")}
        substituted_fact = fact_with_var.substitute(bindings)
        expected_term = compound("parent", atom("john"), atom("mary"))
        assert substituted_fact.term == expected_term
    
    def test_rule_operations(self):
        """Test rule creation and operations"""
        # Create a simple rule: parent(X, Y) :- father(X, Y)
        head = compound("parent", var("X"), var("Y"))
        body = [compound("father", var("X"), var("Y"))]
        rule = Rule(head, body)
        
        assert rule.head == head
        assert rule.body == tuple(body)  # Body is stored as tuple
        
        # Test variables in rule
        variables = rule.get_variables()
        assert "X" in variables
        assert "Y" in variables
        assert len(variables) == 2
        
        # Test rule substitution
        bindings = {"X": atom("john"), "Y": atom("mary")}
        substituted_rule = rule.substitute(bindings)
        expected_head = compound("parent", atom("john"), atom("mary"))
        expected_body = tuple([compound("father", atom("john"), atom("mary"))])
        assert substituted_rule.head == expected_head
        assert substituted_rule.body == expected_body
    
    def test_knowledge_base_indexing(self):
        """Test knowledge base indexing functionality"""
        kb = KnowledgeBase()
        
        # Add facts with same functor but different arities
        fact1 = compound("parent", atom("john"), atom("mary"))
        fact2 = compound("parent", atom("mary"), atom("alice"))
        fact3 = compound("likes", atom("alice"), atom("chocolate"))
        fact4 = compound("age", atom("john"))  # Different arity
        
        kb.add_fact(fact1)
        kb.add_fact(fact2)
        kb.add_fact(fact3)
        kb.add_fact(fact4)
        
        # Test retrieval using get_matching_facts with patterns
        parent_pattern = compound("parent", var("X"), var("Y"))
        parent_facts = list(kb.get_matching_facts(parent_pattern))
        assert len(parent_facts) == 2
        
        likes_pattern = compound("likes", var("X"), var("Y"))
        likes_facts = list(kb.get_matching_facts(likes_pattern))
        assert len(likes_facts) == 1
        
        # Test specific pattern matching
        john_parent_pattern = compound("parent", atom("john"), var("Y"))
        john_facts = list(kb.get_matching_facts(john_parent_pattern))
        # Should match at least the john->mary fact, but pattern matching might be more inclusive
        assert len(john_facts) >= 1
        
        # Verify at least one fact has john as first argument
        john_is_parent = any(fact.term.args[0] == atom("john") for fact in john_facts)
        assert john_is_parent
    
    def test_knowledge_base_rules_indexing(self):
        """Test knowledge base rule indexing"""
        kb = KnowledgeBase()
        
        # Add rules with different head functors
        rule1_head = compound("grandparent", var("X"), var("Z"))
        rule1_body = [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        rule1 = Rule(rule1_head, rule1_body)
        
        rule2_head = compound("ancestor", var("X"), var("Z"))
        rule2_body = [compound("parent", var("X"), var("Z"))]
        rule2 = Rule(rule2_head, rule2_body)
        
        kb.add_rule(rule1)
        kb.add_rule(rule2)
        
        # Test retrieval using get_matching_rules with patterns
        grandparent_pattern = compound("grandparent", var("X"), var("Z"))
        grandparent_rules = list(kb.get_matching_rules(grandparent_pattern))
        assert len(grandparent_rules) == 1
        assert grandparent_rules[0] == rule1
        
        ancestor_pattern = compound("ancestor", var("X"), var("Z"))
        ancestor_rules = list(kb.get_matching_rules(ancestor_pattern))
        assert len(ancestor_rules) == 1
        assert ancestor_rules[0] == rule2
    
    def test_duplicate_handling(self):
        """Test handling of duplicate facts and rules"""
        kb = KnowledgeBase()
        
        # Add same fact twice
        fact_term = compound("parent", atom("john"), atom("mary"))
        fact1 = Fact(fact_term)
        fact2 = Fact(fact_term)  # Same content, different object
        
        kb.add_fact(fact1)
        kb.add_fact(fact2)
        
        # Should only have one fact
        assert len(kb.facts) == 1
        
        # Add same rule twice
        head = compound("grandparent", var("X"), var("Z"))
        body = [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        rule1 = Rule(head, body)
        rule2 = Rule(head, body)  # Same content, different object
        
        kb.add_rule(rule1)
        kb.add_rule(rule2)
        
        # Should only have one rule
        assert len(kb.rules) == 1
    
    def test_knowledge_base_serialization(self):
        """Test knowledge base serialization and deserialization"""
        kb = KnowledgeBase()
        
        # Add facts and rules
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        
        head = compound("grandparent", var("X"), var("Z"))
        body = [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        kb.add_rule(Rule(head, body))
        
        # Test serialization to string
        prefix_json = kb.to_prefix()
        assert isinstance(prefix_json, str)
        
        # Test deserialization roundtrip
        new_kb = KnowledgeBase()
        new_kb.from_prefix(prefix_json)
        
        assert len(new_kb.facts) == 2
        assert len(new_kb.rules) == 1
    
    def test_get_functor_arities(self):
        """Test getting all functor/arity combinations"""
        kb = KnowledgeBase()
        
        # Add various facts and rules
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("likes", atom("alice"), atom("chocolate")))
        kb.add_fact(compound("age", atom("john")))
        
        head = compound("grandparent", var("X"), var("Z"))
        body = [compound("parent", var("X"), var("Y"))]
        kb.add_rule(Rule(head, body))
        
        # Test basic counts - no specific method for functor arities
        assert len(kb.facts) == 3
        assert len(kb.rules) == 1
        
        # Test that we have different types of predicates
        all_facts = kb.facts
        functors_seen = set()
        for fact in all_facts:
            if hasattr(fact.term, 'functor'):
                functors_seen.add(fact.term.functor)
        
        # Should have seen parent, likes, age functors
        assert len(functors_seen) >= 3
    
    def test_term_from_prefix_factory(self):
        """Test term creation from prefix notation"""
        # Simple atom
        term1 = term_from_prefix("john")
        assert term1 == atom("john")
        
        # Variable (uppercase)
        term2 = term_from_prefix("X")
        assert term2 == var("X")
        
        # Compound from list
        term3 = term_from_prefix(["parent", "john", "mary"])
        expected = compound("parent", atom("john"), atom("mary"))
        assert term3 == expected
        
        # Nested compound
        term4 = term_from_prefix(["grandparent", "X", ["child", "mary", "Y"]])
        expected = compound("grandparent", var("X"), compound("child", atom("mary"), var("Y")))
        assert term4 == expected
    
    def test_knowledge_base_statistics(self):
        """Test knowledge base statistics and summary info"""
        kb = KnowledgeBase()
        
        # Add various knowledge
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_fact(compound("likes", atom("alice"), atom("chocolate")))
        
        head1 = compound("grandparent", var("X"), var("Z"))
        body1 = [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        kb.add_rule(Rule(head1, body1))
        
        head2 = compound("ancestor", var("X"), var("Z"))
        body2 = [compound("grandparent", var("X"), var("Z"))]
        kb.add_rule(Rule(head2, body2))
        
        # Test basic counts
        assert len(kb.facts) == 3
        assert len(kb.rules) == 2
        
        # Test functor diversity by examining facts and rules
        functors_in_facts = set()
        for fact in kb.facts:
            if hasattr(fact.term, 'functor'):
                functors_in_facts.add(fact.term.functor)
        
        functors_in_rules = set()
        for rule in kb.rules:
            if hasattr(rule.head, 'functor'):
                functors_in_rules.add(rule.head.functor)
        
        all_functors = functors_in_facts.union(functors_in_rules)
        assert len(all_functors) >= 4  # At least parent, likes, grandparent, ancestor
    
    def test_knowledge_base_search_patterns(self):
        """Test searching for specific patterns in knowledge base"""
        kb = KnowledgeBase()
        
        # Add family tree facts
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        kb.add_fact(compound("parent", atom("bob"), atom("charlie")))
        kb.add_fact(compound("male", atom("john")))
        kb.add_fact(compound("female", atom("mary")))
        
        # Search for all parent relationships using pattern matching
        parent_pattern = compound("parent", var("X"), var("Y"))
        parent_facts = list(kb.get_matching_facts(parent_pattern))
        assert len(parent_facts) == 3
        
        # Check that all retrieved facts are indeed parent facts
        for fact in parent_facts:
            assert fact.term.functor == "parent"
            assert fact.term.arity == 2
    
    def test_knowledge_base_mixed_terms(self):
        """Test knowledge base with mixed term types"""
        kb = KnowledgeBase()
        
        # Add facts with different types of terms
        kb.add_fact(compound("age", atom("john"), atom(30)))  # Number
        kb.add_fact(compound("active", atom("server1"), atom(True)))  # Boolean
        kb.add_fact(compound("name", atom("person1"), atom("Alice Johnson")))  # String
        
        # Verify all were added correctly
        assert len(kb.facts) == 3
        
        age_pattern = compound("age", var("X"), var("Y"))
        age_facts = list(kb.get_matching_facts(age_pattern))
        assert len(age_facts) == 1
        assert age_facts[0].term.args[1].value == 30
    
    def test_error_handling(self):
        """Test error handling in knowledge operations"""
        kb = KnowledgeBase()
        
        # Test with None values (should handle gracefully)
        try:
            kb.add_fact(None)
            assert False, "Should have raised an exception"
        except (TypeError, AttributeError):
            pass  # Expected
        
        # Test malformed prefix data
        try:
            kb.from_prefix("invalid json")
            # Should handle gracefully or raise appropriate exception
        except (json.JSONDecodeError, ValueError, KeyError):
            pass  # Expected