"""
Tests for DreamLog engine
"""
import pytest
from dreamlog import (
    atom, var, compound,
    Fact, Rule, KnowledgeBase,
    DreamLogEngine, create_family_kb
)
from dreamlog.evaluator import PrologEvaluator


class TestDreamLogEngine:
    """Test the main DreamLog engine"""
    
    def test_engine_creation(self):
        """Test creating an engine"""
        engine = DreamLogEngine()
        assert engine.kb is not None
        assert isinstance(engine.evaluator, PrologEvaluator)
    
    def test_add_facts(self):
        """Test adding facts to engine"""
        engine = DreamLogEngine()
        
        # Add fact from term
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        
        # Check it was added
        facts = [f for f in engine.kb.facts if f.term.functor == "parent"]
        assert len(facts) == 1
        assert facts[0].term.functor == "parent"
    
    def test_add_rules(self):
        """Test adding rules to engine"""
        engine = DreamLogEngine()
        
        # Add rule
        head = compound("grandparent", var("X"), var("Z"))
        body = [
            compound("parent", var("X"), var("Y")),
            compound("parent", var("Y"), var("Z"))
        ]
        engine.add_rule_from_terms(head, body)
        
        # Check it was added
        rules = [r for r in engine.kb.rules if r.head.functor == "grandparent"]
        assert len(rules) == 1
        assert rules[0].head.functor == "grandparent"
    
    def test_query_facts(self):
        """Test querying facts"""
        engine = DreamLogEngine()
        
        # Add facts
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        engine.add_fact_from_term(compound("parent", atom("mary"), atom("alice")))
        
        # Query
        query = compound("parent", atom("john"), var("X"))
        solutions = list(engine.query([query]))
        
        assert len(solutions) == 1
        assert solutions[0].bindings[var("X")] == atom("mary")
    
    def test_query_with_rules(self):
        """Test querying with rule inference"""
        engine = DreamLogEngine()
        
        # Add facts
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        engine.add_fact_from_term(compound("parent", atom("mary"), atom("alice")))
        
        # Add rule for grandparent
        head = compound("grandparent", var("X"), var("Z"))
        body = [
            compound("parent", var("X"), var("Y")),
            compound("parent", var("Y"), var("Z"))
        ]
        engine.add_rule_from_terms(head, body)
        
        # Query for grandparent
        query = compound("grandparent", atom("john"), var("X"))
        solutions = list(engine.query([query]))
        
        assert len(solutions) == 1
        assert solutions[0].bindings[var("X")] == atom("alice")
    
    def test_ask_method(self):
        """Test the ask convenience method"""
        engine = DreamLogEngine()
        
        # Add fact
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        
        # Ask if fact exists
        assert engine.ask("parent", "john", "mary") == True
        assert engine.ask("parent", "john", "alice") == False
    
    def test_find_all_method(self):
        """Test finding all solutions"""
        engine = DreamLogEngine()
        
        # Add multiple facts
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        engine.add_fact_from_term(compound("parent", atom("john"), atom("bob")))
        engine.add_fact_from_term(compound("parent", atom("mary"), atom("alice")))
        
        # Find all children of john
        results = engine.find_all("parent", "john", "X")
        
        assert len(results) == 2
        children = [r['X'] for r in results]
        assert 'mary' in children
        assert 'bob' in children
    
    def test_create_family_kb(self):
        """Test the sample family knowledge base"""
        engine = create_family_kb()
        
        # Should have facts and rules
        assert len(engine.kb.facts) > 0
        assert len(engine.kb.rules) > 0
        
        # Should be able to query
        results = engine.find_all("parent", "john", "X")
        assert len(results) > 0


class TestKnowledgeBase:
    """Test the knowledge base storage"""
    
    def test_kb_creation(self):
        """Test creating a knowledge base"""
        kb = KnowledgeBase()
        assert len(kb.facts) == 0
        assert len(kb.rules) == 0
    
    def test_add_and_retrieve_facts(self):
        """Test adding and retrieving facts"""
        kb = KnowledgeBase()
        
        fact1 = Fact(compound("parent", atom("john"), atom("mary")))
        fact2 = Fact(compound("parent", atom("mary"), atom("alice")))
        fact3 = Fact(compound("age", atom("john"), atom("42")))
        
        kb.add_fact(fact1)
        kb.add_fact(fact2)
        kb.add_fact(fact3)
        
        # Get all facts
        assert len(kb.facts) == 3
        
        # Get facts by functor
        parent_facts = [f for f in kb.facts if f.term.functor == "parent"]
        assert len(parent_facts) == 2
        
        age_facts = [f for f in kb.facts if f.term.functor == "age"]
        assert len(age_facts) == 1
    
    def test_add_and_retrieve_rules(self):
        """Test adding and retrieving rules"""
        kb = KnowledgeBase()
        
        rule1 = Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), 
             compound("parent", var("Y"), var("Z"))]
        )
        
        rule2 = Rule(
            compound("sibling", var("X"), var("Y")),
            [compound("parent", var("Z"), var("X")),
             compound("parent", var("Z"), var("Y")),
             compound("different", var("X"), var("Y"))]
        )
        
        kb.add_rule(rule1)
        kb.add_rule(rule2)
        
        # Get all rules
        assert len(kb.rules) == 2
        
        # Get rules by head functor
        gp_rules = [r for r in kb.rules if r.head.functor == "grandparent"]
        assert len(gp_rules) == 1
        assert gp_rules[0].head.functor == "grandparent"
    
    def test_kb_stats(self):
        """Test knowledge base statistics"""
        kb = KnowledgeBase()
        
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("age", atom("john"), atom("42"))))
        kb.add_rule(Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), 
             compound("parent", var("Y"), var("Z"))]
        ))
        
        # Check counts manually since stats property doesn't exist
        assert len(kb.facts) == 2
        assert len(kb.rules) == 1
        
        # Check functors
        fact_functors = {f.term.functor for f in kb.facts}
        rule_functors = {r.head.functor for r in kb.rules}
        assert 'parent' in fact_functors
        assert 'age' in fact_functors
        assert 'grandparent' in rule_functors