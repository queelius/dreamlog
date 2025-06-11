"""
Test suite for JLOG core functionality
"""
import pytest
import sys
from pathlib import Path

# Add the jlog package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jlog import (
    atom, var, compound, term_from_json,
    Fact, Rule, KnowledgeBase,
    Unifier, PrologEvaluator, Solution,
    JLogEngine, LLMHook, create_engine_with_llm
)


"""
Integration tests for JLOG system
"""
import pytest
import sys
from pathlib import Path
import json

# Add the jlog package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jlog import (
    atom, var, compound, JLogEngine,
)




class TestSystemIntegration:
    """Test complete system integration"""
    
    def test_basic_prolog_workflow(self):
        """Test basic Prolog workflow without LLM"""
        engine = JLogEngine()
        
        # Build knowledge base
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        engine.add_fact_from_term(compound("parent", atom("mary"), atom("alice")))
        engine.add_rule_from_terms(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        )
        
        # Test queries
        assert engine.ask(compound("parent", atom("john"), atom("mary")))
        assert not engine.ask(compound("parent", atom("alice"), atom("john")))
        
        solutions = engine.query([compound("grandparent", var("X"), var("Y"))])
        assert len(solutions) == 1
    


class TestTerms:
    """Test term creation and basic operations"""
    
    def test_atom_creation(self):
        """Test atom creation and properties"""
        a = atom("test")
        assert a.value == "test"
        assert str(a) == "test"
        assert a.get_variables() == set()
    
    def test_variable_creation(self):
        """Test variable creation and properties"""
        v = var("X")
        assert v.name == "X"
        assert str(v) == "X"
        assert v.get_variables() == {"X"}
    
    def test_compound_creation(self):
        """Test compound term creation"""
        c = compound("parent", atom("john"), atom("mary"))
        assert c.functor == "parent"
        assert c.arity == 2
        assert len(c.args) == 2
        assert str(c) == "parent(john, mary)"
    
    def test_term_equality(self):
        """Test term equality"""
        a1 = atom("test")
        a2 = atom("test")
        a3 = atom("other")
        
        assert a1 == a2
        assert a1 != a3
        
        c1 = compound("f", atom("a"))
        c2 = compound("f", atom("a"))
        c3 = compound("f", atom("b"))
        
        assert c1 == c2
        assert c1 != c3
    
    def test_term_substitution(self):
        """Test variable substitution in terms"""
        v = var("X")
        a = atom("john")
        
        # Variable substitution
        result = v.substitute({"X": a})
        assert result == a
        
        # Compound substitution
        c = compound("parent", var("X"), atom("mary"))
        result = c.substitute({"X": atom("john")})
        expected = compound("parent", atom("john"), atom("mary"))
        assert result == expected
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization"""
        # Atom
        a = atom("test")
        json_data = a.to_json()
        assert json_data == {"type": "atom", "value": "test"}
        reconstructed = term_from_json(json_data)
        assert reconstructed == a
        
        # Variable
        v = var("X")
        json_data = v.to_json()
        assert json_data == {"type": "variable", "name": "X"}
        reconstructed = term_from_json(json_data)
        assert reconstructed == v
        
        # Compound
        c = compound("parent", atom("john"), var("X"))
        json_data = c.to_json()
        reconstructed = term_from_json(json_data)
        assert reconstructed == c


class TestUnification:
    """Test unification algorithm"""
    
    def test_atom_unification(self):
        """Test unification of atoms"""
        a1 = atom("john")
        a2 = atom("john")
        a3 = atom("mary")
        
        # Same atoms unify
        result = Unifier.unify(a1, a2)
        assert result == {}
        
        # Different atoms don't unify
        result = Unifier.unify(a1, a3)
        assert result is None
    
    def test_variable_unification(self):
        """Test unification with variables"""
        v = var("X")
        a = atom("john")
        
        # Variable unifies with atom
        result = Unifier.unify(v, a)
        assert result == {"X": a}
        
        # Symmetric
        result = Unifier.unify(a, v)
        assert result == {"X": a}
        
        # Two variables unify
        v1 = var("X")
        v2 = var("Y")
        result = Unifier.unify(v1, v2)
        assert result in [{"X": v2}, {"Y": v1}]  # Either is valid
    
    def test_compound_unification(self):
        """Test unification of compound terms"""
        c1 = compound("parent", atom("john"), var("X"))
        c2 = compound("parent", atom("john"), atom("mary"))
        
        result = Unifier.unify(c1, c2)
        assert result == {"X": atom("mary")}
        
        # Different functors don't unify
        c3 = compound("mother", atom("john"), atom("mary"))
        result = Unifier.unify(c1, c3)
        assert result is None
        
        # Different arity doesn't unify
        c4 = compound("parent", atom("john"))
        result = Unifier.unify(c1, c4)
        assert result is None
    
    def test_occurs_check(self):
        """Test occurs check prevents infinite structures"""
        v = var("X")
        c = compound("f", v)
        
        # X should not unify with f(X)
        result = Unifier.unify(v, c)
        assert result is None
    
    def test_complex_unification(self):
        """Test complex unification scenarios"""
        # grandparent(X, Z) unifies with grandparent(john, mary)
        c1 = compound("grandparent", var("X"), var("Z"))
        c2 = compound("grandparent", atom("john"), atom("mary"))
        
        result = Unifier.unify(c1, c2)
        expected = {"X": atom("john"), "Z": atom("mary")}
        assert result == expected


class TestKnowledgeBase:
    """Test knowledge base operations"""
    
    def test_fact_operations(self):
        """Test adding and retrieving facts"""
        kb = KnowledgeBase()
        fact = Fact(compound("parent", atom("john"), atom("mary")))
        
        kb.add_fact(fact)
        assert len(kb.facts) == 1
        assert fact in kb.facts
        
        # Test duplicate prevention
        kb.add_fact(fact)
        assert len(kb.facts) == 1
    
    def test_rule_operations(self):
        """Test adding and retrieving rules"""
        kb = KnowledgeBase()
        rule = Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        )
        
        kb.add_rule(rule)
        assert len(kb.rules) == 1
        assert rule in kb.rules
    
    def test_indexed_lookup(self):
        """Test efficient lookup by functor/arity"""
        kb = KnowledgeBase()
        
        # Add facts with different functors
        fact1 = Fact(compound("parent", atom("john"), atom("mary")))
        fact2 = Fact(compound("parent", atom("mary"), atom("alice")))
        fact3 = Fact(compound("likes", atom("john"), atom("pizza")))
        
        kb.add_fact(fact1)
        kb.add_fact(fact2)
        kb.add_fact(fact3)
        
        # Query for parent facts
        query_term = compound("parent", var("X"), var("Y"))
        matching_facts = list(kb.get_matching_facts(query_term))
        
        assert len(matching_facts) == 2
        assert fact1 in matching_facts
        assert fact2 in matching_facts
        assert fact3 not in matching_facts
    
    def test_json_serialization(self):
        """Test knowledge base JSON serialization"""
        kb = KnowledgeBase()
        
        # Add some facts and rules
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_rule(Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        ))
        
        # Export to JSON
        json_str = kb.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Import to new KB
        kb2 = KnowledgeBase()
        kb2.from_json(json_str)
        
        assert len(kb2.facts) == len(kb.facts)
        assert len(kb2.rules) == len(kb.rules)


class TestEvaluator:
    """Test Prolog evaluator"""
    
    def setup_method(self):
        """Set up test knowledge base"""
        self.kb = KnowledgeBase()
        
        # Add facts
        facts = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("john"), atom("tom"))),
            Fact(compound("parent", atom("mary"), atom("alice"))),
            Fact(compound("male", atom("john"))),
            Fact(compound("male", atom("tom"))),
            Fact(compound("female", atom("mary"))),
            Fact(compound("female", atom("alice")))
        ]
        
        for fact in facts:
            self.kb.add_fact(fact)
        
        # Add rules
        rules = [
            Rule(
                compound("grandparent", var("X"), var("Z")),
                [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
            ),
            Rule(
                compound("father", var("X"), var("Y")),
                [compound("parent", var("X"), var("Y")), compound("male", var("X"))]
            )
        ]
        
        for rule in rules:
            self.kb.add_rule(rule)
        
        self.evaluator = PrologEvaluator(self.kb)
    
    def test_simple_fact_query(self):
        """Test querying simple facts"""
        query = [compound("parent", atom("john"), var("X"))]
        solutions = list(self.evaluator.query(query))
        
        assert len(solutions) == 2
        
        # Check that we get mary and tom as solutions
        x_values = []
        for solution in solutions:
            x_binding = solution.get_binding("X")
            if x_binding:
                x_values.append(x_binding)
        
        assert atom("mary") in x_values
        assert atom("tom") in x_values
    
    def test_rule_query(self):
        """Test querying rules that require inference"""
        query = [compound("grandparent", var("X"), var("Y"))]
        solutions = list(self.evaluator.query(query))
        
        assert len(solutions) == 1
        
        solution = solutions[0]
        x_binding = solution.get_binding("X")
        y_binding = solution.get_binding("Y")
        
        assert x_binding == atom("john")
        assert y_binding == atom("alice")
    
    def test_conjunctive_query(self):
        """Test queries with multiple goals"""
        query = [
            compound("parent", var("X"), var("Y")),
            compound("male", var("X"))
        ]
        solutions = list(self.evaluator.query(query))
        
        # Should find john as parent who is male
        assert len(solutions) == 2  # john->mary and john->tom
        
        for solution in solutions:
            x_binding = solution.get_binding("X")
            assert x_binding == atom("john")
    
    def test_yes_no_query(self):
        """Test yes/no queries"""
        # Should be true
        result = self.evaluator.ask_yes_no(compound("parent", atom("john"), atom("mary")))
        assert result is True
        
        # Should be false
        result = self.evaluator.ask_yes_no(compound("parent", atom("mary"), atom("john")))
        assert result is False



class TestJLogEngine:
    """Test high-level JLogEngine functionality"""
    
    def test_engine_creation(self):
        """Test engine creation and basic operations"""
        engine = JLogEngine()
        
        assert len(engine.facts) == 0
        assert len(engine.rules) == 0
        
        # Add a fact
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        assert len(engine.facts) == 1
        
        # Add a rule
        engine.add_rule_from_terms(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        )
        assert len(engine.rules) == 1
    
    def test_engine_queries(self):
        """Test engine query functionality"""
        engine = JLogEngine()
        
        # Set up knowledge base
        engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        engine.add_fact_from_term(compound("parent", atom("mary"), atom("alice")))
        engine.add_rule_from_terms(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        )
        
        # Test ask (yes/no)
        assert engine.ask(compound("parent", atom("john"), atom("mary"))) is True
        assert engine.ask(compound("parent", atom("alice"), atom("john"))) is False
        
        # Test find_all
        children = engine.find_all(compound("parent", atom("john"), var("X")), "X")
        assert atom("mary") in children
        assert len(children) == 1
        
        # Test query
        solutions = engine.query([compound("grandparent", var("X"), var("Y"))])
        assert len(solutions) == 1
    
    def test_engine_json_operations(self):
        """Test engine JSON import/export"""
        engine1 = JLogEngine()
        
        # Add some knowledge
        engine1.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
        engine1.add_rule_from_terms(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        )
        
        # Export to JSON
        json_data = engine1.save_to_json()
        
        # Import to new engine
        engine2 = JLogEngine()
        engine2.load_from_json(json_data)
        
        # Verify knowledge transferred
        assert len(engine2.facts) == len(engine1.facts)
        assert len(engine2.rules) == len(engine1.rules)
        
        # Verify functionality
        assert engine2.ask(compound("parent", atom("john"), atom("mary"))) is True
    

# Integration tests
class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_complex_reasoning_scenario(self):
        """Test complex reasoning scenario"""
        engine = JLogEngine()
        
        # Build a small family tree
        facts = [
            compound("parent", atom("john"), atom("mary")),
            compound("parent", atom("john"), atom("tom")),
            compound("parent", atom("mary"), atom("alice")),
            compound("parent", atom("tom"), atom("bob")),
            compound("male", atom("john")),
            compound("male", atom("tom")),
            compound("male", atom("bob")),
            compound("female", atom("mary")),
            compound("female", atom("alice"))
        ]
        
        for fact_term in facts:
            engine.add_fact_from_term(fact_term)
        
        # Add complex rules
        rules = [
            # Grandparent rule
            (compound("grandparent", var("X"), var("Z")),
             [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]),
            
            # Ancestor rule (recursive)
            (compound("ancestor", var("X"), var("Y")),
             [compound("parent", var("X"), var("Y"))]),
            
            (compound("ancestor", var("X"), var("Z")),
             [compound("parent", var("X"), var("Y")), compound("ancestor", var("Y"), var("Z"))]),
            
            # Gender-specific rules
            (compound("father", var("X"), var("Y")),
             [compound("parent", var("X"), var("Y")), compound("male", var("X"))]),
            
            (compound("grandmother", var("X"), var("Z")),
             [compound("grandparent", var("X"), var("Z")), compound("female", var("X"))])
        ]
        
        for head, body in rules:
            engine.add_rule_from_terms(head, body)
        
        # Test various queries
        
        # Who are John's grandchildren?
        grandchildren = engine.find_all(compound("grandparent", atom("john"), var("X")), "X")
        assert atom("alice") in grandchildren
        assert atom("bob") in grandchildren
        
        # Who are the fathers?
        fathers = engine.find_all(compound("father", var("X"), var("Y")), "X")
        assert atom("john") in fathers
        assert atom("tom") in fathers
        
        # Is Mary a grandmother?
        assert engine.ask(compound("grandmother", atom("mary"), var("X"))) is False
        
        # Are there any ancestors?
        ancestors = engine.find_all(compound("ancestor", var("X"), var("Y")), "X")
        assert len(ancestors) > 0
        assert atom("john") in ancestors
        
   
