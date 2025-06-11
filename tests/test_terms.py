"""
Comprehensive tests for JLOG term system
"""
import pytest
import sys
from pathlib import Path

# Add the jlog package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jlog import atom, var, compound, term_from_json, Atom, Variable, Compound
import json


class TestAtom:
    """Test atom functionality"""
    
    def test_creation(self):
        """Test atom creation"""
        a = atom("john")
        assert isinstance(a, Atom)
        assert a.value == "john"
    
    def test_string_representation(self):
        """Test string representation"""
        a = atom("test_atom")
        assert str(a) == "test_atom"
    
    def test_equality(self):
        """Test atom equality"""
        a1 = atom("same")
        a2 = atom("same") 
        a3 = atom("different")
        
        assert a1 == a2
        assert a1 != a3
        assert hash(a1) == hash(a2)
        assert hash(a1) != hash(a3)
    
    def test_substitution(self):
        """Test that atoms are not affected by substitution"""
        a = atom("constant")
        result = a.substitute({"X": atom("other")})
        assert result == a
    
    def test_variables(self):
        """Test that atoms contain no variables"""
        a = atom("test")
        assert a.get_variables() == set()
    
    def test_json_serialization(self):
        """Test JSON serialization/deserialization"""
        a = atom("test_value")
        
        # Serialize
        json_data = a.to_json()
        expected = {"type": "atom", "value": "test_value"}
        assert json_data == expected
        
        # Deserialize
        reconstructed = Atom.from_json(json_data)
        assert reconstructed == a
        
        # Round trip through term_from_json
        reconstructed2 = term_from_json(json_data)
        assert reconstructed2 == a


class TestVariable:
    """Test variable functionality"""
    
    def test_creation(self):
        """Test variable creation"""
        v = var("X")
        assert isinstance(v, Variable)
        assert v.name == "X"
    
    def test_string_representation(self):
        """Test string representation"""
        v = var("TestVar")
        assert str(v) == "TestVar"
    
    def test_equality(self):
        """Test variable equality"""
        v1 = var("X")
        v2 = var("X")
        v3 = var("Y")
        
        assert v1 == v2
        assert v1 != v3
        assert hash(v1) == hash(v2)
        assert hash(v1) != hash(v3)
    
    def test_substitution(self):
        """Test variable substitution"""
        v = var("X")
        replacement = atom("john")
        
        # Should substitute when binding exists
        result = v.substitute({"X": replacement})
        assert result == replacement
        
        # Should return self when no binding
        result = v.substitute({"Y": replacement})
        assert result == v
    
    def test_variables(self):
        """Test that variables contain themselves"""
        v = var("Test")
        assert v.get_variables() == {"Test"}
    
    def test_json_serialization(self):
        """Test JSON serialization/deserialization"""
        v = var("TestVariable")
        
        # Serialize
        json_data = v.to_json()
        expected = {"type": "variable", "name": "TestVariable"}
        assert json_data == expected
        
        # Deserialize
        reconstructed = Variable.from_json(json_data)
        assert reconstructed == v
        
        # Round trip through term_from_json
        reconstructed2 = term_from_json(json_data)
        assert reconstructed2 == v


class TestCompound:
    """Test compound term functionality"""
    
    def test_creation(self):
        """Test compound term creation"""
        c = compound("parent", atom("john"), atom("mary"))
        assert isinstance(c, Compound)
        assert c.functor == "parent"
        assert c.arity == 2
        assert len(c.args) == 2
    
    def test_creation_no_args(self):
        """Test compound with no arguments"""
        c = compound("constant")
        assert c.functor == "constant"
        assert c.arity == 0
        assert len(c.args) == 0
    
    def test_string_representation(self):
        """Test string representation"""
        # With arguments
        c1 = compound("likes", atom("john"), atom("pizza"))
        assert str(c1) == "likes(john, pizza)"
        
        # No arguments
        c2 = compound("true")
        assert str(c2) == "true"
        
        # Nested compounds
        c3 = compound("f", compound("g", atom("a")), var("X"))
        assert str(c3) == "f(g(a), X)"
    
    def test_equality(self):
        """Test compound equality"""
        c1 = compound("f", atom("a"), var("X"))
        c2 = compound("f", atom("a"), var("X"))
        c3 = compound("f", atom("a"), var("Y"))
        c4 = compound("g", atom("a"), var("X"))
        
        assert c1 == c2
        assert c1 != c3  # Different variable names
        assert c1 != c4  # Different functor
    
    def test_substitution(self):
        """Test compound substitution"""
        c = compound("parent", var("X"), atom("mary"))
        bindings = {"X": atom("john")}
        
        result = c.substitute(bindings)
        expected = compound("parent", atom("john"), atom("mary"))
        assert result == expected
        
        # Multiple variables
        c2 = compound("relationship", var("X"), var("Y"), var("Z"))
        bindings2 = {"X": atom("a"), "Z": atom("c")}
        result2 = c2.substitute(bindings2)
        expected2 = compound("relationship", atom("a"), var("Y"), atom("c"))
        assert result2 == expected2
    
    def test_variables(self):
        """Test variable extraction"""
        # No variables
        c1 = compound("fact", atom("a"), atom("b"))
        assert c1.get_variables() == set()
        
        # One variable
        c2 = compound("pred", var("X"), atom("a"))
        assert c2.get_variables() == {"X"}
        
        # Multiple variables
        c3 = compound("pred", var("X"), var("Y"), var("X"))  # X appears twice
        assert c3.get_variables() == {"X", "Y"}
        
        # Nested compounds
        c4 = compound("outer", compound("inner", var("X")), var("Y"))
        assert c4.get_variables() == {"X", "Y"}
    
    def test_immutability(self):
        """Test that compound terms are immutable"""
        c = compound("test", atom("a"))
        original_args = c.args
        
        # Args should be a tuple (immutable)
        assert isinstance(c.args, tuple)
        
        # Should not be able to modify
        with pytest.raises((AttributeError, TypeError)):
            c.args[0] = atom("b")
    
    def test_json_serialization(self):
        """Test JSON serialization/deserialization"""
        # Simple compound
        c1 = compound("parent", atom("john"), atom("mary"))
        json_data1 = c1.to_json()
        
        expected1 = {
            "type": "compound",
            "functor": "parent",
            "args": [
                {"type": "atom", "value": "john"},
                {"type": "atom", "value": "mary"}
            ]
        }
        assert json_data1 == expected1
        
        reconstructed1 = Compound.from_json(json_data1)
        assert reconstructed1 == c1
        
        # Complex compound with variables
        c2 = compound("rule", var("X"), compound("f", var("Y")))
        json_data2 = c2.to_json()
        reconstructed2 = term_from_json(json_data2)
        assert reconstructed2 == c2


class TestTermFromJson:
    def test_atom_from_json(self):
        """Test creating atom from JSON"""
        data = {"type": "atom", "value": "test"}
        term = term_from_json(data)
        assert isinstance(term, Atom)
        assert term.value == "test"

    def test_variable_from_json(self):
        """Test creating variable from JSON"""
        data = {"type": "variable", "name": "X"}
        term = term_from_json(data)
        assert isinstance(term, Variable)
        assert term.name == "X"

    def test_compound_from_json(self):
        """Test creating compound from JSON"""
        data = {
            "type": "compound",
            "functor": "parent",
            "args": [
                {"type": "atom", "value": "alice"},
                {"type": "atom", "value": "bob"}
            ]
        }
        term = term_from_json(data)
        assert isinstance(term, Compound)
        assert term.functor == "parent"
        assert len(term.args) == 2

    def test_missing_type(self):
        """Test error handling for missing type"""
        data = {"value": "test"}
        with pytest.raises(KeyError, match="Missing 'type' field"):
            term_from_json(data)

    def test_unknown_type(self):
        """Test error handling for unknown type"""
        data = {"type": "unknown", "value": "test"}
        with pytest.raises(ValueError, match="Unknown term type: unknown"):
            term_from_json(data)


class TestComplexTermStructures:
    """Test complex term structures and edge cases"""
    
    def test_deeply_nested_terms(self):
        """Test deeply nested compound terms"""
        # Build f(g(h(a)))
        inner = compound("h", atom("a"))
        middle = compound("g", inner)
        outer = compound("f", middle)
        
        assert outer.functor == "f"
        assert outer.arity == 1
        
        # Check variables are collected from nested structure
        nested_with_vars = compound("f", compound("g", var("X")), var("Y"))
        assert nested_with_vars.get_variables() == {"X", "Y"}
    
    def test_large_arity_terms(self):
        """Test terms with many arguments"""
        args = [atom(f"arg{i}") for i in range(10)]
        big_term = compound("big_predicate", *args)
        
        assert big_term.arity == 10
        assert len(big_term.args) == 10
        assert big_term.get_variables() == set()
    
    def test_repeated_variables(self):
        """Test terms with repeated variables"""
        c = compound("same_vars", var("X"), var("X"), var("Y"), var("X"))
        variables = c.get_variables()
        
        assert variables == {"X", "Y"}
        assert len(variables) == 2
    
    def test_complex_substitution_chains(self):
        """Test complex substitution scenarios"""
        # X -> f(Y), Y -> a
        term = compound("pred", var("X"))
        bindings1 = {"X": compound("f", var("Y"))}
        bindings2 = {"Y": atom("a")}
        
        # Apply first substitution
        result1 = term.substitute(bindings1)
        expected1 = compound("pred", compound("f", var("Y")))
        assert result1 == expected1
        
        # Apply second substitution
        result2 = result1.substitute(bindings2)
        expected2 = compound("pred", compound("f", atom("a")))
        assert result2 == expected2
    
    def test_json_round_trip_complex(self):
        """Test JSON round trip for complex structures"""
        # Complex nested structure
        complex_term = compound(
            "complex",
            compound("nested", var("X"), atom("constant")),
            var("Y"),
            compound("another", compound("deep", var("Z")))
        )
        
        # Serialize and deserialize
        json_data = complex_term.to_json()
        json_str = json.dumps(json_data)
        parsed_data = json.loads(json_str)
        reconstructed = term_from_json(parsed_data)
        
        assert reconstructed == complex_term
        assert reconstructed.get_variables() == complex_term.get_variables()
