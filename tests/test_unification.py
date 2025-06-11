"""
Tests for unification algorithm
"""
import pytest
import sys
from pathlib import Path

# Add the jlog package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jlog import atom, var, compound, Unifier


class TestBasicUnification:
    """Test basic unification cases"""
    
    def test_identical_atoms(self):
        """Identical atoms unify with empty substitution"""
        a1 = atom("john")
        a2 = atom("john")
        
        result = Unifier.unify(a1, a2)
        assert result == {}
    
    def test_different_atoms(self):
        """Different atoms don't unify"""
        a1 = atom("john")
        a2 = atom("mary")
        
        result = Unifier.unify(a1, a2)
        assert result is None
    
    def test_variable_with_atom(self):
        """Variable unifies with atom"""
        v = var("X")
        a = atom("john")
        
        # Variable first
        result = Unifier.unify(v, a)
        assert result == {"X": a}
        
        # Atom first (should be symmetric)
        result = Unifier.unify(a, v)
        assert result == {"X": a}
    
    def test_variable_with_variable(self):
        """Variables unify with each other"""
        v1 = var("X")
        v2 = var("Y")
        
        result = Unifier.unify(v1, v2)
        # Either X->Y or Y->X is valid
        assert result in [{"X": v2}, {"Y": v1}]
    
    def test_same_variable(self):
        """Same variable unifies with itself"""
        v = var("X")
        
        result = Unifier.unify(v, v)
        assert result == {}


class TestCompoundUnification:
    """Test unification of compound terms"""
    
    def test_identical_compounds(self):
        """Identical compounds unify"""
        c1 = compound("parent", atom("john"), atom("mary"))
        c2 = compound("parent", atom("john"), atom("mary"))
        
        result = Unifier.unify(c1, c2)
        assert result == {}
    
    def test_different_functors(self):
        """Compounds with different functors don't unify"""
        c1 = compound("parent", atom("john"), atom("mary"))
        c2 = compound("likes", atom("john"), atom("mary"))
        
        result = Unifier.unify(c1, c2)
        assert result is None
    
    def test_different_arity(self):
        """Compounds with different arity don't unify"""
        c1 = compound("parent", atom("john"), atom("mary"))
        c2 = compound("parent", atom("john"))
        
        result = Unifier.unify(c1, c2)
        assert result is None
    
    def test_compound_with_variables(self):
        """Compound terms with variables"""
        c1 = compound("parent", var("X"), atom("mary"))
        c2 = compound("parent", atom("john"), atom("mary"))
        
        result = Unifier.unify(c1, c2)
        assert result == {"X": atom("john")}
    
    def test_multiple_variables(self):
        """Multiple variables in compound terms"""
        c1 = compound("relationship", var("X"), var("Y"))
        c2 = compound("relationship", atom("john"), atom("mary"))
        
        result = Unifier.unify(c1, c2)
        expected = {"X": atom("john"), "Y": atom("mary")}
        assert result == expected
    
    def test_shared_variables(self):
        """Compounds sharing variables"""
        c1 = compound("same", var("X"), var("X"))
        c2 = compound("same", atom("john"), atom("john"))
        
        result = Unifier.unify(c1, c2)
        assert result == {"X": atom("john")}
        
        # Should fail if atoms are different
        c3 = compound("same", atom("john"), atom("mary"))
        result = Unifier.unify(c1, c3)
        assert result is None
    
    def test_nested_compounds(self):
        """Nested compound terms"""
        c1 = compound("f", compound("g", var("X")))
        c2 = compound("f", compound("g", atom("a")))
        
        result = Unifier.unify(c1, c2)
        assert result == {"X": atom("a")}


class TestUnificationWithBindings:
    """Test unification with existing bindings"""
    
    def test_compatible_bindings(self):
        """Compatible existing bindings"""
        v = var("X")
        a = atom("john")
        existing = {"Y": atom("mary")}
        
        result = Unifier.unify(v, a, existing)
        expected = {"Y": atom("mary"), "X": a}
        assert result == expected
    
    def test_conflicting_bindings(self):
        """Conflicting bindings should fail"""
        v = var("X")
        a1 = atom("john")
        a2 = atom("mary")
        existing = {"X": a1}
        
        result = Unifier.unify(v, a2, existing)
        assert result is None
    
    def test_variable_already_bound(self):
        """Variable already bound in existing bindings"""
        v = var("X")
        a = atom("john")
        existing = {"X": a}
        
        result = Unifier.unify(v, a, existing)
        assert result == existing
    
    def test_transitive_bindings(self):
        """Transitive variable bindings"""
        v1 = var("X")
        v2 = var("Y")
        a = atom("john")
        existing = {"Y": a}
        
        result = Unifier.unify(v1, v2, existing)
        # X should be bound to john through Y
        assert result["X"] == a or result["Y"] == a


class TestOccursCheck:
    """Test occurs check functionality"""
    
    def test_simple_occurs_check(self):
        """Simple occurs check - X should not unify with f(X)"""
        v = var("X")
        c = compound("f", v)
        
        result = Unifier.unify(v, c)
        assert result is None
    
    def test_nested_occurs_check(self):
        """Nested occurs check - X should not unify with f(g(X))"""
        v = var("X")
        c = compound("f", compound("g", v))
        
        result = Unifier.unify(v, c)
        assert result is None
    
    def test_occurs_check_with_different_variables(self):
        """Occurs check should not prevent different variables"""
        v1 = var("X")
        v2 = var("Y")
        c = compound("f", v2)
        
        result = Unifier.unify(v1, c)
        assert result == {"X": c}
    
    def test_occurs_check_in_compound_args(self):
        """Occurs check within compound arguments"""
        v = var("X")
        c1 = compound("f", v, atom("a"))
        c2 = compound("f", compound("g", v), atom("a"))
        
        result = Unifier.unify(c1, c2)
        assert result is None
    
    def test_complex_occurs_check(self):
        """Complex occurs check scenarios"""
        # X unifies with f(Y) should work
        v1 = var("X")
        v2 = var("Y")
        c1 = compound("f", v2)
        
        result = Unifier.unify(v1, c1)
        assert result == {"X": c1}
        
        # But X should not unify with f(X) even through substitution
        existing = {"Y": v1}
        result = Unifier.unify(v1, c1, existing)
        assert result is None


class TestAdvancedUnification:
    """Test advanced unification scenarios"""
    
    def test_complex_term_unification(self):
        """Complex nested term unification"""
        # parent(X, child(Y, Z)) unifies with parent(john, child(mary, alice))
        c1 = compound("parent", var("X"), compound("child", var("Y"), var("Z")))
        c2 = compound("parent", atom("john"), compound("child", atom("mary"), atom("alice")))
        
        result = Unifier.unify(c1, c2)
        expected = {
            "X": atom("john"),
            "Y": atom("mary"),
            "Z": atom("alice")
        }
        assert result == expected
    
    def test_partial_instantiation(self):
        """Partial instantiation of terms"""
        # f(X, g(Y)) unifies with f(a, Z)
        c1 = compound("f", var("X"), compound("g", var("Y")))
        c2 = compound("f", atom("a"), var("Z"))
        
        result = Unifier.unify(c1, c2)
        expected = {
            "X": atom("a"),
            "Z": compound("g", var("Y"))
        }
        assert result == expected
    
    def test_variable_chains(self):
        """Variable chains in unification"""
        # X=Y, Y=Z, Z=a should result in X=Y=Z=a
        bindings = {}
        
        # X unifies with Y
        result = Unifier.unify(var("X"), var("Y"), bindings)
        assert result is not None
        bindings = result
        print(f"After X=Y: {bindings}")
        
        # Y unifies with Z
        result = Unifier.unify(var("Y"), var("Z"), bindings)
        assert result is not None
        bindings = result
        print(f"After Y=Z: {bindings}")
        
        # Z unifies with a
        result = Unifier.unify(var("Z"), atom("a"), bindings)
        assert result is not None
        print(f"After Z=a: {result}")
        
        # All should resolve to 'a'
        final_x = var("X").substitute(result)
        final_y = var("Y").substitute(result)
        final_z = var("Z").substitute(result)
        
        print(f"final_x: {final_x}")
        print(f"final_y: {final_y}")
        print(f"final_z: {final_z}")
        
        assert final_x == atom("a")
        assert final_y == atom("a")
        assert final_z == atom("a")
    
    def test_bidirectional_unification(self):
        """Test that unification is symmetric"""
        test_cases = [
            (atom("a"), atom("a")),
            (var("X"), atom("a")),
            (compound("f", var("X")), compound("f", atom("a"))),
            (compound("f", var("X"), var("Y")), compound("f", atom("a"), atom("b")))
        ]
        
        for term1, term2 in test_cases:
            result1 = Unifier.unify(term1, term2)
            result2 = Unifier.unify(term2, term1)
            
            if result1 is None:
                assert result2 is None
            else:
                assert result2 is not None
                # Results might have different variable assignments but should be equivalent
                assert term1.substitute(result1) == term2.substitute(result1)
                assert term1.substitute(result2) == term2.substitute(result2)


class TestUnificationUtilities:
    """Test unification utility functions"""
    
    def test_apply_bindings(self):
        """Test applying bindings to terms"""
        term = compound("f", var("X"), atom("a"))
        bindings = {"X": atom("b")}
        
        result = Unifier.apply_bindings(term, bindings)
        expected = compound("f", atom("b"), atom("a"))
        assert result == expected
    
    def test_compose_bindings(self):
        """Test binding composition"""
        bindings1 = {"X": atom("a"), "Y": var("Z")}
        bindings2 = {"Z": atom("b"), "W": atom("c")}
        
        result = Unifier.compose_bindings(bindings1, bindings2)
        
        # X should still map to 'a'
        assert result["X"] == atom("a")
        
        # Y should map to 'b' (through Z)
        assert result["Y"] == atom("b")
        
        # W should be added
        assert result["W"] == atom("c")
    
    def test_ground_term_check(self):
        """Test ground term checking"""
        # Ground terms
        assert Unifier.ground_term(atom("a"), {})
        assert Unifier.ground_term(compound("f", atom("a"), atom("b")), {})
        
        # Non-ground terms
        assert not Unifier.ground_term(var("X"), {})
        assert not Unifier.ground_term(compound("f", var("X")), {})
        
        # Terms that become ground after substitution
        term = compound("f", var("X"))
        bindings = {"X": atom("a")}
        assert Unifier.ground_term(term, bindings)
        
        # Terms with unbound variables after substitution
        term2 = compound("f", var("X"), var("Y"))
        bindings2 = {"X": atom("a")}
        assert not Unifier.ground_term(term2, bindings2)


class TestUnificationEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_compound_unification(self):
        """Test unification of compounds with no arguments"""
        c1 = compound("constant")
        c2 = compound("constant")
        c3 = compound("other")
        
        assert Unifier.unify(c1, c2) == {}
        assert Unifier.unify(c1, c3) is None
    
    def test_unification_with_none_bindings(self):
        """Test unification with None as initial bindings"""
        result = Unifier.unify(var("X"), atom("a"), None)
        assert result == {"X": atom("a")}
    
    def test_self_unification(self):
        """Test unifying a term with itself"""
        terms = [
            atom("a"),
            var("X"),
            compound("f", atom("a")),
            compound("f", var("X"), compound("g", var("Y")))
        ]
        
        for term in terms:
            result = Unifier.unify(term, term)
            assert result == {}
    
    def test_large_term_unification(self):
        """Test unification of large terms"""
        # Create large compound terms
        args1 = [var(f"X{i}") for i in range(20)]
        args2 = [atom(f"a{i}") for i in range(20)]
        
        c1 = compound("big", *args1)
        c2 = compound("big", *args2)
        
        result = Unifier.unify(c1, c2)
        assert result is not None
        assert len(result) == 20
        
        for i in range(20):
            assert result[f"X{i}"] == atom(f"a{i}")
