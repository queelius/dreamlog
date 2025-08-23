"""
Tests for DreamLog unification
"""
import pytest
from dreamlog import atom, var, compound, unify, match, subsumes, Unifier
from dreamlog.unification import apply_substitution, is_ground


class TestBasicUnification:
    """Test basic unification operations"""
    
    def test_unify_atoms(self):
        """Test unifying atoms"""
        # Identical atoms unify with empty substitution
        result = unify(atom("john"), atom("john"))
        assert result is not None
        assert len(result) == 0
        
        # Different atoms don't unify
        result = unify(atom("john"), atom("mary"))
        assert result is None
    
    def test_unify_variables(self):
        """Test unifying variables"""
        # Variable unifies with atom
        x = var("X")
        result = unify(x, atom("john"))
        assert result is not None
        assert result[x] == atom("john")
        
        # Variable unifies with another variable
        y = var("Y")
        result = unify(x, y)
        assert result is not None
        assert result[x] == y or result[y] == x
    
    def test_unify_compound(self):
        """Test unifying compound terms"""
        # Same structure unifies
        t1 = compound("parent", atom("john"), atom("mary"))
        t2 = compound("parent", atom("john"), atom("mary"))
        result = unify(t1, t2)
        assert result is not None
        
        # Different functors don't unify
        t1 = compound("parent", atom("john"), atom("mary"))
        t2 = compound("child", atom("john"), atom("mary"))
        result = unify(t1, t2)
        assert result is None
        
        # Different arity doesn't unify
        t1 = compound("f", atom("a"))
        t2 = compound("f", atom("a"), atom("b"))
        result = unify(t1, t2)
        assert result is None
    
    def test_unify_with_variables(self):
        """Test unifying terms containing variables"""
        t1 = compound("parent", var("X"), atom("mary"))
        t2 = compound("parent", atom("john"), atom("mary"))
        
        result = unify(t1, t2)
        assert result is not None
        assert result[var("X")] == atom("john")
    
    def test_complex_unification(self):
        """Test complex nested unification"""
        t1 = compound("likes", var("X"), compound("food", var("Y")))
        t2 = compound("likes", atom("john"), compound("food", atom("pizza")))
        
        result = unify(t1, t2)
        assert result is not None
        assert result[var("X")] == atom("john")
        assert result[var("Y")] == atom("pizza")


class TestPatternMatching:
    """Test one-way pattern matching"""
    
    def test_match_basic(self):
        """Test basic pattern matching"""
        pattern = compound("parent", var("X"), var("Y"))
        term = compound("parent", atom("john"), atom("mary"))
        
        result = match(pattern, term)
        assert result is not None
        assert result[var("X")] == atom("john")
        assert result[var("Y")] == atom("mary")
    
    def test_match_no_reverse_binding(self):
        """Test that match doesn't bind variables in the term"""
        pattern = compound("parent", atom("john"), atom("mary"))
        term = compound("parent", var("X"), var("Y"))
        
        # This should fail because match is one-way
        result = match(pattern, term)
        assert result is None


class TestSubsumption:
    """Test subsumption checking"""
    
    def test_variable_subsumes_atom(self):
        """Test that variables subsume atoms"""
        assert subsumes(var("X"), atom("john"))
        assert not subsumes(atom("john"), var("X"))
    
    def test_general_subsumes_specific(self):
        """Test that general patterns subsume specific ones"""
        general = compound("parent", var("X"), var("Y"))
        specific = compound("parent", atom("john"), atom("mary"))
        
        assert subsumes(general, specific)
        assert not subsumes(specific, general)
    
    def test_identical_terms_subsume(self):
        """Test that identical terms subsume each other"""
        term = compound("parent", atom("john"), atom("mary"))
        assert subsumes(term, term)


class TestUnifier:
    """Test the stateful Unifier class"""
    
    def test_unifier_creation(self):
        """Test creating a unifier"""
        u = Unifier()
        assert u.bindings == {}
    
    def test_unifier_unify(self):
        """Test unifying with Unifier"""
        u = Unifier()
        
        success = u.unify(var("X"), atom("john"))
        assert success
        assert u.bindings[var("X")] == atom("john")
        
        # Unifying X again should check consistency
        success = u.unify(var("X"), atom("john"))
        assert success  # Same value, should succeed
        
        success = u.unify(var("X"), atom("mary"))
        assert not success  # Different value, should fail
    
    def test_unifier_apply(self):
        """Test applying substitution"""
        u = Unifier()
        u.unify(var("X"), atom("john"))
        u.unify(var("Y"), atom("mary"))
        
        term = compound("parent", var("X"), var("Y"))
        result = u.apply(term)
        
        assert result == compound("parent", atom("john"), atom("mary"))
    
    def test_unifier_copy(self):
        """Test copying a unifier"""
        u1 = Unifier()
        u1.unify(var("X"), atom("john"))
        
        u2 = u1.copy()
        u2.unify(var("Y"), atom("mary"))
        
        # Original shouldn't have Y binding
        assert var("Y") not in u1.bindings
        assert var("Y") in u2.bindings


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_apply_substitution(self):
        """Test applying a substitution to a term"""
        term = compound("parent", var("X"), var("Y"))
        subst = {var("X"): atom("john"), var("Y"): atom("mary")}
        
        result = apply_substitution(term, subst)
        assert result == compound("parent", atom("john"), atom("mary"))
    
    def test_is_ground(self):
        """Test checking if term is ground"""
        assert is_ground(atom("john"))
        assert is_ground(compound("parent", atom("john"), atom("mary")))
        assert not is_ground(var("X"))
        assert not is_ground(compound("parent", var("X"), atom("mary")))