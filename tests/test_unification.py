"""
Tests for DreamLog unification
"""
import pytest
from dreamlog import atom, var, compound, unify, match, subsumes, Unifier
from dreamlog.unification import apply_substitution, is_ground, UnificationMode, extract_variables


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
        assert result["X"] == atom("john")
        
        # Variable unifies with another variable
        y = var("Y")
        result = unify(x, y)
        assert result is not None
        assert result.get("X") == y or result.get("Y") == x
    
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
        assert result["X"] == atom("john")
    
    def test_complex_unification(self):
        """Test complex nested unification"""
        t1 = compound("likes", var("X"), compound("food", var("Y")))
        t2 = compound("likes", atom("john"), compound("food", atom("pizza")))
        
        result = unify(t1, t2)
        assert result is not None
        assert result["X"] == atom("john")
        assert result["Y"] == atom("pizza")


class TestPatternMatching:
    """Test one-way pattern matching"""
    
    def test_match_basic(self):
        """Test basic pattern matching"""
        pattern = compound("parent", var("X"), var("Y"))
        term = compound("parent", atom("john"), atom("mary"))
        
        result = match(pattern, term)
        assert result is not None
        assert result["X"] == atom("john")
        assert result["Y"] == atom("mary")
    
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
    """Test the Unifier class (functional style)"""
    
    def test_unifier_creation(self):
        """Test creating a unifier"""
        u = Unifier()
        # Unifier is functional, doesn't maintain state
        assert u.mode == UnificationMode.STANDARD
    
    def test_unifier_unify(self):
        """Test unifying with Unifier"""
        u = Unifier()
        
        result = u.unify(var("X"), atom("john"))
        assert result.success
        assert result.bindings["X"] == atom("john")
        
        # Unifying X again with same value
        result2 = u.unify(var("X"), atom("john"), bindings=result.bindings)
        assert result2.success  # Same value, should succeed
        
        # Unifying X with different value
        result3 = u.unify(var("X"), atom("mary"), bindings=result.bindings)
        assert not result3.success  # Different value, should fail
    
    def test_unifier_apply(self):
        """Test applying substitution"""
        u = Unifier()
        result1 = u.unify(var("X"), atom("john"))
        result2 = u.unify(var("Y"), atom("mary"), bindings=result1.bindings)
        
        term = compound("parent", var("X"), var("Y"))
        result = apply_substitution(term, result2.bindings)
        
        assert result == compound("parent", atom("john"), atom("mary"))
    
    def test_unifier_copy(self):
        """Test functional nature of unifier"""
        u = Unifier()
        result1 = u.unify(var("X"), atom("john"))
        
        # Each call gets its own bindings
        bindings_copy = result1.bindings.copy()
        result2 = u.unify(var("Y"), atom("mary"), bindings=bindings_copy)
        
        # Original bindings unchanged
        assert "Y" not in result1.bindings
        assert "Y" in result2.bindings


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_apply_substitution(self):
        """Test applying a substitution to a term"""
        term = compound("parent", var("X"), var("Y"))
        subst = {"X": atom("john"), "Y": atom("mary")}
        
        result = apply_substitution(term, subst)
        assert result == compound("parent", atom("john"), atom("mary"))
    
    def test_is_ground(self):
        """Test checking if term is ground"""
        assert is_ground(atom("john"))
        assert is_ground(compound("parent", atom("john"), atom("mary")))
        assert not is_ground(var("X"))
        assert not is_ground(compound("parent", var("X"), atom("mary")))


class TestAdvancedUnification:
    """Test advanced unification features"""
    
    def test_unification_result_bool(self):
        """Test UnificationResult __bool__ method"""
        from dreamlog.unification import UnificationResult
        
        # Successful result evaluates to True
        success_result = UnificationResult(success=True, bindings={"X": atom("john")})
        assert bool(success_result)
        assert success_result  # Direct boolean evaluation
        
        # Failed result evaluates to False
        failure_result = UnificationResult(success=False)
        assert not bool(failure_result)
        assert not failure_result  # Direct boolean evaluation
    
    def test_unification_trace(self):
        """Test UnificationTrace debugging"""
        from dreamlog.unification import UnificationTrace
        
        # Test disabled trace
        trace = UnificationTrace(enabled=False)
        trace.add("test message")
        assert len(trace.get_trace()) == 0
        
        # Test enabled trace
        trace = UnificationTrace(enabled=True)
        trace.add("unifying X with john")
        trace.add("success")
        steps = trace.get_trace()
        assert len(steps) == 2
        assert "unifying X with john" in steps
        assert "success" in steps
    
    def test_unification_with_trace(self):
        """Test unification with tracing enabled"""
        result = unify(var("X"), atom("john"), trace=True)
        assert result is not None
        assert result["X"] == atom("john")
        
        # Test traced unifier
        unifier = Unifier(trace=True)
        result = unifier.unify(var("X"), atom("john"))
        assert result.success
        assert len(result.steps) > 0  # Should have trace steps
    
    def test_compose_substitutions(self):
        """Test substitution composition"""
        from dreamlog.unification import compose_substitutions
        
        s1 = {"X": atom("john"), "Z": atom("mary")}
        s2 = {"Y": var("Z"), "W": atom("alice")}
        
        composed = compose_substitutions(s1, s2)
        
        # Y should be resolved through Z to mary (s1 applied to s2's terms)
        assert composed["Y"] == atom("mary")  # Y = Z, and Z = mary in s1
        # Direct mappings should be preserved
        assert composed["X"] == atom("john")  # from s1
        assert composed["W"] == atom("alice")  # from s2
        assert composed["Z"] == atom("mary")   # from s1
    
    def test_unification_modes(self):
        """Test different unification modes"""
        from dreamlog.unification import UnificationMode
        
        # Standard mode (default)
        result = unify(var("X"), atom("john"), mode=UnificationMode.STANDARD)
        assert result is not None
        
        # Match mode (one-way)
        result = match(var("X"), atom("john"))
        assert result is not None
        assert result["X"] == atom("john")
        
        # Reverse match should fail
        result = match(atom("john"), var("X"))
        assert result is None
        
        # Subsumption mode
        assert subsumes(var("X"), atom("john"))
        assert not subsumes(atom("john"), var("X"))  # Reversed
    
    def test_occurs_check(self):
        """Test occurs check prevents infinite structures"""
        # X = f(X) should fail occurs check
        x = var("X")
        term = compound("f", x)
        
        result = unify(x, term)
        assert result is None  # Should fail due to occurs check
        
        # Test with unifier directly
        unifier = Unifier()
        result = unifier.unify(x, term)
        assert not result.success
    
    def test_pattern_building(self):
        """Test building patterns from templates"""
        from dreamlog.unification import build_pattern
        
        # Simple pattern
        template = ["parent", "X", "mary"]
        pattern = build_pattern(template)
        expected = compound("parent", var("X"), atom("mary"))
        assert pattern == expected
        
        # Nested pattern
        template = ["parent", ["father", "X"], "Y"]
        pattern = build_pattern(template)
        expected = compound("parent", compound("father", var("X")), var("Y"))
        assert pattern == expected
    
    def test_variable_extraction(self):
        """Test extracting variables from terms"""
        from dreamlog.unification import extract_variables
        
        # Simple term with variables
        term = compound("parent", var("X"), var("Y"))
        variables = extract_variables(term)
        assert variables == {"X", "Y"}
        
        # Ground term
        term = compound("parent", atom("john"), atom("mary"))
        variables = extract_variables(term)
        assert len(variables) == 0
        
        # Nested variables
        term = compound("ancestor", var("X"), compound("child", var("Y"), var("Z")))
        variables = extract_variables(term)
        assert variables == {"X", "Y", "Z"}
    
    def test_variable_renaming(self):
        """Test renaming variables to avoid conflicts"""
        from dreamlog.unification import rename_variables
        
        term = compound("parent", var("X"), var("Y"))
        renamed_term, var_mapping = rename_variables(term, "_new")
        
        # Should have new variable names
        variables = extract_variables(renamed_term)
        assert "X_new" in variables
        assert "Y_new" in variables
        
        # Check mapping
        assert var_mapping["X"] == "X_new"
        assert var_mapping["Y"] == "Y_new"
    
    def test_different_unifier_modes(self):
        """Test different unifier initialization modes"""
        # Test different unifier configurations
        unifier1 = Unifier(mode=UnificationMode.STANDARD, trace=False)
        unifier2 = Unifier(mode=UnificationMode.MATCH, trace=True)
        unifier3 = Unifier(mode=UnificationMode.SUBSUME, occurs_check=False)
        
        # Test they have different configurations
        assert unifier1.mode == UnificationMode.STANDARD
        assert unifier2.mode == UnificationMode.MATCH
        assert unifier3.mode == UnificationMode.SUBSUME
        
        # Test basic unification with each mode
        x = var("X")
        john = atom("john")
        
        result1 = unifier1.unify(x, john)
        assert result1.success
        assert result1.bindings["X"] == john
        
        result2 = unifier2.unify(x, john) 
        assert result2.success
        assert len(result2.steps) > 0  # Should have trace steps
        
        result3 = unifier3.unify(x, john)
        assert result3.success
    
    def test_error_conditions(self):
        """Test various error conditions"""
        # Test unifying terms of different arities
        t1 = compound("parent", atom("john"))
        t2 = compound("parent", atom("john"), atom("mary"))
        result = unify(t1, t2)
        assert result is None
        
        # Test unifying different compound structures
        t1 = compound("parent", atom("john"), atom("mary"))
        t2 = compound("child", atom("john"), atom("mary"))
        result = unify(t1, t2)
        assert result is None
    
    def test_complex_unification_scenarios(self):
        """Test complex real-world unification scenarios"""
        # Family tree scenario
        query = compound("grandparent", var("GP"), var("GC"))
        fact = compound("grandparent", atom("john"), atom("alice"))
        
        result = unify(query, fact)
        assert result is not None
        assert result["GP"] == atom("john")
        assert result["GC"] == atom("alice")
        
        # Multiple variable chains
        t1 = compound("chain", var("X"), compound("link", var("Y"), var("Z")))
        t2 = compound("chain", atom("start"), compound("link", atom("middle"), atom("end")))
        
        result = unify(t1, t2)
        assert result is not None
        assert result["X"] == atom("start")
        assert result["Y"] == atom("middle")
        assert result["Z"] == atom("end")