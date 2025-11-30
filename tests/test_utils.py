#!/usr/bin/env python3
"""
Unit tests for DreamLog utility functions

Tests dereference_variable, get_all_variables, and is_ground_term functions.
"""

import pytest
from dreamlog.utils import dereference_variable, get_all_variables, is_ground_term
from dreamlog.terms import Variable, Atom, Compound


# Helper functions for creating terms
def var(name: str) -> Variable:
    """Create a Variable"""
    return Variable(name)


def atom(name: str) -> Atom:
    """Create an Atom"""
    return Atom(name)


def compound(functor: str, *args) -> Compound:
    """Create a Compound term"""
    return Compound(functor, list(args))


class TestDereferenceVariable:
    """Test dereference_variable function"""

    def test_unbound_variable(self):
        """Test dereferencing an unbound variable returns itself"""
        # Given: A variable not in bindings
        variable = Variable("X")
        bindings = {}

        # When: Dereferencing
        result = dereference_variable(variable, bindings)

        # Then: Should return the original variable
        assert result == variable
        assert isinstance(result, Variable)
        assert result.name == "X"

    def test_variable_bound_to_atom(self):
        """Test dereferencing variable bound to atom"""
        # Given: Variable bound to an atom
        variable = Variable("X")
        bindings = {"X": Atom("value")}

        # When: Dereferencing
        result = dereference_variable(variable, bindings)

        # Then: Should return the atom
        assert isinstance(result, Atom)
        assert result.value == "value"

    def test_variable_chain_resolution(self):
        """Test following chain of variable bindings"""
        # Given: Chain of variable bindings X -> Y -> Z -> atom
        variable = Variable("X")
        bindings = {
            "X": Variable("Y"),
            "Y": Variable("Z"),
            "Z": Atom("final_value")
        }

        # When: Dereferencing X
        result = dereference_variable(variable, bindings)

        # Then: Should follow chain to final atom
        assert isinstance(result, Atom)
        assert result.value == "final_value"

    def test_variable_chain_to_unbound(self):
        """Test chain ending in unbound variable"""
        # Given: Chain ending in unbound variable
        variable = Variable("X")
        bindings = {
            "X": Variable("Y"),
            "Y": Variable("Z")
            # Z is not bound
        }

        # When: Dereferencing
        result = dereference_variable(variable, bindings)

        # Then: Should return the last unbound variable
        assert isinstance(result, Variable)
        assert result.name == "Z"

    def test_circular_reference_detection(self):
        """Test that circular references are handled"""
        # Given: Circular bindings X -> Y -> X
        variable = Variable("X")
        bindings = {
            "X": Variable("Y"),
            "Y": Variable("X")
        }

        # When: Dereferencing
        result = dereference_variable(variable, bindings)

        # Then: Should stop at cycle (returns Y since it's the first to revisit X)
        assert isinstance(result, Variable)
        # The function stops when it detects a cycle
        assert result.name in ("X", "Y")

    def test_variable_bound_to_compound(self):
        """Test dereferencing variable bound to compound term"""
        # Given: Variable bound to compound
        variable = Variable("X")
        compound_term = Compound("parent", [Atom("john"), Atom("mary")])
        bindings = {"X": compound_term}

        # When: Dereferencing
        result = dereference_variable(variable, bindings)

        # Then: Should return compound
        assert isinstance(result, Compound)
        assert result.functor == "parent"

    def test_different_variable_not_in_bindings(self):
        """Test variable that exists in bindings but with different name"""
        # Given: Bindings for different variable
        variable = Variable("X")
        bindings = {"Y": Atom("value")}

        # When: Dereferencing X
        result = dereference_variable(variable, bindings)

        # Then: Should return original X (unbound)
        assert result == variable


class TestGetAllVariables:
    """Test get_all_variables function"""

    def test_empty_list(self):
        """Test with empty term list"""
        # Given: Empty list
        terms = []

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should return empty set
        assert result == set()

    def test_single_atom(self):
        """Test with single atom (no variables)"""
        # Given: List with single atom
        terms = [atom("value")]

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should return empty set
        assert result == set()

    def test_single_variable(self):
        """Test with single variable"""
        # Given: List with single variable
        terms = [var("X")]

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should return set with X
        assert result == {"X"}

    def test_multiple_variables(self):
        """Test with multiple different variables"""
        # Given: List with multiple variables
        terms = [var("X"), var("Y"), var("Z")]

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should return all variables
        assert result == {"X", "Y", "Z"}

    def test_duplicate_variables(self):
        """Test with duplicate variable names"""
        # Given: List with duplicate variables
        terms = [var("X"), var("Y"), var("X"), var("Y")]

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should return unique set
        assert result == {"X", "Y"}

    def test_compound_with_variables(self):
        """Test with compound terms containing variables"""
        # Given: Compound terms with variables
        terms = [
            compound("parent", var("X"), atom("mary")),
            compound("sibling", var("Y"), var("X"))
        ]

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should extract all variables from compounds
        assert result == {"X", "Y"}

    def test_nested_compounds(self):
        """Test with nested compound terms"""
        # Given: Nested compound terms
        inner = compound("inner", var("A"), var("B"))
        outer = compound("outer", inner, var("C"))
        terms = [outer]

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should extract all nested variables
        assert result == {"A", "B", "C"}

    def test_mixed_terms(self):
        """Test with mix of atoms, variables, and compounds"""
        # Given: Mixed terms
        terms = [
            atom("constant"),
            var("X"),
            compound("pred", var("Y"), atom("value"), var("Z"))
        ]

        # When: Getting variables
        result = get_all_variables(terms)

        # Then: Should extract all variables
        assert result == {"X", "Y", "Z"}


class TestIsGroundTerm:
    """Test is_ground_term function"""

    def test_atom_is_ground(self):
        """Test that atoms are ground terms"""
        # Given: An atom
        term = atom("value")

        # When: Checking if ground
        result = is_ground_term(term)

        # Then: Should be ground
        assert result is True

    def test_variable_is_not_ground(self):
        """Test that variables are not ground"""
        # Given: A variable
        term = var("X")

        # When: Checking if ground
        result = is_ground_term(term)

        # Then: Should not be ground
        assert result is False

    def test_compound_with_atoms_is_ground(self):
        """Test compound with only atoms is ground"""
        # Given: Compound with atoms only
        term = compound("parent", atom("john"), atom("mary"))

        # When: Checking if ground
        result = is_ground_term(term)

        # Then: Should be ground
        assert result is True

    def test_compound_with_variable_is_not_ground(self):
        """Test compound with variable is not ground"""
        # Given: Compound with variable
        term = compound("parent", var("X"), atom("mary"))

        # When: Checking if ground
        result = is_ground_term(term)

        # Then: Should not be ground
        assert result is False

    def test_deeply_nested_ground_compound(self):
        """Test deeply nested ground compound"""
        # Given: Deeply nested compound with all atoms
        inner = compound("inner", atom("a"), atom("b"))
        middle = compound("middle", inner, atom("c"))
        outer = compound("outer", middle, atom("d"))

        # When: Checking if ground
        result = is_ground_term(outer)

        # Then: Should be ground
        assert result is True

    def test_deeply_nested_with_variable(self):
        """Test deeply nested compound with a variable"""
        # Given: Deeply nested with one variable
        inner = compound("inner", var("X"), atom("b"))
        middle = compound("middle", inner, atom("c"))
        outer = compound("outer", middle, atom("d"))

        # When: Checking if ground
        result = is_ground_term(outer)

        # Then: Should not be ground
        assert result is False


class TestIntegration:
    """Integration tests combining utility functions"""

    def test_dereference_and_check_ground(self):
        """Test dereferencing then checking if result is ground"""
        # Given: Variable bound to ground term
        variable = Variable("X")
        ground_term = compound("fact", atom("a"), atom("b"))
        bindings = {"X": ground_term}

        # When: Dereferencing and checking
        dereferenced = dereference_variable(variable, bindings)
        is_ground = is_ground_term(dereferenced)

        # Then: Should be ground after dereference
        assert is_ground is True

    def test_dereference_and_check_non_ground(self):
        """Test dereferencing to non-ground term"""
        # Given: Variable bound to non-ground term
        variable = Variable("X")
        non_ground = compound("pred", var("Y"), atom("value"))
        bindings = {"X": non_ground}

        # When: Dereferencing and checking
        dereferenced = dereference_variable(variable, bindings)
        is_ground = is_ground_term(dereferenced)

        # Then: Should not be ground
        assert is_ground is False

    def test_get_variables_from_dereferenced_term(self):
        """Test getting variables from a dereferenced term"""
        # Given: Variable bound to term with variables
        variable = Variable("X")
        bound_term = compound("pred", var("A"), var("B"))
        bindings = {"X": bound_term}

        # When: Dereferencing then getting variables
        dereferenced = dereference_variable(variable, bindings)
        variables = get_all_variables([dereferenced])

        # Then: Should find variables in bound term
        assert variables == {"A", "B"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
