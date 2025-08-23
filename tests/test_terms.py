"""
Tests for DreamLog term system
"""
import pytest
from dreamlog import atom, var, compound, Atom, Variable, Compound
from dreamlog.prefix_parser import parse_prefix_notation, parse_s_expression


class TestAtom:
    """Test atom functionality"""
    
    def test_creation(self):
        """Test atom creation"""
        a = atom("john")
        assert isinstance(a, Atom)
        assert a.value == "john"
    
    def test_equality(self):
        """Test atom equality"""
        a1 = atom("john")
        a2 = atom("john")
        a3 = atom("mary")
        
        assert a1 == a2
        assert a1 != a3
    
    def test_string_representation(self):
        """Test atom string conversion"""
        a = atom("test")
        assert str(a) == "test"
    
    def test_hash(self):
        """Test atom hashing for use in sets/dicts"""
        a1 = atom("john")
        a2 = atom("john")
        a3 = atom("mary")
        
        atoms_set = {a1, a2, a3}
        assert len(atoms_set) == 2  # john and mary


class TestVariable:
    """Test variable functionality"""
    
    def test_creation(self):
        """Test variable creation"""
        v = var("X")
        assert isinstance(v, Variable)
        assert v.name == "X"
    
    def test_equality(self):
        """Test variable equality"""
        v1 = var("X")
        v2 = var("X")
        v3 = var("Y")
        
        assert v1 == v2
        assert v1 != v3
    
    def test_string_representation(self):
        """Test variable string conversion"""
        v = var("X")
        assert str(v) == "X"
    
    def test_anonymous_variable(self):
        """Test underscore variable"""
        v = var("_")
        assert v.name == "_"


class TestCompound:
    """Test compound term functionality"""
    
    def test_creation(self):
        """Test compound creation"""
        c = compound("parent", atom("john"), atom("mary"))
        assert isinstance(c, Compound)
        assert c.functor == "parent"
        assert c.arity == 2
        assert c.args[0] == atom("john")
        assert c.args[1] == atom("mary")
    
    def test_nested_compound(self):
        """Test nested compound terms"""
        inner = compound("age", atom("john"), atom("42"))
        outer = compound("fact", inner, atom("true"))
        
        assert outer.functor == "fact"
        assert outer.arity == 2
        assert outer.args[0].functor == "age"
    
    def test_equality(self):
        """Test compound equality"""
        c1 = compound("parent", atom("john"), atom("mary"))
        c2 = compound("parent", atom("john"), atom("mary"))
        c3 = compound("parent", atom("john"), atom("alice"))
        
        assert c1 == c2
        assert c1 != c3
    
    def test_string_representation(self):
        """Test compound string conversion"""
        c = compound("parent", atom("john"), var("X"))
        s = str(c)
        assert "parent" in s
        assert "john" in s
        assert "X" in s
    
    def test_zero_arity_compound(self):
        """Test compound with no arguments"""
        c = compound("true")
        assert c.functor == "true"
        assert c.arity == 0
        assert len(c.args) == 0


class TestTermParsing:
    """Test parsing terms from various formats"""
    
    def test_parse_atom_from_sexp(self):
        """Test parsing atom from S-expression"""
        term = parse_s_expression("john")
        assert isinstance(term, Atom)
        assert term.value == "john"
    
    def test_parse_variable_from_sexp(self):
        """Test parsing variable from S-expression"""
        term = parse_s_expression("X")
        assert isinstance(term, Variable)
        assert term.name == "X"
    
    def test_parse_compound_from_sexp(self):
        """Test parsing compound from S-expression"""
        term = parse_s_expression("(parent john mary)")
        assert isinstance(term, Compound)
        assert term.functor == "parent"
        assert term.arity == 2
        assert term.args[0].value == "john"
        assert term.args[1].value == "mary"
    
    def test_parse_nested_from_sexp(self):
        """Test parsing nested compound from S-expression"""
        term = parse_s_expression("(likes john (food pizza))")
        assert term.functor == "likes"
        assert term.args[0].value == "john"
        assert term.args[1].functor == "food"
        assert term.args[1].args[0].value == "pizza"
    
    def test_parse_from_prefix_json(self):
        """Test parsing from JSON prefix notation"""
        term = parse_prefix_notation(["parent", "john", "mary"])
        assert isinstance(term, Compound)
        assert term.functor == "parent"
        assert term.args[0].value == "john"
        assert term.args[1].value == "mary"
    
    def test_parse_with_variables(self):
        """Test parsing with variables"""
        term = parse_prefix_notation(["parent", "X", "mary"])
        assert term.functor == "parent"
        assert isinstance(term.args[0], Variable)
        assert term.args[0].name == "X"
        assert term.args[1].value == "mary"


class TestTermOperations:
    """Test operations on terms"""
    
    def test_get_variables(self):
        """Test extracting variables from terms"""
        c = compound("parent", var("X"), var("Y"))
        vars = c.get_vars()
        assert len(vars) == 2
        assert var("X") in vars
        assert var("Y") in vars
    
    def test_substitute(self):
        """Test substitution in terms"""
        c = compound("parent", var("X"), atom("mary"))
        subst = {var("X"): atom("john")}
        result = c.substitute(subst)
        
        assert result.functor == "parent"
        assert result.args[0] == atom("john")
        assert result.args[1] == atom("mary")
    
    def test_occurs_check(self):
        """Test occurs check for variables in terms"""
        x = var("X")
        c = compound("f", x, atom("a"))
        
        assert x.occurs_in(c)
        assert not atom("b").occurs_in(c)
    
    def test_ground_check(self):
        """Test if term is ground (no variables)"""
        from dreamlog.terms import is_ground
        
        assert is_ground(atom("john"))
        assert is_ground(compound("parent", atom("john"), atom("mary")))
        assert not is_ground(var("X"))
        assert not is_ground(compound("parent", var("X"), atom("mary")))