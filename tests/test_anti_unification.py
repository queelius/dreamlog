"""
Tests for DreamLog anti-unification (least general generalization).
"""

import pytest
from dreamlog.terms import Compound, Atom, Variable
from dreamlog.factories import atom, var, compound
from dreamlog.anti_unification import node_count, anti_unify


class TestNodeCount:
    def test_atom(self):
        assert node_count(atom("a")) == 1

    def test_variable(self):
        assert node_count(var("X")) == 1

    def test_compound_no_args(self):
        assert node_count(Compound("f", [])) == 1

    def test_compound_with_args(self):
        assert node_count(compound("f", atom("a"), atom("b"))) == 3

    def test_nested_compound(self):
        inner = compound("g", atom("a"))
        assert node_count(compound("f", inner, atom("b"))) == 4


class TestAntiUnifyTwoTerms:
    def test_identical_atoms(self):
        result = anti_unify(atom("a"), atom("a"))
        assert result.generalized == atom("a")
        assert result.variables_introduced == 0
        assert result.shared_structure == 1.0

    def test_different_atoms(self):
        result = anti_unify(atom("a"), atom("b"))
        assert isinstance(result.generalized, Variable)
        assert result.generalized.name.startswith("_G")
        assert result.variables_introduced == 1
        assert result.shared_structure == 0.0

    def test_same_functor_same_arity(self):
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("a"), atom("c"))
        result = anti_unify(t1, t2)
        assert isinstance(result.generalized, Compound)
        assert result.generalized.functor == "f"
        assert result.generalized.args[0] == atom("a")
        assert isinstance(result.generalized.args[1], Variable)
        assert result.variables_introduced == 1

    def test_same_pair_consistency(self):
        t1 = compound("f", atom("a"), atom("a"))
        t2 = compound("f", atom("b"), atom("b"))
        result = anti_unify(t1, t2)
        g = result.generalized
        assert isinstance(g, Compound)
        assert g.args[0] == g.args[1]  # same variable reused
        assert result.variables_introduced == 1

    def test_different_pairs_distinct(self):
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("b"), atom("a"))
        result = anti_unify(t1, t2)
        g = result.generalized
        assert isinstance(g, Compound)
        assert g.args[0] != g.args[1]
        assert result.variables_introduced == 2

    def test_nested_compound(self):
        t1 = compound("f", compound("g", atom("a")), compound("h", atom("b")))
        t2 = compound("f", compound("g", atom("c")), compound("h", atom("d")))
        result = anti_unify(t1, t2)
        g = result.generalized
        assert g.functor == "f"
        assert g.args[0].functor == "g"
        assert isinstance(g.args[0].args[0], Variable)
        assert g.args[1].functor == "h"
        assert isinstance(g.args[1].args[0], Variable)

    def test_mismatched_functors(self):
        result = anti_unify(compound("f", atom("a")), compound("g", atom("a")))
        assert isinstance(result.generalized, Variable)

    def test_mismatched_arities(self):
        result = anti_unify(compound("f", atom("a"), atom("b")), compound("f", atom("a")))
        assert isinstance(result.generalized, Variable)

    def test_substitution_recovery(self):
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("a"), atom("c"))
        result = anti_unify(t1, t2)
        recovered_1 = result.generalized.substitute(result.substitutions[0])
        recovered_2 = result.generalized.substitute(result.substitutions[1])
        assert recovered_1 == t1
        assert recovered_2 == t2

    def test_shared_structure_score(self):
        t1 = compound("f", atom("a"), atom("b"))
        t2 = compound("f", atom("a"), atom("c"))
        result = anti_unify(t1, t2)
        assert abs(result.shared_structure - 2.0 / 3.0) < 0.01
