from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Rule
from dreamlog.unification import clause_subsumes


class TestClauseSubsumes:
    def test_general_rule_subsumes_specific(self):
        general = Rule(compound("anc", var("X"), var("Y")),
                       [compound("par", var("X"), var("Y"))])
        specific = Rule(compound("anc", atom("john"), var("Y")),
                        [compound("par", atom("john"), var("Y"))])
        assert clause_subsumes(general, specific) is True
        assert clause_subsumes(specific, general) is False

    def test_identical_rules(self):
        r = Rule(compound("a", var("X")), [compound("b", var("X"))])
        assert clause_subsumes(r, r) is True

    def test_different_body_length(self):
        r1 = Rule(compound("a", var("X")), [compound("b", var("X"))])
        r2 = Rule(compound("a", var("X")),
                  [compound("b", var("X")), compound("c", var("X"))])
        assert clause_subsumes(r1, r2) is False
        assert clause_subsumes(r2, r1) is False

    def test_bodyless_rules(self):
        general = Rule(compound("a", var("X")), [])
        specific = Rule(compound("a", atom("hello")), [])
        assert clause_subsumes(general, specific) is True

    def test_no_subsumption_different_head(self):
        r1 = Rule(compound("a", var("X")), [compound("b", var("X"))])
        r2 = Rule(compound("c", var("X")), [compound("b", var("X"))])
        assert clause_subsumes(r1, r2) is False

    def test_cross_body_variable_binding_inconsistency(self):
        """p(X) :- q(X, Y), r(Y) should NOT subsume p(a) :- q(a, b), r(c)
        because Y=b from goal 1 and Y=c from goal 2 are inconsistent."""
        general = Rule(compound("p", var("X")),
                       [compound("q", var("X"), var("Y")),
                        compound("r", var("Y"))])
        specific = Rule(compound("p", atom("a")),
                        [compound("q", atom("a"), atom("b")),
                         compound("r", atom("c"))])
        assert clause_subsumes(general, specific) is False

    def test_cross_body_variable_binding_consistent(self):
        """p(X) :- q(X, Y), r(Y) SHOULD subsume p(a) :- q(a, b), r(b)
        because Y=b is consistent across both goals."""
        general = Rule(compound("p", var("X")),
                       [compound("q", var("X"), var("Y")),
                        compound("r", var("Y"))])
        specific = Rule(compound("p", atom("a")),
                        [compound("q", atom("a"), atom("b")),
                         compound("r", atom("b"))])
        assert clause_subsumes(general, specific) is True
