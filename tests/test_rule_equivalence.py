from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Rule
from dreamlog.rule_equivalence import rules_structurally_equivalent

X, Y, Z = var("X"), var("Y"), var("Z")
A, B, C = var("A"), var("B"), var("C")


def test_equivalent_under_renaming_and_reorder():
    r1 = Rule(compound("father", X, Y), [compound("parent", X, Y), compound("male", X)])
    # renamed vars + reordered body
    r2 = Rule(compound("father", A, B), [compound("male", A), compound("parent", A, B)])
    assert rules_structurally_equivalent(r1, r2)


def test_right_vs_left_recursion_not_equivalent():
    right = Rule(compound("anc", X, Z), [compound("par", X, Y), compound("anc", Y, Z)])
    left = Rule(compound("anc", X, Z), [compound("anc", X, Y), compound("par", Y, Z)])
    assert not rules_structurally_equivalent(right, left)


def test_different_body_predicate_not_equivalent():
    r1 = Rule(compound("father", X, Y), [compound("parent", X, Y), compound("male", X)])
    r2 = Rule(compound("father", X, Y), [compound("parent", X, Y), compound("female", X)])
    assert not rules_structurally_equivalent(r1, r2)


def test_different_body_length_not_equivalent():
    r1 = Rule(compound("p", X), [compound("q", X)])
    r2 = Rule(compound("p", X), [compound("q", X), compound("r", X)])
    assert not rules_structurally_equivalent(r1, r2)


def test_atom_constants_must_match():
    r1 = Rule(compound("p", X), [compound("q", X, atom("a"))])
    r2 = Rule(compound("p", X), [compound("q", X, atom("b"))])
    assert not rules_structurally_equivalent(r1, r2)
