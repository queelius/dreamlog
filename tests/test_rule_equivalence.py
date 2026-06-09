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


# Edge cases for the headline proposal-rate metric: constants in the head, the
# bijection's directionality, and not/1 body literals (Op G emits these).

def test_constant_in_head_must_match():
    r1 = Rule(compound("likes", X, atom("john")), [compound("knows", X, atom("john"))])
    r2 = Rule(compound("likes", A, atom("john")), [compound("knows", A, atom("john"))])
    assert rules_structurally_equivalent(r1, r2)
    r3 = Rule(compound("likes", X, atom("mary")), [compound("knows", X, atom("mary"))])
    assert not rules_structurally_equivalent(r1, r3)


def test_repeated_var_not_equivalent_to_distinct_vars():
    # p(X,X):-q(X) shares one variable across both head args; p(A,B):-q(A) does
    # not. No bijection witnesses equivalence, in either direction (symmetry).
    r1 = Rule(compound("p", X, X), [compound("q", X)])
    r2 = Rule(compound("p", A, B), [compound("q", A)])
    assert not rules_structurally_equivalent(r1, r2)
    assert not rules_structurally_equivalent(r2, r1)


def test_negated_body_literal_compared_structurally():
    # not(bad(X)) is a Compound("not", [Compound("bad", [X])]); it must match by
    # structure, so the negated predicate name still matters.
    r1 = Rule(compound("ok", X), [compound("not", compound("bad", X))])
    r2 = Rule(compound("ok", A), [compound("not", compound("bad", A))])
    assert rules_structurally_equivalent(r1, r2)
    r3 = Rule(compound("ok", X), [compound("not", compound("worse", X))])
    assert not rules_structurally_equivalent(r1, r3)
