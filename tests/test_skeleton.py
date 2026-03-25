# tests/test_skeleton.py
import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Rule
from dreamlog.skeleton import extract_skeleton, RuleSkeleton, RuleSetSkeleton


class TestSkeletonExtraction:
    def test_single_non_recursive_rule(self):
        rules = [Rule(compound("ancestor", var("X"), var("Y")),
                      [compound("parent", var("X"), var("Y"))])]
        skeleton, fmap = extract_skeleton("ancestor", rules)
        assert skeleton.param_count == 1
        assert fmap["PARAM_0"] == "parent"
        assert len(skeleton.rules) == 1

    def test_recursive_rule_self_detected(self):
        rules = [
            Rule(compound("anc", var("X"), var("Y")),
                 [compound("par", var("X"), var("Y"))]),
            Rule(compound("anc", var("X"), var("Z")),
                 [compound("par", var("X"), var("Y")),
                  compound("anc", var("Y"), var("Z"))]),
        ]
        skeleton, fmap = extract_skeleton("anc", rules)
        assert skeleton.param_count == 1
        assert fmap["PARAM_0"] == "par"
        recursive_skel = [r for r in skeleton.rules if len(r.body) == 2][0]
        body_roles = [role for role, _ in recursive_skel.body]
        assert "SELF" in body_roles
        assert "PARAM_0" in body_roles

    def test_identical_rule_sets_same_skeleton(self):
        rules_a = [
            Rule(compound("ancestor", var("X"), var("Y")),
                 [compound("parent", var("X"), var("Y"))]),
            Rule(compound("ancestor", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")),
                  compound("ancestor", var("Y"), var("Z"))]),
        ]
        rules_b = [
            Rule(compound("reachable", var("A"), var("B")),
                 [compound("edge", var("A"), var("B"))]),
            Rule(compound("reachable", var("A"), var("C")),
                 [compound("edge", var("A"), var("B")),
                  compound("reachable", var("B"), var("C"))]),
        ]
        skel_a, fmap_a = extract_skeleton("ancestor", rules_a)
        skel_b, fmap_b = extract_skeleton("reachable", rules_b)
        assert skel_a == skel_b
        assert fmap_a["PARAM_0"] == "parent"
        assert fmap_b["PARAM_0"] == "edge"

    def test_different_arity_different_skeleton(self):
        rules_a = [Rule(compound("f", var("X")), [compound("g", var("X"))])]
        rules_b = [Rule(compound("f", var("X"), var("Y")),
                        [compound("g", var("X"), var("Y"))])]
        skel_a, _ = extract_skeleton("f", rules_a)
        skel_b, _ = extract_skeleton("f", rules_b)
        assert skel_a != skel_b

    def test_different_body_length_different_skeleton(self):
        rules_a = [Rule(compound("f", var("X")), [compound("g", var("X"))])]
        rules_b = [Rule(compound("f", var("X")),
                        [compound("g", var("X")), compound("h", var("X"))])]
        skel_a, _ = extract_skeleton("f", rules_a)
        skel_b, _ = extract_skeleton("f", rules_b)
        assert skel_a != skel_b

    def test_different_variable_connectivity(self):
        rules_a = [Rule(compound("f", var("X"), var("Y")),
                        [compound("g", var("X"), var("Y"))])]
        rules_b = [Rule(compound("f", var("X"), var("Y")),
                        [compound("g", var("Y"), var("X"))])]
        skel_a, _ = extract_skeleton("f", rules_a)
        skel_b, _ = extract_skeleton("f", rules_b)
        assert skel_a != skel_b

    def test_rule_order_independent(self):
        base = Rule(compound("f", var("X"), var("Y")),
                    [compound("g", var("X"), var("Y"))])
        recursive = Rule(compound("f", var("X"), var("Z")),
                         [compound("g", var("X"), var("Y")),
                          compound("f", var("Y"), var("Z"))])
        skel_a, _ = extract_skeleton("f", [base, recursive])
        skel_b, _ = extract_skeleton("f", [recursive, base])
        assert skel_a == skel_b

    def test_same_param_in_multiple_body_goals(self):
        rules = [Rule(compound("related", var("X"), var("Z")),
                      [compound("friend", var("X"), var("Y")),
                       compound("friend", var("Y"), var("Z"))])]
        skeleton, fmap = extract_skeleton("related", rules)
        assert fmap["PARAM_0"] == "friend"
        skel_rule = skeleton.rules[0]
        assert all(role == "PARAM_0" for role, _ in skel_rule.body)

    def test_not_in_body_is_opaque(self):
        rules = [Rule(compound("f", var("X")),
                      [compound("g", var("X")),
                       compound("not", compound("h", var("X")))])]
        skeleton, fmap = extract_skeleton("f", rules)
        # g -> PARAM_0, not -> PARAM_1 (not is treated as opaque functor)
        assert skeleton.param_count == 2

    def test_skeleton_is_hashable(self):
        rules = [Rule(compound("f", var("X")), [compound("g", var("X"))])]
        skel, _ = extract_skeleton("f", rules)
        d = {skel: "test"}
        assert d[skel] == "test"
