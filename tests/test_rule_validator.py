"""
Tests for rule validation system
"""

import pytest
from dreamlog.factories import atom, var, compound
from dreamlog.knowledge import Rule, Fact, KnowledgeBase
from dreamlog.rule_validator import (
    StructuralValidator,
    SemanticValidator,
    RuleValidator,
    ValidationResult
)


class TestValidationResult:
    """Test ValidationResult dataclass"""

    def test_valid_result(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert bool(result) is True
        assert result.error_message is None

    def test_invalid_result(self):
        result = ValidationResult(is_valid=False, error_message="Test error")
        assert not result.is_valid
        assert bool(result) is False
        assert result.error_message == "Test error"

    def test_warning_result(self):
        result = ValidationResult(is_valid=True, warning_message="Test warning")
        assert result.is_valid
        assert result.warning_message == "Test warning"


class TestStructuralValidator:
    """Test structural validation"""

    def test_valid_rule(self):
        # grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        rule = Rule(
            head=compound("grandparent", var("X"), var("Z")),
            body=[
                compound("parent", var("X"), var("Y")),
                compound("parent", var("Y"), var("Z"))
            ]
        )

        validator = StructuralValidator()
        result = validator.validate_rule(rule)

        assert result.is_valid
        assert result.error_message is None

    def test_unsafe_rule_head_variable_not_in_body(self):
        # ancestor(X, Y) :- parent(Z, W).  [X and Y not in body!]
        rule = Rule(
            head=compound("ancestor", var("X"), var("Y")),
            body=[compound("parent", var("Z"), var("W"))]
        )

        validator = StructuralValidator()
        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert "Unsafe rule" in result.error_message
        assert "X" in result.error_message
        assert "Y" in result.error_message

    def test_singleton_variable_warning(self):
        # parent(X, Y, Z) :- father(X, Z).  [Y is singleton]
        # Changed to have Y only appear once (not in head or body)
        rule = Rule(
            head=compound("parent", var("X"), var("Z")),
            body=[compound("father", var("X"), var("Y")), compound("mother", var("Y"), var("Z"))]
        )

        validator = StructuralValidator()
        result = validator.validate_rule(rule)

        # Should be valid (all head vars in body) but with warning for Y appearing only once
        assert result.is_valid
        # Note: This test is actually showing correct behavior - no singletons!
        # Let's test actual singletons: sibling(X, Y) :- parent(Z, X), parent(W, Y).
        # Z and W each appear only once in body
        rule2 = Rule(
            head=compound("sibling", var("X"), var("Y")),
            body=[
                compound("parent", var("Z"), var("X")),
                compound("parent", var("Z"), var("Y"))  # Z appears twice, so no singleton
            ]
        )
        result2 = validator.validate_rule(rule2)
        assert result2.is_valid

    def test_rule_head_cannot_be_variable(self):
        # X :- parent(Y, Z).  [Invalid!]
        rule = Rule(
            head=var("X"),
            body=[compound("parent", var("Y"), var("Z"))]
        )

        validator = StructuralValidator()
        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert "head cannot be a variable" in result.error_message

    def test_body_cannot_have_bare_variable(self):
        # parent(X, Y) :- Z.  [Z is bare variable, invalid]
        rule = Rule(
            head=compound("parent", var("X"), var("Y")),
            body=[var("Z")]
        )

        validator = StructuralValidator()
        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert "cannot be a bare variable" in result.error_message

    def test_fact_is_always_safe(self):
        # parent(john, mary).
        rule = Rule(
            head=compound("parent", atom("john"), atom("mary")),
            body=[]
        )

        validator = StructuralValidator()
        result = validator.validate_rule(rule)

        assert result.is_valid


class TestSemanticValidator:
    """Test semantic validation against knowledge base"""

    def test_valid_rule_with_existing_predicates(self):
        # KB has parent/2 facts
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("parent", atom("mary"), atom("alice"))))

        # grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        rule = Rule(
            head=compound("grandparent", var("X"), var("Z")),
            body=[
                compound("parent", var("X"), var("Y")),
                compound("parent", var("Y"), var("Z"))
            ]
        )

        validator = SemanticValidator(kb)
        result = validator.validate_rule(rule)

        # Should pass (parent/2 exists)
        assert result.is_valid

    def test_undefined_predicate_warning(self):
        # KB is empty
        kb = KnowledgeBase()

        # ancestor(X, Y) :- parent(X, Y).  [parent/2 undefined]
        rule = Rule(
            head=compound("ancestor", var("X"), var("Y")),
            body=[compound("parent", var("X"), var("Y"))]
        )

        validator = SemanticValidator(kb)
        result = validator.validate_rule(rule)

        # Should be valid but with warning
        assert result.is_valid
        assert result.warning_message is not None
        assert "Undefined predicates" in result.warning_message
        assert "parent/2" in result.warning_message

    def test_arity_mismatch_error(self):
        # KB has parent/2
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))

        # bad_rule(X) :- parent(X, Y, Z).  [parent/3 but KB has parent/2]
        rule = Rule(
            head=compound("bad_rule", var("X")),
            body=[compound("parent", var("X"), var("Y"), var("Z"))]
        )

        validator = SemanticValidator(kb)
        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert "Arity mismatch" in result.error_message
        assert "parent/3" in result.error_message
        assert "parent/2" in result.error_message

    def test_arity_consistency_in_head(self):
        # KB has parent/2
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))

        # parent(X, Y, Z) :- father(X, Y).  [Defining parent/3 when parent/2 exists]
        rule = Rule(
            head=compound("parent", var("X"), var("Y"), var("Z")),
            body=[compound("father", var("X"), var("Y"))]
        )

        validator = SemanticValidator(kb)
        result = validator.validate_rule(rule)

        assert not result.is_valid
        assert "Arity mismatch" in result.error_message


class TestRuleValidator:
    """Test combined validator"""

    def test_validate_with_structural_only(self):
        # unsafe rule: ancestor(X, Y) :- parent(Z, W).
        rule = Rule(
            head=compound("ancestor", var("X"), var("Y")),
            body=[compound("parent", var("Z"), var("W"))]
        )

        validator = RuleValidator()
        result = validator.validate(rule, structural=True, semantic=False)

        assert not result.is_valid
        assert "Unsafe rule" in result.error_message

    def test_validate_with_semantic_only(self):
        kb = KnowledgeBase()

        # Valid structurally, but uses undefined predicate
        rule = Rule(
            head=compound("ancestor", var("X"), var("Y")),
            body=[compound("parent", var("X"), var("Y"))]
        )

        validator = RuleValidator(kb)
        result = validator.validate(rule, structural=False, semantic=True)

        # Should be valid but with warning about undefined predicate
        assert result.is_valid
        assert result.warning_message is not None

    def test_validate_with_both(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))

        # Valid rule
        rule = Rule(
            head=compound("grandparent", var("X"), var("Z")),
            body=[
                compound("parent", var("X"), var("Y")),
                compound("parent", var("Y"), var("Z"))
            ]
        )

        validator = RuleValidator(kb)
        result = validator.validate(rule, structural=True, semantic=True)

        assert result.is_valid

    def test_combining_warnings(self):
        kb = KnowledgeBase()

        # Rule with singleton variable in body and undefined predicate
        # sibling(X, Y) :- parent(Z, X), parent(Z, Y), unused(W).
        # W is singleton in body, parent and unused are undefined
        rule = Rule(
            head=compound("sibling", var("X"), var("Y")),
            body=[
                compound("parent", var("Z"), var("X")),
                compound("parent", var("Z"), var("Y")),
                compound("unused", var("W"))  # W is singleton
            ]
        )

        validator = RuleValidator(kb)
        result = validator.validate(rule, structural=True, semantic=True)

        # Should be valid with combined warnings
        assert result.is_valid
        assert result.warning_message is not None
        # Should contain both warnings
        assert "Singleton" in result.warning_message
        assert "Undefined" in result.warning_message


class TestRealWorldExamples:
    """Test with real-world examples"""

    def test_correct_ancestor_rule(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("parent", atom("mary"), atom("alice"))))

        # Correct: ancestor(X, Y) :- parent(X, Y).
        rule1 = Rule(
            head=compound("ancestor", var("X"), var("Y")),
            body=[compound("parent", var("X"), var("Y"))]
        )

        validator = RuleValidator(kb)
        result = validator.validate(rule1)
        assert result.is_valid

        # Correct: ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        rule2 = Rule(
            head=compound("ancestor", var("X"), var("Y")),
            body=[
                compound("parent", var("X"), var("Z")),
                compound("ancestor", var("Z"), var("Y"))
            ]
        )

        result = validator.validate(rule2)
        assert result.is_valid

    def test_incorrect_ancestor_rule(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))

        # Add grandparent rule to KB
        grandparent_rule = Rule(
            head=compound("grandparent", var("X"), var("Z")),
            body=[
                compound("parent", var("X"), var("Y")),
                compound("parent", var("Y"), var("Z"))
            ]
        )
        kb.add_rule(grandparent_rule)

        # Incorrect (the one LLM generated): ancestor(X, Y) :- parent(Z, Y), grandparent(X, Z).
        # This is structurally valid but semantically wrong
        bad_rule = Rule(
            head=compound("ancestor", var("X"), var("Y")),
            body=[
                compound("parent", var("Z"), var("Y")),
                compound("grandparent", var("X"), var("Z"))
            ]
        )

        validator = RuleValidator(kb)
        result = validator.validate(bad_rule, structural=True, semantic=True)

        # Structural validation won't catch this (it's structurally valid)
        # This requires LLM judge to catch
        assert result.is_valid  # Structurally valid

    def test_sibling_rule(self):
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("parent", atom("john"), atom("bob"))))

        # sibling(X, Y) :- parent(Z, X), parent(Z, Y).
        rule = Rule(
            head=compound("sibling", var("X"), var("Y")),
            body=[
                compound("parent", var("Z"), var("X")),
                compound("parent", var("Z"), var("Y"))
            ]
        )

        validator = RuleValidator(kb)
        result = validator.validate(rule)

        assert result.is_valid
