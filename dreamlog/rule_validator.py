"""
Validation system for generated rules and facts.

Provides structural validation, semantic checks, and LLM-based verification.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from .terms import Term, Variable, Compound, Atom
from .knowledge import Rule, Fact


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    error_message: Optional[str] = None
    warning_message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_valid


class StructuralValidator:
    """Validates structural properties of rules"""

    def validate_rule(self, rule: Rule) -> ValidationResult:
        """
        Perform all structural validations on a rule.

        Checks:
        1. Well-formedness (head/body structure)
        2. Variable safety: all head variables must appear in body
        3. No singleton variables (appear only once)
        """
        checks = [
            self._check_well_formed,  # Check well-formedness FIRST
            self._check_variable_safety,
            self._check_singleton_variables,
        ]

        warnings = []
        for check in checks:
            result = check(rule)
            if not result.is_valid:
                return result
            if result.warning_message:
                warnings.append(result.warning_message)

        if warnings:
            return ValidationResult(is_valid=True, warning_message="; ".join(warnings))

        return ValidationResult(is_valid=True)

    def _check_variable_safety(self, rule: Rule) -> ValidationResult:
        """
        Check that all variables in the head appear in the body.

        This is the "safety" or "range-restriction" requirement in logic programming.
        Without this, variables in the head would be unbound.
        """
        if rule.is_fact:
            # Facts don't have bodies, so they trivially satisfy this
            return ValidationResult(is_valid=True)

        head_vars = rule.head.get_variables()
        body_vars: Set[str] = set()
        for term in rule.body:
            body_vars.update(term.get_variables())

        unbound_vars = head_vars - body_vars

        if unbound_vars:
            var_list = ", ".join(sorted(unbound_vars))
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsafe rule: variables {var_list} appear in head but not in body"
            )

        return ValidationResult(is_valid=True)

    def _check_singleton_variables(self, rule: Rule) -> ValidationResult:
        """
        Check for singleton variables (variables that appear exactly once).

        These are usually logic errors, though in Prolog they're allowed.
        We'll issue a warning rather than an error.
        """
        var_counts: Dict[str, int] = {}

        # Count in head
        for var in rule.head.get_variables():
            var_counts[var] = var_counts.get(var, 0) + 1

        # Count in body
        for term in rule.body:
            for var in term.get_variables():
                var_counts[var] = var_counts.get(var, 0) + 1

        singletons = [var for var, count in var_counts.items() if count == 1]

        if singletons:
            var_list = ", ".join(sorted(singletons))
            return ValidationResult(
                is_valid=True,  # Warning, not error
                warning_message=f"Singleton variables: {var_list} (appear only once)"
            )

        return ValidationResult(is_valid=True)

    def _check_well_formed(self, rule: Rule) -> ValidationResult:
        """Check that the rule is well-formed"""
        # Check head is not a variable
        if isinstance(rule.head, Variable):
            return ValidationResult(
                is_valid=False,
                error_message="Rule head cannot be a variable"
            )

        # Check body terms are not bare variables
        for i, term in enumerate(rule.body):
            if isinstance(term, Variable):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Body term {i+1} cannot be a bare variable"
                )

        return ValidationResult(is_valid=True)


class SemanticValidator:
    """Validates semantic properties of rules against a knowledge base"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def validate_rule(self, rule: Rule) -> ValidationResult:
        """
        Perform semantic validation checks.

        Checks:
        1. Predicates used in body exist in KB or are built-in
        2. Arity consistency for predicates
        3. No circular dependencies (optional, can be disabled)
        """
        checks = [
            self._check_predicate_existence,
            self._check_arity_consistency,
        ]

        warnings = []
        for check in checks:
            result = check(rule)
            if not result.is_valid:
                return result
            if result.warning_message:
                warnings.append(result.warning_message)

        if warnings:
            return ValidationResult(is_valid=True, warning_message="; ".join(warnings))

        return ValidationResult(is_valid=True)

    def _check_predicate_existence(self, rule: Rule) -> ValidationResult:
        """
        Check that predicates used in body exist in the knowledge base.

        This is a warning rather than an error, since we might be building
        up rules incrementally.
        """
        # Get all functors used in the body
        body_functors = set()
        for term in rule.body:
            if isinstance(term, Compound):
                body_functors.add((term.functor, term.arity))
            elif isinstance(term, Atom):
                body_functors.add((term.value, 0))

        # Check which ones are defined
        undefined_predicates = []
        for functor, arity in body_functors:
            # Check if any fact or rule defines this predicate
            has_definition = False

            # Check facts
            for fact in self.kb.facts:
                term = fact.term
                if isinstance(term, Compound) and term.functor == functor and term.arity == arity:
                    has_definition = True
                    break
                elif isinstance(term, Atom) and term.value == functor and arity == 0:
                    has_definition = True
                    break

            # Check rules
            if not has_definition:
                for kb_rule in self.kb.rules:
                    if isinstance(kb_rule.head, Compound) and kb_rule.head.functor == functor and kb_rule.head.arity == arity:
                        has_definition = True
                        break
                    elif isinstance(kb_rule.head, Atom) and kb_rule.head.value == functor and arity == 0:
                        has_definition = True
                        break

            if not has_definition:
                undefined_predicates.append(f"{functor}/{arity}")

        if undefined_predicates:
            pred_list = ", ".join(undefined_predicates)
            return ValidationResult(
                is_valid=True,  # Warning, not error
                warning_message=f"Undefined predicates: {pred_list}"
            )

        return ValidationResult(is_valid=True)

    def _check_arity_consistency(self, rule: Rule) -> ValidationResult:
        """
        Check that predicates are used with consistent arity.

        If parent/2 is defined, we shouldn't see parent/3.
        """
        # Build index of known arities
        known_arities: Dict[str, Set[int]] = {}

        for fact in self.kb.facts:
            term = fact.term
            if isinstance(term, Compound):
                if term.functor not in known_arities:
                    known_arities[term.functor] = set()
                known_arities[term.functor].add(term.arity)
            elif isinstance(term, Atom):
                if term.value not in known_arities:
                    known_arities[term.value] = set()
                known_arities[term.value].add(0)

        for kb_rule in self.kb.rules:
            if isinstance(kb_rule.head, Compound):
                if kb_rule.head.functor not in known_arities:
                    known_arities[kb_rule.head.functor] = set()
                known_arities[kb_rule.head.functor].add(kb_rule.head.arity)
            elif isinstance(kb_rule.head, Atom):
                if kb_rule.head.value not in known_arities:
                    known_arities[kb_rule.head.value] = set()
                known_arities[kb_rule.head.value].add(0)

        # Check new rule for inconsistencies
        inconsistencies = []

        # Check head
        head_functor = None
        head_arity = None
        if isinstance(rule.head, Compound):
            head_functor = rule.head.functor
            head_arity = rule.head.arity
        elif isinstance(rule.head, Atom):
            head_functor = rule.head.value
            head_arity = 0

        if head_functor and head_functor in known_arities:
            if head_arity not in known_arities[head_functor]:
                existing = list(known_arities[head_functor])
                inconsistencies.append(
                    f"{head_functor}/{head_arity} (existing: {head_functor}/{existing[0]})"
                )

        # Check body
        for term in rule.body:
            functor = None
            arity = None
            if isinstance(term, Compound):
                functor = term.functor
                arity = term.arity
            elif isinstance(term, Atom):
                functor = term.value
                arity = 0

            if functor and functor in known_arities:
                if arity not in known_arities[functor]:
                    existing = list(known_arities[functor])
                    inconsistencies.append(
                        f"{functor}/{arity} (existing: {functor}/{existing[0]})"
                    )

        if inconsistencies:
            return ValidationResult(
                is_valid=False,
                error_message=f"Arity mismatch: {', '.join(inconsistencies)}"
            )

        return ValidationResult(is_valid=True)


class RuleValidator:
    """
    Main validator combining structural and semantic checks.
    """

    def __init__(self, knowledge_base=None):
        self.structural_validator = StructuralValidator()
        self.semantic_validator = SemanticValidator(knowledge_base) if knowledge_base is not None else None

    def validate(self, rule: Rule,
                 structural: bool = True,
                 semantic: bool = True) -> ValidationResult:
        """
        Validate a rule with specified checks.

        Args:
            rule: Rule to validate
            structural: Perform structural validation
            semantic: Perform semantic validation (requires knowledge_base)

        Returns:
            ValidationResult with overall result and any messages
        """
        warnings = []

        # Structural validation
        if structural:
            result = self.structural_validator.validate_rule(rule)
            if not result.is_valid:
                return result
            if result.warning_message:
                warnings.append(result.warning_message)

        # Semantic validation
        if semantic and self.semantic_validator:
            result = self.semantic_validator.validate_rule(rule)
            if not result.is_valid:
                return result
            if result.warning_message:
                warnings.append(result.warning_message)

        # Combine warnings
        if warnings:
            return ValidationResult(
                is_valid=True,
                warning_message="; ".join(warnings)
            )

        return ValidationResult(is_valid=True)
