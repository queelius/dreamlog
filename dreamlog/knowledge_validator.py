"""
Knowledge Validation System for DreamLog Persistent Learning

This module implements comprehensive validation mechanisms to ensure knowledge base
consistency and correctness. It validates that KB changes preserve correctness
while enabling generalization.

Key Components:
- KnowledgeValidator: Main validation engine
- ValidationTest: Individual test cases for validation
- ValidationResult: Results of validation tests
- SampleQueryGenerator: Generate test queries for validation
"""

import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Iterator, Callable

from .knowledge import KnowledgeBase, Fact, Rule
from .terms import Term, Variable, Atom, Compound
from .evaluator import PrologEvaluator, Solution
from .factories import atom, var, compound
from .unification import unify

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation failures"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationTestType(Enum):
    """Types of validation tests"""
    CONSISTENCY = "consistency"  # KB is logically consistent
    COMPLETENESS = "completeness"  # KB can answer expected queries
    CORRECTNESS = "correctness"  # KB gives correct answers
    PRESERVATION = "preservation"  # New KB preserves old behavior
    GENERALIZATION = "generalization"  # KB can generalize appropriately


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    test_type: ValidationTestType
    passed: bool
    severity: ValidationSeverity
    message: str
    expected: Any = None
    actual: Any = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    kb_hash: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[ValidationResult]
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Get the success rate as a percentage"""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100.0
    
    @property
    def critical_failures(self) -> List[ValidationResult]:
        """Get all critical failures"""
        return [r for r in self.results if not r.passed and r.severity == ValidationSeverity.CRITICAL]
    
    @property
    def errors(self) -> List[ValidationResult]:
        """Get all error-level failures"""
        return [r for r in self.results if not r.passed and r.severity == ValidationSeverity.ERROR]


class ValidationTest(ABC):
    """Abstract base class for validation tests"""
    
    def __init__(self, name: str, test_type: ValidationTestType, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.test_type = test_type
        self.severity = severity
    
    @abstractmethod
    def run(self, kb: KnowledgeBase, baseline_kb: Optional[KnowledgeBase] = None) -> ValidationResult:
        """Run the validation test"""
        pass


class ConsistencyTest(ValidationTest):
    """Test that the knowledge base is logically consistent"""
    
    def __init__(self):
        super().__init__("Logical Consistency", ValidationTestType.CONSISTENCY, ValidationSeverity.CRITICAL)
    
    def run(self, kb: KnowledgeBase, baseline_kb: Optional[KnowledgeBase] = None) -> ValidationResult:
        start_time = datetime.now()
        
        try:
            evaluator = PrologEvaluator(kb)
            
            # Check for direct contradictions
            contradictions = self._find_contradictions(kb, evaluator)
            
            if contradictions:
                return ValidationResult(
                    test_name=self.name,
                    test_type=self.test_type,
                    passed=False,
                    severity=self.severity,
                    message=f"Found {len(contradictions)} logical contradictions",
                    actual=contradictions,
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    metadata={"contradictions": contradictions}
                )
            
            return ValidationResult(
                test_name=self.name,
                test_type=self.test_type,
                passed=True,
                severity=self.severity,
                message="Knowledge base is logically consistent",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        
        except Exception as e:
            return ValidationResult(
                test_name=self.name,
                test_type=self.test_type,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Consistency test failed with error: {str(e)}",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                metadata={"error": str(e)}
            )
    
    def _find_contradictions(self, kb: KnowledgeBase, evaluator: PrologEvaluator) -> List[str]:
        """Find logical contradictions in the knowledge base"""
        contradictions = []
        
        # Simple contradiction detection: look for facts that directly contradict
        facts = [fact.term for fact in kb.facts]
        
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                if self._are_contradictory(fact1, fact2):
                    contradictions.append(f"{fact1} contradicts {fact2}")
        
        # Check for rule-based contradictions
        # This is simplified - real contradiction detection would be more sophisticated
        
        return contradictions
    
    def _are_contradictory(self, term1: Term, term2: Term) -> bool:
        """Check if two terms are contradictory"""
        if isinstance(term1, Compound) and isinstance(term2, Compound):
            # Check for explicit negation
            if (term1.functor == "not" and len(term1.args) == 1 and term1.args[0] == term2):
                return True
            if (term2.functor == "not" and len(term2.args) == 1 and term2.args[0] == term1):
                return True
            
            # Domain-specific contradictions
            contradictory_pairs = [
                ("alive", "dead"),
                ("true", "false"),
                ("present", "absent"),
                ("on", "off"),
                ("open", "closed")
            ]
            
            for pos_pred, neg_pred in contradictory_pairs:
                if ((term1.functor == pos_pred and term2.functor == neg_pred) or
                    (term1.functor == neg_pred and term2.functor == pos_pred)):
                    # Same arguments?
                    if term1.args == term2.args:
                        return True
        
        return False


class CompletenessTest(ValidationTest):
    """Test that the knowledge base can answer expected queries"""
    
    def __init__(self, test_queries: List[Tuple[List[Term], int]]):
        """
        Args:
            test_queries: List of (query_terms, expected_solution_count) pairs
        """
        super().__init__("Query Completeness", ValidationTestType.COMPLETENESS, ValidationSeverity.WARNING)
        self.test_queries = test_queries
    
    def run(self, kb: KnowledgeBase, baseline_kb: Optional[KnowledgeBase] = None) -> ValidationResult:
        start_time = datetime.now()
        
        try:
            evaluator = PrologEvaluator(kb)
            failed_queries = []
            
            for query_terms, expected_count in self.test_queries:
                solutions = list(evaluator.query(query_terms))
                if len(solutions) != expected_count:
                    failed_queries.append({
                        "query": [str(term) for term in query_terms],
                        "expected_solutions": expected_count,
                        "actual_solutions": len(solutions)
                    })
            
            if failed_queries:
                return ValidationResult(
                    test_name=self.name,
                    test_type=self.test_type,
                    passed=False,
                    severity=self.severity,
                    message=f"{len(failed_queries)} queries returned unexpected number of solutions",
                    expected=[q["expected_solutions"] for q in failed_queries],
                    actual=[q["actual_solutions"] for q in failed_queries],
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    metadata={"failed_queries": failed_queries}
                )
            
            return ValidationResult(
                test_name=self.name,
                test_type=self.test_type,
                passed=True,
                severity=self.severity,
                message=f"All {len(self.test_queries)} test queries returned expected results",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        
        except Exception as e:
            return ValidationResult(
                test_name=self.name,
                test_type=self.test_type,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Completeness test failed with error: {str(e)}",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                metadata={"error": str(e)}
            )


class PreservationTest(ValidationTest):
    """Test that new KB preserves behavior of baseline KB"""
    
    def __init__(self, sample_queries: List[List[Term]], tolerance: float = 0.0):
        """
        Args:
            sample_queries: List of queries to test
            tolerance: Allowed difference in solution counts (0.0 = exact match)
        """
        super().__init__("Behavior Preservation", ValidationTestType.PRESERVATION, ValidationSeverity.ERROR)
        self.sample_queries = sample_queries
        self.tolerance = tolerance
    
    def run(self, kb: KnowledgeBase, baseline_kb: Optional[KnowledgeBase] = None) -> ValidationResult:
        if baseline_kb is None:
            return ValidationResult(
                test_name=self.name,
                test_type=self.test_type,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Baseline knowledge base required for preservation test",
                execution_time_ms=0.0
            )
        
        start_time = datetime.now()
        
        try:
            new_evaluator = PrologEvaluator(kb)
            baseline_evaluator = PrologEvaluator(baseline_kb)
            
            behavior_changes = []
            
            for query_terms in self.sample_queries:
                # Get solutions from both KBs
                new_solutions = list(new_evaluator.query(query_terms))
                baseline_solutions = list(baseline_evaluator.query(query_terms))
                
                # Compare solution counts
                new_count = len(new_solutions)
                baseline_count = len(baseline_solutions)
                
                if abs(new_count - baseline_count) > self.tolerance:
                    behavior_changes.append({
                        "query": [str(term) for term in query_terms],
                        "baseline_solutions": baseline_count,
                        "new_solutions": new_count,
                        "difference": new_count - baseline_count
                    })
            
            if behavior_changes:
                return ValidationResult(
                    test_name=self.name,
                    test_type=self.test_type,
                    passed=False,
                    severity=self.severity,
                    message=f"{len(behavior_changes)} queries show behavior changes",
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    metadata={"behavior_changes": behavior_changes}
                )
            
            return ValidationResult(
                test_name=self.name,
                test_type=self.test_type,
                passed=True,
                severity=self.severity,
                message=f"All {len(self.sample_queries)} test queries preserved baseline behavior",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
        
        except Exception as e:
            return ValidationResult(
                test_name=self.name,
                test_type=self.test_type,
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Preservation test failed with error: {str(e)}",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                metadata={"error": str(e)}
            )


class SampleQueryGenerator:
    """Generate sample queries for validation testing"""
    
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
    
    def generate_fact_queries(self, count: int = 10) -> List[List[Term]]:
        """Generate queries based on existing facts"""
        queries = []
        facts = list(self.kb.facts)
        
        if not facts:
            return queries
        
        # Sample facts and create queries with variables
        sample_facts = random.sample(facts, min(count, len(facts)))
        
        for fact in sample_facts:
            if isinstance(fact.term, Compound):
                # Create query with some arguments replaced by variables
                query_terms = []
                for i, arg in enumerate(fact.term.args):
                    if isinstance(arg, Atom) and random.random() < 0.5:
                        # Replace with variable
                        query_terms.append(var(f"X{i}"))
                    else:
                        query_terms.append(arg)
                
                query = compound(fact.term.functor, *query_terms)
                queries.append([query])
        
        return queries
    
    def generate_rule_queries(self, count: int = 10) -> List[List[Term]]:
        """Generate queries based on rule heads"""
        queries = []
        rules = list(self.kb.rules)
        
        if not rules:
            return queries
        
        # Sample rules and create queries for their heads
        sample_rules = random.sample(rules, min(count, len(rules)))
        
        for rule in sample_rules:
            if isinstance(rule.head, Compound):
                # Create query with variables
                query_terms = []
                for i, arg in enumerate(rule.head.args):
                    query_terms.append(var(f"X{i}"))
                
                query = compound(rule.head.functor, *query_terms)
                queries.append([query])
        
        return queries
    
    def generate_composite_queries(self, count: int = 5) -> List[List[Term]]:
        """Generate composite queries with multiple goals"""
        queries = []
        facts = list(self.kb.facts)
        
        if len(facts) < 2:
            return queries
        
        for _ in range(count):
            # Pick 2-3 random facts and create a composite query
            sample_size = min(random.randint(2, 3), len(facts))
            sample_facts = random.sample(facts, sample_size)
            
            query_goals = []
            shared_vars = {}
            
            for i, fact in enumerate(sample_facts):
                if isinstance(fact.term, Compound):
                    # Create query with shared variables
                    query_args = []
                    for j, arg in enumerate(fact.term.args):
                        if isinstance(arg, Atom):
                            var_name = f"X{len(shared_vars)}"
                            if random.random() < 0.3:  # 30% chance to share variable
                                if arg.value not in shared_vars:
                                    shared_vars[arg.value] = var_name
                                query_args.append(var(shared_vars[arg.value]))
                            else:
                                query_args.append(var(var_name))
                                shared_vars[f"{i}_{j}"] = var_name
                        else:
                            query_args.append(arg)
                    
                    query_goals.append(compound(fact.term.functor, *query_args))
            
            if query_goals:
                queries.append(query_goals)
        
        return queries


class KnowledgeValidator:
    """
    Main validation engine for knowledge bases
    
    Provides comprehensive validation to ensure KB changes preserve correctness
    while enabling appropriate generalization.
    """
    
    def __init__(self):
        self.tests: List[ValidationTest] = []
        self.query_generator: Optional[SampleQueryGenerator] = None
    
    def add_test(self, test: ValidationTest) -> None:
        """Add a validation test"""
        self.tests.append(test)
    
    def create_standard_tests(self, kb: KnowledgeBase) -> None:
        """Create standard validation tests for a knowledge base"""
        self.query_generator = SampleQueryGenerator(kb)
        
        # Clear existing tests
        self.tests.clear()
        
        # Add consistency test
        self.tests.append(ConsistencyTest())
        
        # Add completeness tests based on existing knowledge
        fact_queries = self.query_generator.generate_fact_queries(10)
        rule_queries = self.query_generator.generate_rule_queries(5)
        
        test_queries = []
        for query in fact_queries + rule_queries:
            # Get expected solution count from current KB
            evaluator = PrologEvaluator(kb)
            solutions = list(evaluator.query(query))
            test_queries.append((query, len(solutions)))
        
        if test_queries:
            self.tests.append(CompletenessTest(test_queries))
        
        # Add preservation test with sample queries
        sample_queries = (self.query_generator.generate_fact_queries(5) + 
                         self.query_generator.generate_rule_queries(5) +
                         self.query_generator.generate_composite_queries(3))
        
        # Flatten single-goal queries
        flattened_queries = []
        for query in sample_queries:
            if len(query) == 1:
                flattened_queries.append(query)
            else:
                flattened_queries.append(query)
        
        if flattened_queries:
            self.tests.append(PreservationTest(flattened_queries))
    
    def validate(self, kb: KnowledgeBase, baseline_kb: Optional[KnowledgeBase] = None) -> ValidationReport:
        """Run all validation tests on a knowledge base"""
        start_time = datetime.now()
        
        results = []
        for test in self.tests:
            try:
                result = test.run(kb, baseline_kb)
                results.append(result)
            except Exception as e:
                # Create error result for failed test
                results.append(ValidationResult(
                    test_name=test.name,
                    test_type=test.test_type,
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Test execution failed: {str(e)}",
                    metadata={"error": str(e)}
                ))
        
        # Calculate report metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Calculate KB hash for tracking
        import hashlib
        kb_hash = hashlib.sha256(kb.to_prefix().encode()).hexdigest()[:16]
        
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        report = ValidationReport(
            kb_hash=kb_hash,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            results=results,
            execution_time_ms=execution_time
        )
        
        logger.info(f"Validation completed: {passed_tests}/{total_tests} tests passed "
                   f"({report.success_rate:.1f}%) in {execution_time:.1f}ms")
        
        return report
    
    def validate_knowledge_change(self, 
                                  original_kb: KnowledgeBase, 
                                  modified_kb: KnowledgeBase) -> ValidationReport:
        """Validate that knowledge base changes preserve correctness"""
        
        # Create tests based on original KB
        self.create_standard_tests(original_kb)
        
        # Run validation on modified KB with original as baseline
        return self.validate(modified_kb, original_kb)
    
    def quick_validate(self, kb: KnowledgeBase) -> bool:
        """Quick validation check - returns True if basic tests pass"""
        consistency_test = ConsistencyTest()
        result = consistency_test.run(kb)
        
        return result.passed