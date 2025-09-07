"""
Sleep Cycle Manager for DreamLog Persistent Learning

This module implements the sleep cycle system that performs knowledge compression,
generalization, and optimization during idle periods. The sleep cycle helps
consolidate learned knowledge and improve system performance.

Key Components:
- SleepCycleManager: Main orchestrator for sleep operations
- KnowledgeCompressor: Removes redundant knowledge
- RuleGeneralizer: Creates more general rules from specific facts
- KnowledgeOptimizer: Optimizes KB structure for better performance
- SleepCycleScheduler: Manages when and how often sleep cycles run
"""

import time
import logging
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Callable, Iterator

from .knowledge import KnowledgeBase, Fact, Rule
from .terms import Term, Variable, Atom, Compound
from .evaluator import PrologEvaluator, Solution
from .unification import unify, match
from .factories import atom, var, compound
from .knowledge_validator import KnowledgeValidator, ValidationReport
from .persistent_learning import PersistentKnowledgeBase

logger = logging.getLogger(__name__)


class SleepPhase(Enum):
    """Phases of the sleep cycle"""
    LIGHT_SLEEP = "light_sleep"      # Basic cleanup and optimization
    DEEP_SLEEP = "deep_sleep"        # Rule generalization and consolidation
    REM_SLEEP = "rem_sleep"          # Creative rule generation and hypothesis formation


class CompressionStrategy(Enum):
    """Strategies for knowledge compression"""
    REMOVE_DUPLICATES = "remove_duplicates"
    SUBSUMPTION_BASED = "subsumption_based"
    FREQUENCY_BASED = "frequency_based"
    UTILITY_BASED = "utility_based"


@dataclass
class SleepCycleConfig:
    """Configuration for sleep cycle behavior"""
    # Scheduling
    light_sleep_interval: timedelta = timedelta(minutes=30)
    deep_sleep_interval: timedelta = timedelta(hours=4)
    rem_sleep_interval: timedelta = timedelta(hours=8)
    
    # Thresholds
    min_idle_time: timedelta = timedelta(minutes=5)
    max_knowledge_growth: float = 0.2  # Trigger sleep if KB grows by 20%
    min_compression_ratio: float = 0.05  # Minimum compression to be worthwhile
    
    # Limits
    max_sleep_duration: timedelta = timedelta(minutes=10)
    max_facts_to_compress: int = 1000
    max_rules_to_generate: int = 50
    
    # Safety
    require_validation: bool = True
    backup_before_sleep: bool = True


@dataclass
class SleepCycleReport:
    """Report from a completed sleep cycle"""
    cycle_id: str
    phase: SleepPhase
    start_time: datetime
    end_time: datetime
    
    # Before/after metrics
    initial_facts: int
    initial_rules: int
    final_facts: int
    final_rules: int
    
    # Operations performed
    operations: List[str]
    facts_removed: int = 0
    rules_removed: int = 0
    rules_generated: int = 0
    facts_generalized: int = 0
    
    # Quality metrics
    compression_ratio: float = 0.0
    validation_report: Optional[ValidationReport] = None
    performance_improvement: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time
    
    @property
    def net_knowledge_change(self) -> int:
        return (self.final_facts + self.final_rules) - (self.initial_facts + self.initial_rules)


class SleepOperation(ABC):
    """Abstract base class for sleep cycle operations"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, kb: KnowledgeBase, config: SleepCycleConfig) -> Tuple[KnowledgeBase, Dict[str, Any]]:
        """
        Execute the sleep operation
        
        Returns:
            Tuple of (modified_kb, operation_metrics)
        """
        pass


class DuplicateRemover(SleepOperation):
    """Remove duplicate facts and rules"""
    
    def __init__(self):
        super().__init__("Remove Duplicates")
    
    def execute(self, kb: KnowledgeBase, config: SleepCycleConfig) -> Tuple[KnowledgeBase, Dict[str, Any]]:
        new_kb = KnowledgeBase()
        
        # Remove duplicate facts
        seen_facts = set()
        unique_facts = []
        for fact in kb.facts:
            fact_str = str(fact.term)
            if fact_str not in seen_facts:
                seen_facts.add(fact_str)
                unique_facts.append(fact)
        
        # Remove duplicate rules
        seen_rules = set()
        unique_rules = []
        for rule in kb.rules:
            rule_str = str(rule)
            if rule_str not in seen_rules:
                seen_rules.add(rule_str)
                unique_rules.append(rule)
        
        # Build new KB
        for fact in unique_facts:
            new_kb.add_fact(fact)
        for rule in unique_rules:
            new_kb.add_rule(rule)
        
        facts_removed = len(kb.facts) - len(unique_facts)
        rules_removed = len(kb.rules) - len(unique_rules)
        
        return new_kb, {
            "facts_removed": facts_removed,
            "rules_removed": rules_removed,
            "total_removed": facts_removed + rules_removed
        }


class SubsumptionCompressor(SleepOperation):
    """Remove facts and rules that are subsumed by more general ones"""
    
    def __init__(self):
        super().__init__("Subsumption Compression")
    
    def execute(self, kb: KnowledgeBase, config: SleepCycleConfig) -> Tuple[KnowledgeBase, Dict[str, Any]]:
        new_kb = KnowledgeBase()
        
        # Find subsumed facts
        facts_to_keep = []
        for i, fact1 in enumerate(kb.facts):
            is_subsumed = False
            for j, fact2 in enumerate(kb.facts):
                if i != j and self._subsumes_fact(fact2.term, fact1.term):
                    is_subsumed = True
                    break
            
            if not is_subsumed:
                facts_to_keep.append(fact1)
        
        # Find subsumed rules
        rules_to_keep = []
        for i, rule1 in enumerate(kb.rules):
            is_subsumed = False
            for j, rule2 in enumerate(kb.rules):
                if i != j and self._subsumes_rule(rule2, rule1):
                    is_subsumed = True
                    break
            
            if not is_subsumed:
                rules_to_keep.append(rule1)
        
        # Build compressed KB
        for fact in facts_to_keep:
            new_kb.add_fact(fact)
        for rule in rules_to_keep:
            new_kb.add_rule(rule)
        
        facts_removed = len(kb.facts) - len(facts_to_keep)
        rules_removed = len(kb.rules) - len(rules_to_keep)
        
        return new_kb, {
            "facts_removed": facts_removed,
            "rules_removed": rules_removed,
            "total_removed": facts_removed + rules_removed
        }
    
    def _subsumes_fact(self, general_term: Term, specific_term: Term) -> bool:
        """Check if general_term subsumes specific_term"""
        # Try to unify - if general can be unified with specific, it subsumes it
        unification = unify(general_term, specific_term)
        return unification is not None
    
    def _subsumes_rule(self, general_rule: Rule, specific_rule: Rule) -> bool:
        """Check if general_rule subsumes specific_rule"""
        # A rule subsumes another if:
        # 1. The head of general can unify with head of specific
        # 2. The body of general is a subset (up to unification) of body of specific
        
        head_unification = unify(general_rule.head, specific_rule.head)
        if head_unification is None:
            return False
        
        # Check if all body goals of general rule can be matched in specific rule
        for general_goal in general_rule.body:
            found_match = False
            for specific_goal in specific_rule.body:
                if unify(general_goal, specific_goal) is not None:
                    found_match = True
                    break
            if not found_match:
                return False
        
        return True


class RuleGeneralizer(SleepOperation):
    """Generate more general rules from specific facts and patterns"""
    
    def __init__(self):
        super().__init__("Rule Generalization")
    
    def execute(self, kb: KnowledgeBase, config: SleepCycleConfig) -> Tuple[KnowledgeBase, Dict[str, Any]]:
        new_kb = KnowledgeBase()
        
        # Copy existing knowledge
        for fact in kb.facts:
            new_kb.add_fact(fact)
        for rule in kb.rules:
            new_kb.add_rule(rule)
        
        # Generate new rules
        generated_rules = []
        
        # Pattern 1: Transitivity rules
        transitivity_rules = self._generate_transitivity_rules(kb)
        generated_rules.extend(transitivity_rules)
        
        # Pattern 2: Symmetry rules
        symmetry_rules = self._generate_symmetry_rules(kb)
        generated_rules.extend(symmetry_rules)
        
        # Pattern 3: Generalization from similar facts
        generalization_rules = self._generate_generalization_rules(kb)
        generated_rules.extend(generalization_rules)
        
        # Add generated rules to KB (up to limit)
        rules_added = 0
        for rule in generated_rules[:config.max_rules_to_generate]:
            # Check if rule already exists
            if not any(self._rules_equivalent(rule, existing_rule) for existing_rule in new_kb.rules):
                new_kb.add_rule(rule)
                rules_added += 1
        
        return new_kb, {
            "rules_generated": rules_added,
            "transitivity_rules": len(transitivity_rules),
            "symmetry_rules": len(symmetry_rules),
            "generalization_rules": len(generalization_rules)
        }
    
    def _generate_transitivity_rules(self, kb: KnowledgeBase) -> List[Rule]:
        """Generate transitivity rules like: rel(X,Z) :- rel(X,Y), rel(Y,Z)"""
        rules = []
        
        # Find binary relations that might be transitive
        binary_relations = defaultdict(list)
        for fact in kb.facts:
            if isinstance(fact.term, Compound) and len(fact.term.args) == 2:
                relation = fact.term.functor
                args = fact.term.args
                if isinstance(args[0], Atom) and isinstance(args[1], Atom):
                    binary_relations[relation].append((args[0].value, args[1].value))
        
        # Check for transitivity patterns
        for relation, pairs in binary_relations.items():
            if len(pairs) >= 3:  # Need at least 3 facts to infer transitivity
                # Look for chains: (a,b), (b,c) -> should have (a,c)
                chains_found = 0
                for a, b in pairs:
                    for b2, c in pairs:
                        if b == b2:  # Found potential chain
                            # Check if (a,c) exists
                            if (a, c) in pairs:
                                chains_found += 1
                
                # If we found multiple chains, create transitivity rule
                if chains_found >= 2:
                    rule = Rule(
                        compound(relation, var("X"), var("Z")),
                        [
                            compound(relation, var("X"), var("Y")),
                            compound(relation, var("Y"), var("Z"))
                        ]
                    )
                    rules.append(rule)
        
        return rules
    
    def _generate_symmetry_rules(self, kb: KnowledgeBase) -> List[Rule]:
        """Generate symmetry rules like: rel(Y,X) :- rel(X,Y)"""
        rules = []
        
        # Find binary relations that might be symmetric
        binary_relations = defaultdict(list)
        for fact in kb.facts:
            if isinstance(fact.term, Compound) and len(fact.term.args) == 2:
                relation = fact.term.functor
                args = fact.term.args
                if isinstance(args[0], Atom) and isinstance(args[1], Atom):
                    binary_relations[relation].append((args[0].value, args[1].value))
        
        # Check for symmetry patterns
        for relation, pairs in binary_relations.items():
            symmetric_count = 0
            for a, b in pairs:
                if (b, a) in pairs:
                    symmetric_count += 1
            
            # If most pairs are symmetric, create symmetry rule
            if symmetric_count >= len(pairs) * 0.7:  # 70% threshold
                rule = Rule(
                    compound(relation, var("Y"), var("X")),
                    [compound(relation, var("X"), var("Y"))]
                )
                rules.append(rule)
        
        return rules
    
    def _generate_generalization_rules(self, kb: KnowledgeBase) -> List[Rule]:
        """Generate rules by generalizing from common patterns in facts"""
        rules = []
        
        # Group facts by functor
        facts_by_functor = defaultdict(list)
        for fact in kb.facts:
            if isinstance(fact.term, Compound):
                facts_by_functor[fact.term.functor].append(fact.term)
        
        # Look for patterns within each functor group
        for functor, terms in facts_by_functor.items():
            if len(terms) >= 3:  # Need multiple facts to generalize
                # Find common patterns
                patterns = self._find_common_patterns(terms)
                for pattern in patterns:
                    rule = self._create_rule_from_pattern(functor, pattern)
                    if rule:
                        rules.append(rule)
        
        return rules
    
    def _find_common_patterns(self, terms: List[Term]) -> List[Tuple[List[Term], Term]]:
        """Find common patterns that could become rule bodies"""
        patterns = []
        
        # This is a simplified pattern finding - could be much more sophisticated
        # For now, just look for terms that share common arguments
        
        return patterns  # Simplified implementation
    
    def _create_rule_from_pattern(self, functor: str, pattern: Tuple[List[Term], Term]) -> Optional[Rule]:
        """Create a rule from a discovered pattern"""
        # Simplified implementation
        return None
    
    def _rules_equivalent(self, rule1: Rule, rule2: Rule) -> bool:
        """Check if two rules are equivalent"""
        return str(rule1) == str(rule2)  # Simplified check


class KnowledgeOptimizer(SleepOperation):
    """Optimize knowledge base structure for better performance"""
    
    def __init__(self):
        super().__init__("Knowledge Optimization")
    
    def execute(self, kb: KnowledgeBase, config: SleepCycleConfig) -> Tuple[KnowledgeBase, Dict[str, Any]]:
        new_kb = KnowledgeBase()
        
        # Sort facts by functor for better indexing
        sorted_facts = sorted(kb.facts, key=lambda f: self._get_sort_key(f.term))
        
        # Sort rules by head functor
        sorted_rules = sorted(kb.rules, key=lambda r: self._get_sort_key(r.head))
        
        # Add sorted knowledge to new KB
        for fact in sorted_facts:
            new_kb.add_fact(fact)
        for rule in sorted_rules:
            new_kb.add_rule(rule)
        
        return new_kb, {
            "facts_reordered": len(sorted_facts),
            "rules_reordered": len(sorted_rules)
        }
    
    def _get_sort_key(self, term: Term) -> str:
        """Get sorting key for a term"""
        if isinstance(term, Compound):
            return f"{term.functor}_{len(term.args)}"
        elif isinstance(term, Atom):
            return f"atom_{term.value}"
        else:
            return "variable"


class SleepCycleManager:
    """
    Main orchestrator for sleep cycle operations
    
    Manages the scheduling and execution of different sleep phases to compress,
    generalize, and optimize the knowledge base during idle periods.
    """
    
    def __init__(self, 
                 persistent_kb: PersistentKnowledgeBase,
                 config: Optional[SleepCycleConfig] = None):
        self.persistent_kb = persistent_kb
        self.config = config or SleepCycleConfig()
        self.validator = KnowledgeValidator()
        
        # State tracking
        self.last_light_sleep = datetime.now()
        self.last_deep_sleep = datetime.now()
        self.last_rem_sleep = datetime.now()
        self.last_activity = datetime.now()
        
        # Sleep operations by phase
        self.light_sleep_ops = [
            DuplicateRemover(),
            KnowledgeOptimizer()
        ]
        
        self.deep_sleep_ops = [
            SubsumptionCompressor(),
            RuleGeneralizer()
        ]
        
        self.rem_sleep_ops = [
            # REM sleep operations would include creative hypothesis generation
            # Not implemented in this version
        ]
        
        # Scheduling
        self.sleep_thread: Optional[threading.Thread] = None
        self.running = False
        self.sleep_history: List[SleepCycleReport] = []
    
    def start_background_sleep(self) -> None:
        """Start background sleep cycle scheduling"""
        if self.sleep_thread and self.sleep_thread.is_alive():
            logger.warning("Sleep cycle already running")
            return
        
        self.running = True
        self.sleep_thread = threading.Thread(target=self._sleep_loop, daemon=True)
        self.sleep_thread.start()
        logger.info("Background sleep cycle started")
    
    def stop_background_sleep(self) -> None:
        """Stop background sleep cycle scheduling"""
        self.running = False
        if self.sleep_thread:
            self.sleep_thread.join(timeout=5.0)
        logger.info("Background sleep cycle stopped")
    
    def trigger_activity(self) -> None:
        """Signal that activity occurred (resets idle timer)"""
        self.last_activity = datetime.now()
    
    def _sleep_loop(self) -> None:
        """Main sleep cycle loop"""
        while self.running:
            try:
                current_time = datetime.now()
                idle_time = current_time - self.last_activity
                
                # Check if we've been idle long enough
                if idle_time >= self.config.min_idle_time:
                    # Determine which sleep phase to run
                    if (current_time - self.last_light_sleep) >= self.config.light_sleep_interval:
                        self._execute_sleep_phase(SleepPhase.LIGHT_SLEEP)
                    elif (current_time - self.last_deep_sleep) >= self.config.deep_sleep_interval:
                        self._execute_sleep_phase(SleepPhase.DEEP_SLEEP)
                    elif (current_time - self.last_rem_sleep) >= self.config.rem_sleep_interval:
                        self._execute_sleep_phase(SleepPhase.REM_SLEEP)
                
                # Sleep for a short interval before checking again
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in sleep loop: {e}")
                time.sleep(60)
    
    def _execute_sleep_phase(self, phase: SleepPhase) -> SleepCycleReport:
        """Execute a specific sleep phase"""
        cycle_id = f"{phase.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = SleepCycleReport(
            cycle_id=cycle_id,
            phase=phase,
            start_time=datetime.now(),
            end_time=datetime.now(),  # Will be updated
            initial_facts=len(self.persistent_kb.kb1.facts),
            initial_rules=len(self.persistent_kb.kb1.rules)
        )
        
        try:
            logger.info(f"Starting {phase.value} cycle: {cycle_id}")
            
            # Backup if configured
            if self.config.backup_before_sleep:
                backup_version = self.persistent_kb.create_version_snapshot({
                    "reason": f"pre_{phase.value}_backup"
                })
                logger.info(f"Created backup version: {backup_version.version_id}")
            
            # Get operations for this phase
            operations = self._get_operations_for_phase(phase)
            
            # Execute operations
            current_kb = self._copy_kb(self.persistent_kb.kb1)
            
            for operation in operations:
                try:
                    modified_kb, metrics = operation.execute(current_kb, self.config)
                    
                    # Validate changes if required
                    if self.config.require_validation:
                        self.validator.create_standard_tests(current_kb)
                        validation_report = self.validator.validate(modified_kb, current_kb)
                        
                        # Only apply changes if validation passes
                        if validation_report.critical_failures or len(validation_report.errors) > 0:
                            logger.warning(f"Validation failed for {operation.name}, skipping operation")
                            report.errors.append(f"Validation failed for {operation.name}")
                            continue
                        
                        report.validation_report = validation_report
                    
                    # Apply changes
                    current_kb = modified_kb
                    report.operations.append(operation.name)
                    
                    # Update metrics
                    if "facts_removed" in metrics:
                        report.facts_removed += metrics["facts_removed"]
                    if "rules_removed" in metrics:
                        report.rules_removed += metrics["rules_removed"]
                    if "rules_generated" in metrics:
                        report.rules_generated += metrics["rules_generated"]
                    
                    logger.info(f"Completed {operation.name}: {metrics}")
                    
                except Exception as e:
                    error_msg = f"Error in {operation.name}: {str(e)}"
                    logger.error(error_msg)
                    report.errors.append(error_msg)
            
            # Update KB if changes were made
            if report.operations:
                # Replace learned knowledge with optimized version
                self.persistent_kb.kb1 = current_kb
                
                # Update final metrics
                report.final_facts = len(current_kb.facts)
                report.final_rules = len(current_kb.rules)
                
                # Calculate compression ratio
                initial_size = report.initial_facts + report.initial_rules
                final_size = report.final_facts + report.final_rules
                if initial_size > 0:
                    report.compression_ratio = (initial_size - final_size) / initial_size
                
                # Save changes
                self.persistent_kb.save()
                
                # Log learning event
                self.persistent_kb.learning_events.append((
                    datetime.now(),
                    f"sleep_cycle_{phase.value}",
                    {
                        "cycle_id": cycle_id,
                        "operations": report.operations,
                        "compression_ratio": report.compression_ratio,
                        "knowledge_change": report.net_knowledge_change
                    }
                ))
                
                logger.info(f"Sleep cycle {cycle_id} completed: "
                           f"compression={report.compression_ratio:.2%}, "
                           f"operations={len(report.operations)}")
            else:
                logger.info(f"Sleep cycle {cycle_id} completed with no changes")
            
            # Update last sleep times
            current_time = datetime.now()
            if phase == SleepPhase.LIGHT_SLEEP:
                self.last_light_sleep = current_time
            elif phase == SleepPhase.DEEP_SLEEP:
                self.last_deep_sleep = current_time
            elif phase == SleepPhase.REM_SLEEP:
                self.last_rem_sleep = current_time
            
        except Exception as e:
            error_msg = f"Sleep cycle {cycle_id} failed: {str(e)}"
            logger.error(error_msg)
            report.errors.append(error_msg)
        
        finally:
            report.end_time = datetime.now()
            self.sleep_history.append(report)
            
            # Keep only recent history
            if len(self.sleep_history) > 100:
                self.sleep_history = self.sleep_history[-100:]
        
        return report
    
    def _get_operations_for_phase(self, phase: SleepPhase) -> List[SleepOperation]:
        """Get operations to execute for a sleep phase"""
        if phase == SleepPhase.LIGHT_SLEEP:
            return self.light_sleep_ops
        elif phase == SleepPhase.DEEP_SLEEP:
            return self.deep_sleep_ops
        elif phase == SleepPhase.REM_SLEEP:
            return self.rem_sleep_ops
        else:
            return []
    
    def _copy_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Create a deep copy of a knowledge base"""
        new_kb = KnowledgeBase()
        for fact in kb.facts:
            new_kb.add_fact(fact)
        for rule in kb.rules:
            new_kb.add_rule(rule)
        return new_kb
    
    def force_sleep_cycle(self, phase: SleepPhase) -> SleepCycleReport:
        """Force execution of a specific sleep phase immediately"""
        return self._execute_sleep_phase(phase)
    
    def get_sleep_metrics(self) -> Dict[str, Any]:
        """Get metrics about sleep cycle performance"""
        if not self.sleep_history:
            return {"error": "No sleep history available"}
        
        recent_cycles = [c for c in self.sleep_history 
                        if c.start_time > datetime.now() - timedelta(days=7)]
        
        total_compression = sum(c.compression_ratio for c in recent_cycles if c.compression_ratio > 0)
        avg_duration = sum((c.duration.total_seconds() for c in recent_cycles), 0) / len(recent_cycles) if recent_cycles else 0
        
        return {
            "total_cycles": len(self.sleep_history),
            "recent_cycles_7d": len(recent_cycles),
            "avg_compression_ratio": total_compression / len(recent_cycles) if recent_cycles else 0,
            "avg_cycle_duration_sec": avg_duration,
            "total_facts_removed": sum(c.facts_removed for c in self.sleep_history),
            "total_rules_generated": sum(c.rules_generated for c in self.sleep_history),
            "cycles_with_errors": sum(1 for c in self.sleep_history if c.errors),
            "last_light_sleep": self.last_light_sleep.isoformat(),
            "last_deep_sleep": self.last_deep_sleep.isoformat(),
            "last_rem_sleep": self.last_rem_sleep.isoformat()
        }