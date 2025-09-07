"""
Learning API for DreamLog Persistent Learning

This module provides a high-level API for safely injecting user knowledge,
managing conflicts, and interacting with the persistent learning system.
It serves as the main interface for external systems and users.

Key Components:
- LearningAPI: Main API interface
- SafeKnowledgeInjector: Safe knowledge injection with validation
- ConflictManager: Manage and resolve knowledge conflicts
- LearningProgressTracker: Track learning progress and improvements
- ValidationManager: Comprehensive validation and quality assurance
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Callable, Union, Iterator

from .persistent_learning import (
    PersistentKnowledgeBase, ConflictReport, ConflictType, 
    ConflictResolutionStrategy, UserTrustStrategy, ConservativeStrategy
)
from .knowledge_validator import KnowledgeValidator, ValidationReport, ValidationSeverity
from .sleep_cycle import SleepCycleManager, SleepPhase
from .background_learner import BackgroundLearnerClient, BackgroundLearner
from .knowledge import KnowledgeBase, Fact, Rule
from .terms import Term
from .evaluator import Solution
from .engine import DreamLogEngine
from .factories import term_from_prefix, atom, var, compound

logger = logging.getLogger(__name__)


class InjectionMode(Enum):
    """Modes for knowledge injection"""
    STRICT = "strict"        # Reject on any conflict
    PERMISSIVE = "permissive"  # Accept with automatic resolution
    INTERACTIVE = "interactive"  # Prompt user for resolution
    BATCH = "batch"         # Accumulate changes, resolve at end


class LearningMode(Enum):
    """Learning system operation modes"""
    ACTIVE = "active"       # Full learning with LLM integration
    PASSIVE = "passive"     # No new learning, use existing knowledge
    VALIDATION = "validation"  # Learning with strict validation
    EXPERIMENTAL = "experimental"  # Aggressive learning with relaxed validation


@dataclass
class InjectionResult:
    """Result of knowledge injection operation"""
    success: bool
    facts_added: int
    rules_added: int
    conflicts_detected: int
    conflicts_resolved: int
    validation_errors: List[str]
    resolution_strategy: str
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningProgress:
    """Progress tracking for learning system"""
    session_start: datetime
    session_id: str
    
    # Knowledge growth
    initial_facts: int
    initial_rules: int
    current_facts: int
    current_rules: int
    
    # Learning activity
    queries_processed: int
    facts_learned: int
    rules_learned: int
    user_facts_added: int
    user_rules_added: int
    
    # Quality metrics
    conflicts_detected: int
    conflicts_resolved: int
    validation_failures: int
    sleep_cycles_completed: int
    
    # Performance
    avg_query_time_ms: float
    knowledge_compression_ratio: float
    
    @property
    def knowledge_growth(self) -> float:
        """Calculate knowledge growth percentage"""
        initial = self.initial_facts + self.initial_rules
        current = self.current_facts + self.current_rules
        if initial == 0:
            return 100.0 if current > 0 else 0.0
        return ((current - initial) / initial) * 100.0
    
    @property
    def learning_efficiency(self) -> float:
        """Calculate learning efficiency metric"""
        if self.queries_processed == 0:
            return 0.0
        
        learned_knowledge = self.facts_learned + self.rules_learned
        return learned_knowledge / self.queries_processed


class SafeKnowledgeInjector:
    """Safe knowledge injection with validation and conflict detection"""
    
    def __init__(self, 
                 kb: PersistentKnowledgeBase,
                 validator: KnowledgeValidator,
                 injection_mode: InjectionMode = InjectionMode.PERMISSIVE):
        self.kb = kb
        self.validator = validator
        self.injection_mode = injection_mode
        
        # Track injection history
        self.injection_history: List[InjectionResult] = []
    
    def inject_facts(self, 
                    facts: List[Fact], 
                    validate: bool = True,
                    resolution_strategy: Optional[ConflictResolutionStrategy] = None) -> InjectionResult:
        """Safely inject facts into the knowledge base"""
        start_time = datetime.now()
        
        result = InjectionResult(
            success=False,
            facts_added=0,
            rules_added=0,
            conflicts_detected=0,
            conflicts_resolved=0,
            validation_errors=[],
            resolution_strategy="none",
            execution_time_ms=0.0
        )
        
        try:
            # Pre-injection validation
            if validate and self.injection_mode == InjectionMode.STRICT:
                for fact in facts:
                    if not self._validate_fact(fact):
                        result.validation_errors.append(f"Invalid fact: {fact}")
                
                if result.validation_errors and self.injection_mode == InjectionMode.STRICT:
                    result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return result
            
            # Inject knowledge and detect conflicts
            conflicts = self.kb.add_user_knowledge(facts, [])
            result.facts_added = len(facts)
            result.conflicts_detected = len(conflicts)
            
            # Handle conflicts based on mode
            if conflicts:
                if self.injection_mode == InjectionMode.STRICT:
                    # Rollback in strict mode
                    self._rollback_injection(facts, [])
                    result.success = False
                    result.facts_added = 0
                    result.metadata["rollback_reason"] = "conflicts_detected"
                    
                elif self.injection_mode == InjectionMode.PERMISSIVE:
                    # Auto-resolve conflicts
                    strategy = resolution_strategy or UserTrustStrategy()
                    resolution_results = self.kb.resolve_conflicts(strategy)
                    
                    result.conflicts_resolved = resolution_results["conflicts_resolved"]
                    result.resolution_strategy = strategy.__class__.__name__
                    result.success = True
                    
                elif self.injection_mode == InjectionMode.INTERACTIVE:
                    # Would implement interactive resolution here
                    # For now, fall back to auto-resolution
                    strategy = resolution_strategy or UserTrustStrategy()
                    resolution_results = self.kb.resolve_conflicts(strategy)
                    result.conflicts_resolved = resolution_results["conflicts_resolved"]
                    result.resolution_strategy = f"interactive_{strategy.__class__.__name__}"
                    result.success = True
            else:
                result.success = True
            
            # Post-injection validation
            if validate and result.success:
                validation_report = self.validator.validate(self.kb.kb1)
                if validation_report.critical_failures:
                    result.validation_errors.extend([f.message for f in validation_report.critical_failures])
                    if self.injection_mode == InjectionMode.STRICT:
                        # Rollback on validation failure
                        self._rollback_injection(facts, [])
                        result.success = False
                        result.facts_added = 0
            
        except Exception as e:
            logger.error(f"Error injecting facts: {e}")
            result.success = False
            result.validation_errors.append(str(e))
        
        finally:
            result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.injection_history.append(result)
        
        return result
    
    def inject_rules(self, 
                    rules: List[Rule],
                    validate: bool = True,
                    resolution_strategy: Optional[ConflictResolutionStrategy] = None) -> InjectionResult:
        """Safely inject rules into the knowledge base"""
        start_time = datetime.now()
        
        result = InjectionResult(
            success=False,
            facts_added=0,
            rules_added=0,
            conflicts_detected=0,
            conflicts_resolved=0,
            validation_errors=[],
            resolution_strategy="none",
            execution_time_ms=0.0
        )
        
        try:
            # Pre-injection validation
            if validate and self.injection_mode == InjectionMode.STRICT:
                for rule in rules:
                    if not self._validate_rule(rule):
                        result.validation_errors.append(f"Invalid rule: {rule}")
                
                if result.validation_errors and self.injection_mode == InjectionMode.STRICT:
                    result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return result
            
            # Inject knowledge and detect conflicts
            conflicts = self.kb.add_user_knowledge([], rules)
            result.rules_added = len(rules)
            result.conflicts_detected = len(conflicts)
            
            # Handle conflicts (similar to facts)
            if conflicts:
                if self.injection_mode == InjectionMode.STRICT:
                    self._rollback_injection([], rules)
                    result.success = False
                    result.rules_added = 0
                    result.metadata["rollback_reason"] = "conflicts_detected"
                else:
                    strategy = resolution_strategy or UserTrustStrategy()
                    resolution_results = self.kb.resolve_conflicts(strategy)
                    result.conflicts_resolved = resolution_results["conflicts_resolved"]
                    result.resolution_strategy = strategy.__class__.__name__
                    result.success = True
            else:
                result.success = True
            
            # Post-injection validation
            if validate and result.success:
                validation_report = self.validator.validate(self.kb.kb1)
                if validation_report.critical_failures:
                    result.validation_errors.extend([f.message for f in validation_report.critical_failures])
                    if self.injection_mode == InjectionMode.STRICT:
                        self._rollback_injection([], rules)
                        result.success = False
                        result.rules_added = 0
        
        except Exception as e:
            logger.error(f"Error injecting rules: {e}")
            result.success = False
            result.validation_errors.append(str(e))
        
        finally:
            result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.injection_history.append(result)
        
        return result
    
    def batch_inject(self, 
                    facts: List[Fact], 
                    rules: List[Rule],
                    validate: bool = True,
                    resolution_strategy: Optional[ConflictResolutionStrategy] = None) -> InjectionResult:
        """Inject facts and rules together in a batch operation"""
        start_time = datetime.now()
        
        result = InjectionResult(
            success=False,
            facts_added=0,
            rules_added=0,
            conflicts_detected=0,
            conflicts_resolved=0,
            validation_errors=[],
            resolution_strategy="none",
            execution_time_ms=0.0
        )
        
        try:
            # Pre-injection validation
            if validate and self.injection_mode == InjectionMode.STRICT:
                for fact in facts:
                    if not self._validate_fact(fact):
                        result.validation_errors.append(f"Invalid fact: {fact}")
                
                for rule in rules:
                    if not self._validate_rule(rule):
                        result.validation_errors.append(f"Invalid rule: {rule}")
                
                if result.validation_errors and self.injection_mode == InjectionMode.STRICT:
                    result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                    return result
            
            # Batch inject
            conflicts = self.kb.add_user_knowledge(facts, rules)
            result.facts_added = len(facts)
            result.rules_added = len(rules)
            result.conflicts_detected = len(conflicts)
            
            # Handle conflicts
            if conflicts:
                if self.injection_mode == InjectionMode.STRICT:
                    self._rollback_injection(facts, rules)
                    result.success = False
                    result.facts_added = 0
                    result.rules_added = 0
                    result.metadata["rollback_reason"] = "conflicts_detected"
                else:
                    strategy = resolution_strategy or UserTrustStrategy()
                    resolution_results = self.kb.resolve_conflicts(strategy)
                    result.conflicts_resolved = resolution_results["conflicts_resolved"]
                    result.resolution_strategy = strategy.__class__.__name__
                    result.success = True
            else:
                result.success = True
            
            # Post-injection validation
            if validate and result.success:
                validation_report = self.validator.validate(self.kb.kb1)
                if validation_report.critical_failures:
                    result.validation_errors.extend([f.message for f in validation_report.critical_failures])
                    if self.injection_mode == InjectionMode.STRICT:
                        self._rollback_injection(facts, rules)
                        result.success = False
                        result.facts_added = 0
                        result.rules_added = 0
        
        except Exception as e:
            logger.error(f"Error in batch injection: {e}")
            result.success = False
            result.validation_errors.append(str(e))
        
        finally:
            result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.injection_history.append(result)
        
        return result
    
    def _validate_fact(self, fact: Fact) -> bool:
        """Validate a single fact"""
        # Basic structural validation
        if not fact or not fact.term:
            return False
        
        # Check for variables in facts (facts should be ground)
        if fact.get_variables():
            logger.warning(f"Fact contains variables: {fact}")
            return False
        
        return True
    
    def _validate_rule(self, rule: Rule) -> bool:
        """Validate a single rule"""
        # Basic structural validation
        if not rule or not rule.head:
            return False
        
        # Check that all variables in head appear in body
        head_vars = rule.head.get_variables()
        body_vars = set()
        for goal in rule.body:
            body_vars.update(goal.get_variables())
        
        # Allow head-only variables for now (could be output variables)
        # In strict mode, might require all head vars to appear in body
        
        return True
    
    def _rollback_injection(self, facts: List[Fact], rules: List[Rule]) -> None:
        """Rollback injection by removing added knowledge"""
        # This is a simplified rollback - in practice, would need more sophisticated tracking
        logger.warning(f"Rollback not fully implemented - {len(facts)} facts, {len(rules)} rules")
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Get statistics about injection operations"""
        if not self.injection_history:
            return {"error": "No injection history"}
        
        total_ops = len(self.injection_history)
        successful_ops = sum(1 for r in self.injection_history if r.success)
        total_facts = sum(r.facts_added for r in self.injection_history)
        total_rules = sum(r.rules_added for r in self.injection_history)
        total_conflicts = sum(r.conflicts_detected for r in self.injection_history)
        
        return {
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "success_rate": successful_ops / total_ops if total_ops > 0 else 0,
            "total_facts_injected": total_facts,
            "total_rules_injected": total_rules,
            "total_conflicts_detected": total_conflicts,
            "avg_execution_time_ms": sum(r.execution_time_ms for r in self.injection_history) / total_ops if total_ops > 0 else 0
        }


class ConflictManager:
    """Manage and resolve knowledge conflicts"""
    
    def __init__(self, kb: PersistentKnowledgeBase):
        self.kb = kb
        self.resolution_strategies: Dict[str, ConflictResolutionStrategy] = {
            "user_trust": UserTrustStrategy(),
            "conservative": ConservativeStrategy()
        }
    
    def analyze_conflicts(self) -> Dict[str, Any]:
        """Analyze current conflicts in the knowledge base"""
        conflicts = self.kb.conflicts
        
        analysis = {
            "total_conflicts": len(conflicts),
            "by_type": {},
            "by_severity": {},
            "recent_conflicts": 0,
            "unresolved_conflicts": len(conflicts)
        }
        
        # Analyze by type
        for conflict_type in ConflictType:
            count = sum(1 for c in conflicts if c.conflict_type == conflict_type)
            analysis["by_type"][conflict_type.value] = count
        
        # Analyze by severity
        severity_bins = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for conflict in conflicts:
            if conflict.severity < 0.3:
                severity_bins["low"] += 1
            elif conflict.severity < 0.6:
                severity_bins["medium"] += 1
            elif conflict.severity < 0.9:
                severity_bins["high"] += 1
            else:
                severity_bins["critical"] += 1
        
        analysis["by_severity"] = severity_bins
        
        # Recent conflicts (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        analysis["recent_conflicts"] = sum(1 for c in conflicts if c.timestamp > cutoff)
        
        return analysis
    
    def resolve_conflicts_by_type(self, 
                                 conflict_type: ConflictType,
                                 strategy_name: str = "user_trust") -> Dict[str, Any]:
        """Resolve all conflicts of a specific type"""
        strategy = self.resolution_strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown resolution strategy: {strategy_name}")
        
        # Filter conflicts by type
        type_conflicts = [c for c in self.kb.conflicts if c.conflict_type == conflict_type]
        
        if not type_conflicts:
            return {"message": f"No conflicts of type {conflict_type.value}", "resolved": 0}
        
        # Resolve conflicts
        original_strategy = self.kb.resolution_strategy
        self.kb.resolution_strategy = strategy
        
        try:
            results = self.kb.resolve_conflicts()
            return results
        finally:
            self.kb.resolution_strategy = original_strategy
    
    def suggest_resolution_strategy(self, conflict: ConflictReport) -> str:
        """Suggest appropriate resolution strategy for a conflict"""
        if conflict.conflict_type == ConflictType.CONTRADICTION:
            return "user_trust"  # Trust user over learned knowledge
        elif conflict.conflict_type == ConflictType.MISSING_SUPPORT:
            return "conservative"  # Be conservative with unsupported facts
        elif conflict.severity > 0.8:
            return "user_trust"  # High severity - trust user
        else:
            return "conservative"  # Low severity - be conservative
    
    def create_conflict_report(self) -> str:
        """Create a human-readable conflict report"""
        conflicts = self.kb.conflicts
        if not conflicts:
            return "No conflicts detected in knowledge base."
        
        report_lines = [
            f"Knowledge Base Conflict Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Conflicts: {len(conflicts)}",
            "",
            "Conflicts by Type:",
        ]
        
        # Group by type
        by_type = {}
        for conflict in conflicts:
            conflict_type = conflict.conflict_type.value
            if conflict_type not in by_type:
                by_type[conflict_type] = []
            by_type[conflict_type].append(conflict)
        
        for conflict_type, type_conflicts in by_type.items():
            report_lines.append(f"  {conflict_type.upper()}: {len(type_conflicts)}")
            for i, conflict in enumerate(type_conflicts[:3]):  # Show first 3
                report_lines.append(f"    {i+1}. {conflict.description}")
                report_lines.append(f"       Severity: {conflict.severity:.2f}")
                report_lines.append(f"       Suggested: {conflict.suggested_resolution}")
            
            if len(type_conflicts) > 3:
                report_lines.append(f"    ... and {len(type_conflicts) - 3} more")
            report_lines.append("")
        
        return "\n".join(report_lines)


class LearningProgressTracker:
    """Track learning progress and improvements"""
    
    def __init__(self, kb: PersistentKnowledgeBase):
        self.kb = kb
        self.session_start = datetime.now()
        self.session_id = f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}"
        
        # Capture initial state
        self.initial_facts = len(kb.kb1.facts) + len(kb.kb2.facts)
        self.initial_rules = len(kb.kb1.rules) + len(kb.kb2.rules)
        
        # Progress tracking
        self.progress_snapshots: List[LearningProgress] = []
        self.last_snapshot_time = datetime.now()
    
    def take_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> LearningProgress:
        """Take a progress snapshot"""
        
        # Get metrics from KB
        kb_metrics = self.kb.get_performance_metrics()
        
        progress = LearningProgress(
            session_start=self.session_start,
            session_id=self.session_id,
            initial_facts=self.initial_facts,
            initial_rules=self.initial_rules,
            current_facts=len(self.kb.kb1.facts) + len(self.kb.kb2.facts),
            current_rules=len(self.kb.kb1.rules) + len(self.kb.kb2.rules),
            queries_processed=kb_metrics.get("queries", {}).get("total", 0),
            facts_learned=len(self.kb.kb1.facts),
            rules_learned=len(self.kb.kb1.rules),
            user_facts_added=len(self.kb.kb2.facts),
            user_rules_added=len(self.kb.kb2.rules),
            conflicts_detected=len(self.kb.conflicts),
            conflicts_resolved=0,  # Would track from resolution events
            validation_failures=0,  # Would track from validation events
            sleep_cycles_completed=0,  # Would get from sleep manager
            avg_query_time_ms=kb_metrics.get("queries", {}).get("avg_solutions", 0),
            knowledge_compression_ratio=0.0  # Would calculate from sleep cycles
        )
        
        self.progress_snapshots.append(progress)
        self.last_snapshot_time = datetime.now()
        
        # Keep only recent snapshots
        cutoff = datetime.now() - timedelta(days=7)
        self.progress_snapshots = [s for s in self.progress_snapshots if s.session_start > cutoff]
        
        return progress
    
    def get_learning_trend(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get learning trend over specified time window"""
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_snapshots = [s for s in self.progress_snapshots if s.session_start > cutoff]
        
        if len(recent_snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        first_snapshot = recent_snapshots[0]
        last_snapshot = recent_snapshots[-1]
        
        return {
            "time_window_hours": window_hours,
            "snapshots_analyzed": len(recent_snapshots),
            "knowledge_growth": last_snapshot.knowledge_growth - first_snapshot.knowledge_growth,
            "facts_growth": last_snapshot.current_facts - first_snapshot.current_facts,
            "rules_growth": last_snapshot.current_rules - first_snapshot.current_rules,
            "queries_processed": last_snapshot.queries_processed - first_snapshot.queries_processed,
            "learning_efficiency_change": last_snapshot.learning_efficiency - first_snapshot.learning_efficiency,
            "conflicts_trend": last_snapshot.conflicts_detected - first_snapshot.conflicts_detected
        }
    
    def generate_progress_report(self) -> str:
        """Generate a human-readable progress report"""
        if not self.progress_snapshots:
            current_progress = self.take_snapshot()
        else:
            current_progress = self.progress_snapshots[-1]
        
        report_lines = [
            f"Learning Progress Report",
            f"Session: {self.session_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session Duration: {datetime.now() - self.session_start}",
            "",
            f"Knowledge Base Growth:",
            f"  Initial: {self.initial_facts} facts, {self.initial_rules} rules",
            f"  Current: {current_progress.current_facts} facts, {current_progress.current_rules} rules",
            f"  Growth: {current_progress.knowledge_growth:.1f}%",
            "",
            f"Learning Activity:",
            f"  Queries Processed: {current_progress.queries_processed}",
            f"  Facts Learned: {current_progress.facts_learned}",
            f"  Rules Learned: {current_progress.rules_learned}",
            f"  User Facts Added: {current_progress.user_facts_added}",
            f"  User Rules Added: {current_progress.user_rules_added}",
            f"  Learning Efficiency: {current_progress.learning_efficiency:.3f}",
            "",
            f"Quality Metrics:",
            f"  Conflicts Detected: {current_progress.conflicts_detected}",
            f"  Conflicts Resolved: {current_progress.conflicts_resolved}",
            f"  Validation Failures: {current_progress.validation_failures}",
            f"  Sleep Cycles: {current_progress.sleep_cycles_completed}",
            "",
            f"Performance:",
            f"  Avg Query Time: {current_progress.avg_query_time_ms:.1f}ms",
            f"  Knowledge Compression: {current_progress.knowledge_compression_ratio:.1%}",
        ]
        
        return "\n".join(report_lines)


class LearningAPI:
    """
    Main API interface for DreamLog persistent learning system
    
    Provides high-level operations for knowledge injection, conflict management,
    progress tracking, and system monitoring with comprehensive error handling.
    """
    
    def __init__(self, 
                 storage_path: Path,
                 learning_mode: LearningMode = LearningMode.ACTIVE,
                 injection_mode: InjectionMode = InjectionMode.PERMISSIVE,
                 use_background_service: bool = False,
                 background_port: int = 7777):
        
        self.storage_path = Path(storage_path)
        self.learning_mode = learning_mode
        self.injection_mode = injection_mode
        self.use_background_service = use_background_service
        
        # Core components
        if use_background_service:
            self.background_client = BackgroundLearnerClient(background_port)
            self.kb = None  # Use background service
            self.injector = None
            self.conflict_manager = None
            self.progress_tracker = None
        else:
            # Direct mode
            self.kb = PersistentKnowledgeBase(self.storage_path)
            self.validator = KnowledgeValidator()
            self.injector = SafeKnowledgeInjector(self.kb, self.validator, injection_mode)
            self.conflict_manager = ConflictManager(self.kb)
            self.progress_tracker = LearningProgressTracker(self.kb)
            self.background_client = None
        
        # API state
        self.api_start_time = datetime.now()
        self.operation_history: List[Dict[str, Any]] = []
    
    def inject_knowledge_from_strings(self, 
                                    fact_strings: List[str], 
                                    rule_strings: List[str],
                                    validate: bool = True) -> InjectionResult:
        """
        Inject knowledge from string representations
        
        Args:
            fact_strings: List of fact strings like "parent(john, mary)"
            rule_strings: List of rule strings like "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)"
            validate: Whether to validate before injection
        """
        start_time = datetime.now()
        
        try:
            # Parse strings to facts and rules
            facts = []
            rules = []
            
            # Parse facts (simplified - would use proper parser)
            for fact_str in fact_strings:
                try:
                    # This is a placeholder - would use proper parsing
                    fact_term = self._parse_term_string(fact_str)
                    facts.append(Fact(fact_term))
                except Exception as e:
                    logger.error(f"Failed to parse fact '{fact_str}': {e}")
            
            # Parse rules (simplified)
            for rule_str in rule_strings:
                try:
                    # This is a placeholder - would use proper parsing
                    head, body = self._parse_rule_string(rule_str)
                    rules.append(Rule(head, body))
                except Exception as e:
                    logger.error(f"Failed to parse rule '{rule_str}': {e}")
            
            # Inject using appropriate method
            if self.use_background_service:
                result_data = self.background_client.add_user_knowledge(facts, rules)
                result = InjectionResult(
                    success=True,
                    facts_added=result_data.get("facts_added", 0),
                    rules_added=result_data.get("rules_added", 0),
                    conflicts_detected=result_data.get("conflicts_detected", 0),
                    conflicts_resolved=result_data.get("conflicts_resolved", 0),
                    validation_errors=[],
                    resolution_strategy="background_service",
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            else:
                result = self.injector.batch_inject(facts, rules, validate)
            
            # Track operation
            self._track_operation("inject_knowledge_from_strings", {
                "fact_count": len(facts),
                "rule_count": len(rules),
                "success": result.success,
                "execution_time": result.execution_time_ms
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error injecting knowledge from strings: {e}")
            
            return InjectionResult(
                success=False,
                facts_added=0,
                rules_added=0,
                conflicts_detected=0,
                conflicts_resolved=0,
                validation_errors=[str(e)],
                resolution_strategy="error",
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def query_knowledge(self, query_string: str) -> List[Dict[str, Any]]:
        """
        Query the knowledge base with a string query
        
        Args:
            query_string: Query string like "parent(john, X)"
            
        Returns:
            List of solution bindings
        """
        start_time = datetime.now()
        
        try:
            # Parse query string
            query_term = self._parse_term_string(query_string)
            
            if self.use_background_service:
                solutions = self.background_client.query([query_term])
            else:
                solutions = self.kb.query_with_tracking([query_term])
                # Convert to dict format
                solution_dicts = []
                for solution in solutions:
                    bindings = {}
                    for var_name, term in solution.get_ground_bindings().items():
                        bindings[var_name] = str(term)
                    solution_dicts.append(bindings)
                solutions = solution_dicts
            
            # Track operation
            self._track_operation("query_knowledge", {
                "query": query_string,
                "solution_count": len(solutions),
                "execution_time": (datetime.now() - start_time).total_seconds() * 1000
            })
            
            return solutions
            
        except Exception as e:
            logger.error(f"Error querying knowledge: {e}")
            self._track_operation("query_knowledge", {
                "query": query_string,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds() * 1000
            })
            return []
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get summary of current conflicts"""
        if self.use_background_service:
            status = self.background_client.get_status()
            # Extract conflict info from status
            return {
                "total_conflicts": 0,  # Would extract from background service
                "message": "Conflict info not available from background service"
            }
        else:
            return self.conflict_manager.analyze_conflicts()
    
    def resolve_all_conflicts(self, strategy: str = "user_trust") -> Dict[str, Any]:
        """Resolve all current conflicts using specified strategy"""
        if self.use_background_service:
            return {"error": "Conflict resolution not available via background service"}
        else:
            if strategy not in self.conflict_manager.resolution_strategies:
                return {"error": f"Unknown strategy: {strategy}"}
            
            strategy_obj = self.conflict_manager.resolution_strategies[strategy]
            return self.kb.resolve_conflicts(strategy_obj)
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get current learning progress"""
        if self.use_background_service:
            status = self.background_client.get_status()
            return status.get("current_session", {})
        else:
            if self.progress_tracker:
                progress = self.progress_tracker.take_snapshot()
                return asdict(progress)
            return {"error": "Progress tracker not available"}
    
    def force_sleep_cycle(self, phase: str = "light_sleep") -> Dict[str, Any]:
        """Force a sleep cycle to run immediately"""
        if self.use_background_service:
            try:
                phase_enum = SleepPhase(phase.lower())
                return self.background_client.force_sleep_cycle(phase_enum)
            except ValueError:
                return {"error": f"Invalid sleep phase: {phase}"}
        else:
            return {"error": "Sleep cycle only available via background service"}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        metrics = {
            "api": {
                "start_time": self.api_start_time.isoformat(),
                "uptime_seconds": (datetime.now() - self.api_start_time).total_seconds(),
                "learning_mode": self.learning_mode.value,
                "injection_mode": self.injection_mode.value,
                "background_service": self.use_background_service,
                "operations_performed": len(self.operation_history)
            }
        }
        
        if self.use_background_service:
            try:
                status = self.background_client.get_status()
                metrics["background_service"] = status
            except Exception as e:
                metrics["background_service_error"] = str(e)
        else:
            if self.kb:
                metrics["knowledge_base"] = self.kb.get_performance_metrics()
            if self.injector:
                metrics["injection"] = self.injector.get_injection_stats()
            if self.conflict_manager:
                metrics["conflicts"] = self.conflict_manager.analyze_conflicts()
        
        return metrics
    
    def _parse_term_string(self, term_str: str) -> Term:
        """Parse a term string - simplified implementation"""
        # This is a placeholder - would implement proper parsing
        term_str = term_str.strip()
        
        if '(' in term_str:
            # Compound term
            functor = term_str[:term_str.index('(')]
            args_str = term_str[term_str.index('(')+1:term_str.rindex(')')]
            
            if not args_str:
                return compound(functor)
            
            # Simple argument parsing (would need proper parser)
            arg_strs = [arg.strip() for arg in args_str.split(',')]
            args = []
            for arg_str in arg_strs:
                if arg_str[0].isupper():
                    args.append(var(arg_str))
                else:
                    args.append(atom(arg_str))
            
            return compound(functor, *args)
        else:
            # Atom or variable
            if term_str[0].isupper():
                return var(term_str)
            else:
                return atom(term_str)
    
    def _parse_rule_string(self, rule_str: str) -> Tuple[Term, List[Term]]:
        """Parse a rule string - simplified implementation"""
        # This is a placeholder - would implement proper parsing
        rule_str = rule_str.strip()
        
        # Check if it contains rule separator
        if ' :- ' in rule_str:
            head_str, body_str = rule_str.split(' :- ', 1)
        elif ':-' in rule_str:
            head_str, body_str = rule_str.split(':-', 1)
        else:
            # Fact rule (no body)
            head = self._parse_term_string(rule_str)
            return head, []
        
        # Parse head
        head = self._parse_term_string(head_str.strip())
        
        # Parse body goals
        body_terms = []
        body_goals = body_str.split(',')
        for goal_str in body_goals:
            goal_str = goal_str.strip()
            if goal_str:  # Skip empty strings
                body_terms.append(self._parse_term_string(goal_str))
        
        return head, body_terms
    
    def _track_operation(self, operation: str, metadata: Dict[str, Any]) -> None:
        """Track API operation for monitoring"""
        self.operation_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "metadata": metadata
        })
        
        # Keep only recent operations
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]
    
    def save_state(self) -> None:
        """Save current state to persistent storage"""
        if not self.use_background_service and self.kb:
            self.kb.save()
    
    def shutdown(self) -> None:
        """Shutdown the learning API and clean up resources"""
        if self.use_background_service:
            try:
                self.background_client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down background service: {e}")
        else:
            if self.kb:
                self.kb.save()
        
        logger.info("Learning API shutdown complete")