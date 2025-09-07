"""
Persistent Learning Architecture for DreamLog

This module implements a comprehensive persistent learning system that enables:
1. Dual knowledge base system (KB_1 learned + KB_2 user ground truth)
2. Persistent storage with versioning and conflict detection
3. Background learning service with sleep cycle integration
4. Knowledge validation and consistency checking
5. Safe user fact/rule injection with conflict resolution

Core Components:
- PersistentKnowledgeBase: Dual KB system with conflict detection
- KnowledgeValidator: Consistency checking and validation
- BackgroundLearner: Long-running learning service with IPC
- SleepCycleManager: Knowledge compression and generalization
- ConflictResolver: Strategies for handling KB inconsistencies
"""

import json
import logging
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Iterator, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .knowledge import KnowledgeBase, Fact, Rule
from .terms import Term, Variable, Atom, Compound
from .evaluator import PrologEvaluator, Solution
from .unification import unify, match
from .factories import term_from_prefix
from .llm_hook import LLMHook


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of knowledge base conflicts"""
    CONTRADICTION = "contradiction"  # KB_1 says p(X), KB_2 says ~p(X)
    INCONSISTENCY = "inconsistency"  # KB_1 ∪ KB_2 entails contradiction
    MISSING_SUPPORT = "missing_support"  # KB_2 fact has no support in KB_1
    OVERGENERALIZATION = "overgeneralization"  # KB_1 rule too broad
    REDUNDANCY = "redundancy"  # KB_1 and KB_2 contain equivalent knowledge


@dataclass
class ConflictReport:
    """Report of a knowledge base conflict"""
    conflict_type: ConflictType
    kb1_items: List[Union[Fact, Rule]]
    kb2_items: List[Union[Fact, Rule]]
    description: str
    severity: float  # 0.0 = minor, 1.0 = critical
    suggested_resolution: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeVersion:
    """Versioned snapshot of knowledge base state"""
    version_id: str
    timestamp: datetime
    kb1_hash: str
    kb2_hash: str
    facts_count: int
    rules_count: int
    conflicts: List[ConflictReport]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictResolutionStrategy(ABC):
    """Abstract base class for conflict resolution strategies"""
    
    @abstractmethod
    def resolve(self, conflict: ConflictReport, kb1: KnowledgeBase, kb2: KnowledgeBase) -> Tuple[KnowledgeBase, str]:
        """
        Resolve a conflict between knowledge bases
        
        Returns:
            Tuple of (resolved_kb1, resolution_description)
        """
        pass


class UserTrustStrategy(ConflictResolutionStrategy):
    """Always trust user knowledge (KB_2) over learned knowledge (KB_1)"""
    
    def resolve(self, conflict: ConflictReport, kb1: KnowledgeBase, kb2: KnowledgeBase) -> Tuple[KnowledgeBase, str]:
        new_kb1 = KnowledgeBase()
        
        # Copy all non-conflicting knowledge from KB_1
        conflicting_items = set(str(item) for item in conflict.kb1_items)
        
        for fact in kb1.facts:
            if str(fact) not in conflicting_items:
                new_kb1.add_fact(fact)
                
        for rule in kb1.rules:
            if str(rule) not in conflicting_items:
                new_kb1.add_rule(rule)
        
        # Add KB_2 items
        for item in conflict.kb2_items:
            if isinstance(item, Fact):
                new_kb1.add_fact(item)
            elif isinstance(item, Rule):
                new_kb1.add_rule(item)
        
        return new_kb1, f"Removed {len(conflict.kb1_items)} conflicting items, trusted user knowledge"


class ConservativeStrategy(ConflictResolutionStrategy):
    """Remove conflicting knowledge, keep only consensus"""
    
    def resolve(self, conflict: ConflictReport, kb1: KnowledgeBase, kb2: KnowledgeBase) -> Tuple[KnowledgeBase, str]:
        new_kb1 = KnowledgeBase()
        
        # Only keep non-conflicting knowledge
        conflicting_items = set(str(item) for item in conflict.kb1_items + conflict.kb2_items)
        
        for fact in kb1.facts:
            if str(fact) not in conflicting_items:
                new_kb1.add_fact(fact)
                
        for rule in kb1.rules:
            if str(rule) not in conflicting_items:
                new_kb1.add_rule(rule)
        
        return new_kb1, f"Removed {len(conflicting_items)} conflicting items to maintain consistency"


class PersistentKnowledgeBase:
    """
    Dual knowledge base system with persistent storage and conflict detection
    
    KB_1: Primary learned knowledge base (facts + rules from LLM inference)
    KB_2: User knowledge base (user-injected facts and rules, treated as ground truth)
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Dual knowledge bases
        self.kb1 = KnowledgeBase()  # Learned knowledge
        self.kb2 = KnowledgeBase()  # User ground truth
        
        # Conflict detection and resolution
        self.conflicts: List[ConflictReport] = []
        self.resolution_strategy: ConflictResolutionStrategy = UserTrustStrategy()
        
        # Versioning
        self.versions: List[KnowledgeVersion] = []
        self.current_version: Optional[KnowledgeVersion] = None
        
        # Performance tracking
        self.query_history: List[Tuple[datetime, List[Term], int]] = []  # (timestamp, query, solution_count)
        self.learning_events: List[Tuple[datetime, str, Dict[str, Any]]] = []  # (timestamp, event_type, metadata)
        
        # Load existing data
        self._load_from_storage()
    
    def add_learned_knowledge(self, facts: List[Fact], rules: List[Rule]) -> None:
        """Add knowledge learned from LLM to KB_1"""
        for fact in facts:
            self.kb1.add_fact(fact)
        for rule in rules:
            self.kb1.add_rule(rule)
        
        # Log learning event
        self.learning_events.append((
            datetime.now(),
            "learned_knowledge_added",
            {"facts_count": len(facts), "rules_count": len(rules)}
        ))
        
        # Check for new conflicts
        self._detect_conflicts()
    
    def add_user_knowledge(self, facts: List[Fact], rules: List[Rule]) -> List[ConflictReport]:
        """
        Add user knowledge to KB_2 and detect conflicts
        
        Returns:
            List of conflicts detected with existing KB_1
        """
        # Store KB_1 state before adding user knowledge
        original_kb1_facts = len(self.kb1.facts)
        original_kb1_rules = len(self.kb1.rules)
        
        for fact in facts:
            self.kb2.add_fact(fact)
        for rule in rules:
            self.kb2.add_rule(rule)
        
        # Detect conflicts
        new_conflicts = self._detect_conflicts()
        
        # Log user knowledge addition
        self.learning_events.append((
            datetime.now(),
            "user_knowledge_added",
            {
                "facts_count": len(facts),
                "rules_count": len(rules),
                "conflicts_detected": len(new_conflicts)
            }
        ))
        
        return new_conflicts
    
    def _detect_conflicts(self) -> List[ConflictReport]:
        """Detect conflicts between KB_1 and KB_2"""
        new_conflicts = []
        
        # Check for direct contradictions
        new_conflicts.extend(self._detect_contradictions())
        
        # Check for inconsistencies
        new_conflicts.extend(self._detect_inconsistencies())
        
        # Check for missing support
        new_conflicts.extend(self._detect_missing_support())
        
        # Add to conflicts list
        for conflict in new_conflicts:
            if conflict not in self.conflicts:
                self.conflicts.append(conflict)
        
        return new_conflicts
    
    def _detect_contradictions(self) -> List[ConflictReport]:
        """Detect direct contradictions between KB_1 and KB_2"""
        conflicts = []
        
        # Simple check: same fact in both KBs but negated
        # For now, we'll check for facts that are structurally similar but incompatible
        kb1_terms = set(str(fact.term) for fact in self.kb1.facts)
        kb2_terms = set(str(fact.term) for fact in self.kb2.facts)
        
        # Check for potential contradictions (this is simplified)
        for kb1_fact in self.kb1.facts:
            for kb2_fact in self.kb2.facts:
                if self._are_contradictory(kb1_fact.term, kb2_fact.term):
                    conflicts.append(ConflictReport(
                        conflict_type=ConflictType.CONTRADICTION,
                        kb1_items=[kb1_fact],
                        kb2_items=[kb2_fact],
                        description=f"Direct contradiction: KB_1 has '{kb1_fact.term}' but KB_2 has '{kb2_fact.term}'",
                        severity=0.9,
                        suggested_resolution="Trust user knowledge (KB_2) and remove conflicting learned knowledge"
                    ))
        
        return conflicts
    
    def _detect_inconsistencies(self) -> List[ConflictReport]:
        """Detect logical inconsistencies when KB_1 ∪ KB_2"""
        conflicts = []
        
        # Create combined knowledge base
        combined_kb = KnowledgeBase()
        for fact in self.kb1.facts + self.kb2.facts:
            combined_kb.add_fact(fact)
        for rule in self.kb1.rules + self.kb2.rules:
            combined_kb.add_rule(rule)
        
        # Use evaluator to detect inconsistencies
        evaluator = PrologEvaluator(combined_kb)
        
        # Check if we can derive contradictory facts
        # This is a simplified check - in practice, we'd need more sophisticated consistency checking
        fact_terms = [fact.term for fact in combined_kb.facts]
        for i, term1 in enumerate(fact_terms):
            for term2 in fact_terms[i+1:]:
                if self._are_contradictory(term1, term2):
                    # Found potential inconsistency
                    kb1_items = [f for f in self.kb1.facts if f.term == term1 or f.term == term2]
                    kb2_items = [f for f in self.kb2.facts if f.term == term1 or f.term == term2]
                    
                    if kb1_items and kb2_items:
                        conflicts.append(ConflictReport(
                            conflict_type=ConflictType.INCONSISTENCY,
                            kb1_items=kb1_items,
                            kb2_items=kb2_items,
                            description=f"Inconsistency detected: '{term1}' and '{term2}' are contradictory",
                            severity=0.8,
                            suggested_resolution="Resolve by trusting user knowledge and updating learned rules"
                        ))
        
        return conflicts
    
    def _detect_missing_support(self) -> List[ConflictReport]:
        """Detect KB_2 facts that have no support in KB_1"""
        conflicts = []
        
        for kb2_fact in self.kb2.facts:
            # Check if this fact can be derived from KB_1
            evaluator = PrologEvaluator(self.kb1)
            solutions = list(evaluator.query([kb2_fact.term]))
            
            if not solutions:
                conflicts.append(ConflictReport(
                    conflict_type=ConflictType.MISSING_SUPPORT,
                    kb1_items=[],
                    kb2_items=[kb2_fact],
                    description=f"User fact '{kb2_fact.term}' has no support in learned knowledge",
                    severity=0.3,
                    suggested_resolution="Generate supporting knowledge for user facts"
                ))
        
        return conflicts
    
    def _are_contradictory(self, term1: Term, term2: Term) -> bool:
        """Check if two terms are contradictory (simplified implementation)"""
        # This is a simplified check - real contradiction detection would be more complex
        if isinstance(term1, Compound) and isinstance(term2, Compound):
            # Check for explicit negation patterns
            if (term1.functor == "not" and len(term1.args) == 1 and 
                term1.args[0] == term2):
                return True
            if (term2.functor == "not" and len(term2.args) == 1 and 
                term2.args[0] == term1):
                return True
            
            # Check for same functor with contradictory arguments (domain-specific)
            if term1.functor == term2.functor and len(term1.args) == len(term2.args):
                # For predicates like alive/1, dead/1 etc.
                contradictory_pairs = [
                    ("alive", "dead"),
                    ("true", "false"),
                    ("present", "absent")
                ]
                for pos_pred, neg_pred in contradictory_pairs:
                    if ((term1.functor == pos_pred and term2.functor == neg_pred) or
                        (term1.functor == neg_pred and term2.functor == pos_pred)):
                        # Check if they refer to the same entity
                        if term1.args == term2.args:
                            return True
        
        return False
    
    def resolve_conflicts(self, strategy: Optional[ConflictResolutionStrategy] = None) -> Dict[str, Any]:
        """Resolve all detected conflicts using the specified strategy"""
        if strategy:
            self.resolution_strategy = strategy
        
        resolution_results = {
            "conflicts_resolved": 0,
            "items_modified": 0,
            "resolution_descriptions": []
        }
        
        for conflict in self.conflicts[:]:  # Create copy to modify during iteration
            try:
                resolved_kb1, description = self.resolution_strategy.resolve(conflict, self.kb1, self.kb2)
                
                # Update KB_1 with resolved version
                old_kb1_size = len(self.kb1.facts) + len(self.kb1.rules)
                self.kb1 = resolved_kb1
                new_kb1_size = len(self.kb1.facts) + len(self.kb1.rules)
                
                # Track resolution
                resolution_results["conflicts_resolved"] += 1
                resolution_results["items_modified"] += abs(new_kb1_size - old_kb1_size)
                resolution_results["resolution_descriptions"].append(description)
                
                # Remove resolved conflict
                self.conflicts.remove(conflict)
                
                # Log resolution
                self.learning_events.append((
                    datetime.now(),
                    "conflict_resolved",
                    {
                        "conflict_type": conflict.conflict_type.value,
                        "resolution_strategy": self.resolution_strategy.__class__.__name__,
                        "description": description
                    }
                ))
                
            except Exception as e:
                logger.error(f"Failed to resolve conflict: {e}")
                # Log failed resolution
                self.learning_events.append((
                    datetime.now(),
                    "conflict_resolution_failed",
                    {"error": str(e), "conflict_type": conflict.conflict_type.value}
                ))
        
        return resolution_results
    
    def create_version_snapshot(self, metadata: Optional[Dict[str, Any]] = None) -> KnowledgeVersion:
        """Create a versioned snapshot of current knowledge state"""
        version_id = hashlib.sha256(
            f"{datetime.now().isoformat()}{len(self.kb1.facts)}{len(self.kb2.facts)}".encode()
        ).hexdigest()[:16]
        
        kb1_hash = hashlib.sha256(self.kb1.to_prefix().encode()).hexdigest()
        kb2_hash = hashlib.sha256(self.kb2.to_prefix().encode()).hexdigest()
        
        version = KnowledgeVersion(
            version_id=version_id,
            timestamp=datetime.now(),
            kb1_hash=kb1_hash,
            kb2_hash=kb2_hash,
            facts_count=len(self.kb1.facts) + len(self.kb2.facts),
            rules_count=len(self.kb1.rules) + len(self.kb2.rules),
            conflicts=self.conflicts.copy(),
            metadata=metadata or {}
        )
        
        self.versions.append(version)
        self.current_version = version
        
        # Save snapshot to storage
        self._save_version(version)
        
        return version
    
    def query_with_tracking(self, goals: List[Term]) -> List[Solution]:
        """Query the combined knowledge base and track performance"""
        # Create combined KB for querying
        combined_kb = KnowledgeBase()
        
        # Add KB_1 first, then KB_2 (KB_2 takes precedence for conflicts)
        for fact in self.kb1.facts:
            combined_kb.add_fact(fact)
        for rule in self.kb1.rules:
            combined_kb.add_rule(rule)
        
        # Add KB_2 (may override conflicting knowledge)
        for fact in self.kb2.facts:
            combined_kb.add_fact(fact)  # KnowledgeBase handles duplicates
        for rule in self.kb2.rules:
            combined_kb.add_rule(rule)
        
        # Execute query
        evaluator = PrologEvaluator(combined_kb)
        solutions = list(evaluator.query(goals))
        
        # Track query
        self.query_history.append((datetime.now(), goals, len(solutions)))
        
        return solutions
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the knowledge base"""
        recent_queries = [q for q in self.query_history if q[0] > datetime.now() - timedelta(hours=24)]
        
        return {
            "kb1_size": {"facts": len(self.kb1.facts), "rules": len(self.kb1.rules)},
            "kb2_size": {"facts": len(self.kb2.facts), "rules": len(self.kb2.rules)},
            "conflicts": {
                "total": len(self.conflicts),
                "by_type": {ct.value: sum(1 for c in self.conflicts if c.conflict_type == ct) 
                          for ct in ConflictType}
            },
            "queries": {
                "total": len(self.query_history),
                "recent_24h": len(recent_queries),
                "avg_solutions": sum(q[2] for q in recent_queries) / len(recent_queries) if recent_queries else 0
            },
            "versions": len(self.versions),
            "learning_events": len(self.learning_events)
        }
    
    def _save_to_storage(self) -> None:
        """Save current state to persistent storage"""
        # Save KB_1
        kb1_path = self.storage_path / "kb1.json"
        with open(kb1_path, 'w') as f:
            f.write(self.kb1.to_prefix())
        
        # Save KB_2
        kb2_path = self.storage_path / "kb2.json"
        with open(kb2_path, 'w') as f:
            f.write(self.kb2.to_prefix())
        
        # Save conflicts
        conflicts_path = self.storage_path / "conflicts.json"
        conflicts_data = []
        for conflict in self.conflicts:
            conflicts_data.append({
                "conflict_type": conflict.conflict_type.value,
                "kb1_items": [str(item) for item in conflict.kb1_items],
                "kb2_items": [str(item) for item in conflict.kb2_items],
                "description": conflict.description,
                "severity": conflict.severity,
                "suggested_resolution": conflict.suggested_resolution,
                "timestamp": conflict.timestamp.isoformat()
            })
        
        with open(conflicts_path, 'w') as f:
            json.dump(conflicts_data, f, indent=2)
        
        # Save versions metadata
        versions_path = self.storage_path / "versions.json"
        versions_data = []
        for version in self.versions:
            versions_data.append({
                "version_id": version.version_id,
                "timestamp": version.timestamp.isoformat(),
                "kb1_hash": version.kb1_hash,
                "kb2_hash": version.kb2_hash,
                "facts_count": version.facts_count,
                "rules_count": version.rules_count,
                "conflicts_count": len(version.conflicts),
                "metadata": version.metadata
            })
        
        with open(versions_path, 'w') as f:
            json.dump(versions_data, f, indent=2)
        
        # Save learning events
        events_path = self.storage_path / "learning_events.json"
        events_data = []
        for timestamp, event_type, metadata in self.learning_events:
            events_data.append({
                "timestamp": timestamp.isoformat(),
                "event_type": event_type,
                "metadata": metadata
            })
        
        with open(events_path, 'w') as f:
            json.dump(events_data, f, indent=2)
    
    def _load_from_storage(self) -> None:
        """Load state from persistent storage"""
        try:
            # Load KB_1
            kb1_path = self.storage_path / "kb1.json"
            if kb1_path.exists():
                with open(kb1_path, 'r') as f:
                    self.kb1.from_prefix(f.read())
            
            # Load KB_2
            kb2_path = self.storage_path / "kb2.json"
            if kb2_path.exists():
                with open(kb2_path, 'r') as f:
                    self.kb2.from_prefix(f.read())
            
            # Load versions
            versions_path = self.storage_path / "versions.json"
            if versions_path.exists():
                with open(versions_path, 'r') as f:
                    versions_data = json.load(f)
                    for version_data in versions_data:
                        version = KnowledgeVersion(
                            version_id=version_data["version_id"],
                            timestamp=datetime.fromisoformat(version_data["timestamp"]),
                            kb1_hash=version_data["kb1_hash"],
                            kb2_hash=version_data["kb2_hash"],
                            facts_count=version_data["facts_count"],
                            rules_count=version_data["rules_count"],
                            conflicts=[],  # Conflicts are loaded separately
                            metadata=version_data.get("metadata", {})
                        )
                        self.versions.append(version)
                    
                    if self.versions:
                        self.current_version = self.versions[-1]
            
            # Load learning events
            events_path = self.storage_path / "learning_events.json"
            if events_path.exists():
                with open(events_path, 'r') as f:
                    events_data = json.load(f)
                    for event_data in events_data:
                        self.learning_events.append((
                            datetime.fromisoformat(event_data["timestamp"]),
                            event_data["event_type"],
                            event_data["metadata"]
                        ))
            
            logger.info(f"Loaded persistent knowledge base: {len(self.kb1.facts)} learned facts, "
                       f"{len(self.kb2.facts)} user facts, {len(self.versions)} versions")
                       
        except Exception as e:
            logger.error(f"Failed to load from storage: {e}")
    
    def _save_version(self, version: KnowledgeVersion) -> None:
        """Save a specific version snapshot"""
        version_dir = self.storage_path / "versions" / version.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save KB states
        kb1_path = version_dir / "kb1.json"
        with open(kb1_path, 'w') as f:
            f.write(self.kb1.to_prefix())
        
        kb2_path = version_dir / "kb2.json"
        with open(kb2_path, 'w') as f:
            f.write(self.kb2.to_prefix())
        
        # Save version metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "version_id": version.version_id,
                "timestamp": version.timestamp.isoformat(),
                "kb1_hash": version.kb1_hash,
                "kb2_hash": version.kb2_hash,
                "facts_count": version.facts_count,
                "rules_count": version.rules_count,
                "conflicts_count": len(version.conflicts),
                "metadata": version.metadata
            }, f, indent=2)
    
    def save(self) -> None:
        """Save current state to storage"""
        self._save_to_storage()