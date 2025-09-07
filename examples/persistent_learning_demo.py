#!/usr/bin/env python3
"""
DreamLog Persistent Learning System Demo

This demo showcases the complete persistent learning architecture including:
1. Dual knowledge base system with conflict detection
2. Safe knowledge injection and validation
3. Sleep cycle operations for knowledge optimization
4. Background learning service with IPC communication
5. Progress tracking and performance monitoring

Run this demo to see the persistent learning system in action.
"""

import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Import the persistent learning system
from dreamlog.persistent_learning import PersistentKnowledgeBase, UserTrustStrategy
from dreamlog.knowledge_validator import KnowledgeValidator
from dreamlog.sleep_cycle import SleepCycleManager, SleepPhase, SleepCycleConfig
from dreamlog.learning_api import LearningAPI, InjectionMode, LearningMode
from dreamlog.background_learner import BackgroundLearner, BackgroundLearnerClient
from dreamlog.logging_config import setup_logging, get_logger

# Import core DreamLog components
from dreamlog.knowledge import Fact, Rule
from dreamlog.factories import atom, var, compound

# Setup logging
setup_logging(enable_file_logging=False, log_level="INFO")
logger = get_logger(__name__)


def demo_basic_persistent_learning():
    """Demo 1: Basic persistent learning operations"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Persistent Learning Operations")
    print("="*60)
    
    # Create temporary storage
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary storage: {temp_dir}")
    
    try:
        # Initialize persistent knowledge base
        print("\n1. Initializing persistent knowledge base...")
        kb = PersistentKnowledgeBase(temp_dir)
        
        # Add some learned knowledge (simulating LLM-generated knowledge)
        print("\n2. Adding learned knowledge to KB_1...")
        learned_facts = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("mary"), atom("alice"))),
            Fact(compound("parent", atom("bob"), atom("charlie"))),
            Fact(compound("likes", atom("john"), atom("pizza"))),
            Fact(compound("likes", atom("mary"), atom("books")))
        ]
        
        learned_rules = [
            Rule(
                compound("grandparent", var("X"), var("Z")),
                [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
            ),
            Rule(
                compound("ancestor", var("X"), var("Y")),
                [compound("parent", var("X"), var("Y"))]
            )
        ]
        
        kb.add_learned_knowledge(learned_facts, learned_rules)
        print(f"   Added {len(learned_facts)} facts and {len(learned_rules)} rules to learned KB")
        
        # Test querying
        print("\n3. Testing queries...")
        query_goals = [compound("grandparent", var("X"), var("Y"))]
        solutions = kb.query_with_tracking(query_goals)
        print(f"   Query 'grandparent(X,Y)' found {len(solutions)} solutions:")
        for i, solution in enumerate(solutions, 1):
            bindings = solution.get_ground_bindings()
            print(f"     {i}. {dict(bindings)}")
        
        # Add user knowledge that might conflict
        print("\n4. Adding user knowledge to KB_2...")
        user_facts = [
            Fact(compound("parent", atom("john"), atom("tom"))),  # New fact
            Fact(compound("parent", atom("mary"), atom("alice"))),  # Duplicate (no conflict)
        ]
        
        user_rules = [
            Rule(
                compound("sibling", var("X"), var("Y")),
                [compound("parent", var("Z"), var("X")), compound("parent", var("Z"), var("Y"))]
            )
        ]
        
        conflicts = kb.add_user_knowledge(user_facts, user_rules)
        print(f"   Added {len(user_facts)} facts and {len(user_rules)} rules to user KB")
        print(f"   Detected {len(conflicts)} conflicts")
        
        # Show performance metrics
        print("\n5. Performance metrics:")
        metrics = kb.get_performance_metrics()
        print(f"   KB_1 size: {metrics['kb1_size']['facts']} facts, {metrics['kb1_size']['rules']} rules")
        print(f"   KB_2 size: {metrics['kb2_size']['facts']} facts, {metrics['kb2_size']['rules']} rules")
        print(f"   Total queries: {metrics['queries']['total']}")
        print(f"   Learning events: {metrics['learning_events']}")
        
        # Create version snapshot
        print("\n6. Creating version snapshot...")
        version = kb.create_version_snapshot({"demo": "basic_operations"})
        print(f"   Created version: {version.version_id}")
        print(f"   Version timestamp: {version.timestamp}")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary storage")


def demo_conflict_resolution():
    """Demo 2: Conflict detection and resolution"""
    print("\n" + "="*60)
    print("DEMO 2: Conflict Detection and Resolution")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary storage: {temp_dir}")
    
    try:
        kb = PersistentKnowledgeBase(temp_dir)
        
        # Add learned knowledge
        print("\n1. Adding learned knowledge...")
        learned_facts = [
            Fact(compound("alive", atom("john"))),
            Fact(compound("status", atom("mary"), atom("student"))),
            Fact(compound("location", atom("bob"), atom("home")))
        ]
        kb.add_learned_knowledge(learned_facts, [])
        print(f"   Added {len(learned_facts)} learned facts")
        
        # Add conflicting user knowledge
        print("\n2. Adding potentially conflicting user knowledge...")
        user_facts = [
            Fact(compound("dead", atom("john"))),  # Contradicts alive(john)
            Fact(compound("status", atom("mary"), atom("teacher"))),  # Contradicts student status
            Fact(compound("location", atom("bob"), atom("work")))  # Contradicts home location
        ]
        conflicts = kb.add_user_knowledge(user_facts, [])
        print(f"   Added {len(user_facts)} user facts")
        print(f"   Detected {len(conflicts)} potential conflicts")
        
        # Analyze conflicts
        print("\n3. Analyzing conflicts...")
        for i, conflict in enumerate(conflicts, 1):
            print(f"   Conflict {i}: {conflict.conflict_type.value}")
            print(f"     Description: {conflict.description}")
            print(f"     Severity: {conflict.severity:.2f}")
            print(f"     Suggested resolution: {conflict.suggested_resolution}")
        
        # Resolve conflicts using user trust strategy
        print("\n4. Resolving conflicts with UserTrustStrategy...")
        resolution_results = kb.resolve_conflicts(UserTrustStrategy())
        print(f"   Conflicts resolved: {resolution_results['conflicts_resolved']}")
        print(f"   Items modified: {resolution_results['items_modified']}")
        
        for desc in resolution_results['resolution_descriptions']:
            print(f"     - {desc}")
        
        # Show final state
        print("\n5. Final knowledge base state:")
        final_metrics = kb.get_performance_metrics()
        if 'kb1_size' in final_metrics:
            print(f"   KB_1: {final_metrics['kb1_size']['facts']} facts, {final_metrics['kb1_size']['rules']} rules")
            print(f"   KB_2: {final_metrics['kb2_size']['facts']} facts, {final_metrics['kb2_size']['rules']} rules")
            print(f"   Remaining conflicts: {final_metrics['conflicts']['total']}")
        else:
            print(f"   KB_1: {len(kb.kb1.facts)} facts, {len(kb.kb1.rules)} rules")
            print(f"   KB_2: {len(kb.kb2.facts)} facts, {len(kb.kb2.rules)} rules")
            print(f"   Remaining conflicts: {len(kb.conflicts)}")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary storage")


def demo_sleep_cycles():
    """Demo 3: Sleep cycle operations"""
    print("\n" + "="*60)
    print("DEMO 3: Sleep Cycle Operations")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary storage: {temp_dir}")
    
    try:
        kb = PersistentKnowledgeBase(temp_dir)
        
        # Add knowledge with redundancy and duplicates
        print("\n1. Adding knowledge with redundancy...")
        facts_with_duplicates = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("john"), atom("mary"))),  # Duplicate
            Fact(compound("parent", atom("mary"), atom("alice"))),
            Fact(compound("parent", atom("mary"), atom("alice"))),  # Duplicate
            Fact(compound("likes", atom("john"), atom("pizza"))),
            Fact(compound("likes", atom("mary"), atom("books"))),
            Fact(compound("parent", atom("bob"), atom("tom"))),
            Fact(compound("parent", atom("alice"), atom("sam"))),
        ]
        
        rules = [
            Rule(compound("grandparent", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]),
            Rule(compound("ancestor", var("X"), var("Y")),
                 [compound("parent", var("X"), var("Y"))]),
        ]
        
        kb.add_learned_knowledge(facts_with_duplicates, rules)
        print(f"   Added {len(facts_with_duplicates)} facts (with duplicates) and {len(rules)} rules")
        
        initial_facts = len(kb.kb1.facts)
        initial_rules = len(kb.kb1.rules)
        print(f"   Initial KB_1 size: {initial_facts} facts, {initial_rules} rules")
        
        # Setup sleep cycle manager
        print("\n2. Setting up sleep cycle manager...")
        config = SleepCycleConfig(
            require_validation=False,  # Skip validation for demo speed
            backup_before_sleep=True
        )
        manager = SleepCycleManager(kb, config)
        
        # Execute light sleep cycle (cleanup and optimization)
        print("\n3. Executing light sleep cycle...")
        light_report = manager.force_sleep_cycle(SleepPhase.LIGHT_SLEEP)
        print(f"   Cycle ID: {light_report.cycle_id}")
        print(f"   Operations performed: {light_report.operations}")
        print(f"   Facts removed: {light_report.facts_removed}")
        print(f"   Rules removed: {light_report.rules_removed}")
        print(f"   Compression ratio: {light_report.compression_ratio:.2%}")
        print(f"   Duration: {light_report.duration.total_seconds():.2f} seconds")
        
        # Execute deep sleep cycle (rule generalization)
        print("\n4. Executing deep sleep cycle...")
        deep_report = manager.force_sleep_cycle(SleepPhase.DEEP_SLEEP)
        print(f"   Cycle ID: {deep_report.cycle_id}")
        print(f"   Operations performed: {deep_report.operations}")
        print(f"   Rules generated: {deep_report.rules_generated}")
        print(f"   Knowledge change: {deep_report.net_knowledge_change}")
        
        # Show final state
        final_facts = len(kb.kb1.facts)
        final_rules = len(kb.kb1.rules)
        print(f"\n5. Final KB_1 size: {final_facts} facts, {final_rules} rules")
        print(f"   Net fact change: {final_facts - initial_facts}")
        print(f"   Net rule change: {final_rules - initial_rules}")
        
        # Show sleep metrics
        print("\n6. Sleep cycle metrics:")
        sleep_metrics = manager.get_sleep_metrics()
        print(f"   Total cycles: {sleep_metrics['total_cycles']}")
        print(f"   Total facts removed: {sleep_metrics['total_facts_removed']}")
        print(f"   Total rules generated: {sleep_metrics['total_rules_generated']}")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary storage")


def demo_learning_api():
    """Demo 4: High-level Learning API"""
    print("\n" + "="*60)
    print("DEMO 4: High-level Learning API")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary storage: {temp_dir}")
    
    try:
        # Initialize API
        print("\n1. Initializing Learning API...")
        api = LearningAPI(
            temp_dir,
            learning_mode=LearningMode.ACTIVE,
            injection_mode=InjectionMode.PERMISSIVE,
            use_background_service=False
        )
        print("   API initialized in direct mode")
        
        # Inject knowledge using strings
        print("\n2. Injecting knowledge from strings...")
        fact_strings = [
            "parent(john, mary)",
            "parent(mary, alice)",
            "parent(bob, tom)",
            "likes(john, pizza)",
            "likes(mary, books)"
        ]
        
        rule_strings = [
            "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)",
            "sibling(X,Y) :- parent(Z,X), parent(Z,Y)"
        ]
        
        injection_result = api.inject_knowledge_from_strings(fact_strings, rule_strings)
        print(f"   Injection success: {injection_result.success}")
        print(f"   Facts added: {injection_result.facts_added}")
        print(f"   Rules added: {injection_result.rules_added}")
        print(f"   Conflicts detected: {injection_result.conflicts_detected}")
        print(f"   Execution time: {injection_result.execution_time_ms:.1f}ms")
        
        # Query knowledge
        print("\n3. Querying knowledge...")
        queries = [
            "parent(john, X)",
            "grandparent(X, alice)",
            "sibling(X, Y)"
        ]
        
        for query in queries:
            solutions = api.query_knowledge(query)
            print(f"   Query '{query}' -> {len(solutions)} solutions")
            for i, solution in enumerate(solutions[:3], 1):  # Show first 3 solutions
                print(f"     {i}. {solution}")
        
        # Check learning progress
        print("\n4. Learning progress:")
        progress = api.get_learning_progress()
        if isinstance(progress, dict) and "error" not in progress:
            print(f"   Knowledge growth: {progress.get('knowledge_growth', 0):.1f}%")
            print(f"   Current facts: {progress.get('current_facts', 0)}")
            print(f"   Current rules: {progress.get('current_rules', 0)}")
            print(f"   Learning efficiency: {progress.get('learning_efficiency', 0):.3f}")
        
        # Get system metrics
        print("\n5. System metrics:")
        metrics = api.get_system_metrics()
        print(f"   API uptime: {metrics['api']['uptime_seconds']:.1f} seconds")
        print(f"   Operations performed: {metrics['api']['operations_performed']}")
        print(f"   Learning mode: {metrics['api']['learning_mode']}")
        print(f"   Injection mode: {metrics['api']['injection_mode']}")
        
        # Show knowledge base metrics
        if 'knowledge_base' in metrics:
            kb_metrics = metrics['knowledge_base']
            print(f"   KB total queries: {kb_metrics['queries']['total']}")
            print(f"   KB learning events: {kb_metrics['learning_events']}")
        
    finally:
        api.shutdown()
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary storage")


def demo_validation_system():
    """Demo 5: Knowledge validation system"""
    print("\n" + "="*60)
    print("DEMO 5: Knowledge Validation System")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary storage: {temp_dir}")
    
    try:
        kb = PersistentKnowledgeBase(temp_dir)
        
        # Add initial knowledge
        print("\n1. Adding initial knowledge...")
        facts = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("mary"), atom("alice"))),
            Fact(compound("parent", atom("bob"), atom("tom"))),
        ]
        
        rules = [
            Rule(compound("grandparent", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]),
        ]
        
        kb.add_learned_knowledge(facts, rules)
        print(f"   Added {len(facts)} facts and {len(rules)} rules")
        
        # Setup validator
        print("\n2. Setting up knowledge validator...")
        validator = KnowledgeValidator()
        validator.create_standard_tests(kb.kb1)
        print(f"   Created {len(validator.tests)} validation tests")
        
        # Run validation on original KB
        print("\n3. Validating original knowledge base...")
        original_report = validator.validate(kb.kb1)
        print(f"   Tests run: {original_report.total_tests}")
        print(f"   Tests passed: {original_report.passed_tests}")
        print(f"   Tests failed: {original_report.failed_tests}")
        print(f"   Success rate: {original_report.success_rate:.1f}%")
        print(f"   Execution time: {original_report.execution_time_ms:.1f}ms")
        
        if original_report.results:
            print("   Test results:")
            for result in original_report.results:
                status = "PASS" if result.passed else "FAIL"
                print(f"     {status}: {result.test_name} ({result.execution_time_ms:.1f}ms)")
        
        # Modify knowledge base
        print("\n4. Modifying knowledge base...")
        new_facts = [Fact(compound("parent", atom("alice"), atom("sam")))]
        kb.add_learned_knowledge(new_facts, [])
        print(f"   Added {len(new_facts)} new facts")
        
        # Validate changes
        print("\n5. Validating knowledge changes...")
        change_report = validator.validate_knowledge_change(kb.kb1, kb.kb1)
        print(f"   Change validation - Success rate: {change_report.success_rate:.1f}%")
        
        if change_report.critical_failures:
            print("   Critical failures:")
            for failure in change_report.critical_failures:
                print(f"     - {failure.test_name}: {failure.message}")
        
        # Quick validation
        print("\n6. Quick validation check...")
        quick_result = validator.quick_validate(kb.kb1)
        print(f"   Quick validation result: {'PASS' if quick_result else 'FAIL'}")
        
    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary storage")


def run_all_demos():
    """Run all persistent learning demos"""
    print("DreamLog Persistent Learning System Demo")
    print("="*60)
    print("This demo showcases the complete persistent learning architecture")
    print("including dual knowledge bases, conflict resolution, sleep cycles,")
    print("validation, and high-level APIs.")
    print()
    
    try:
        demo_basic_persistent_learning()
        demo_conflict_resolution()
        demo_sleep_cycles()
        demo_learning_api()
        demo_validation_system()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print()
        print("The persistent learning system demonstrates:")
        print("✓ Dual knowledge base architecture (KB_1 learned + KB_2 user)")
        print("✓ Automatic conflict detection and resolution")
        print("✓ Sleep cycle optimization and knowledge compression")
        print("✓ Comprehensive validation and quality assurance")
        print("✓ High-level API for safe knowledge injection")
        print("✓ Performance monitoring and progress tracking")
        print()
        print("The system provides a robust foundation for continuous")
        print("learning and knowledge refinement in DreamLog.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_demos()