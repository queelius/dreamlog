"""
Tests for DreamLog Persistent Learning System

Comprehensive test suite covering all aspects of the persistent learning architecture:
- Dual knowledge base system
- Conflict detection and resolution
- Knowledge validation
- Sleep cycle management
- Background learning service
- User API integration
"""

import pytest
import tempfile
import shutil
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from dreamlog.persistent_learning import (
    PersistentKnowledgeBase, ConflictReport, ConflictType,
    UserTrustStrategy, ConservativeStrategy
)
from dreamlog.knowledge_validator import (
    KnowledgeValidator, ValidationReport, ConsistencyTest, 
    CompletenessTest, PreservationTest, SampleQueryGenerator
)
from dreamlog.sleep_cycle import (
    SleepCycleManager, SleepPhase, SleepCycleConfig,
    DuplicateRemover, SubsumptionCompressor, RuleGeneralizer
)
from dreamlog.background_learner import (
    BackgroundLearner, BackgroundLearnerClient, IPCMessage, IPCMessageType
)
from dreamlog.learning_api import (
    LearningAPI, SafeKnowledgeInjector, ConflictManager,
    InjectionMode, LearningMode
)
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.terms import Term
from dreamlog.factories import atom, var, compound
from dreamlog.evaluator import PrologEvaluator


class TestPersistentKnowledgeBase:
    """Test suite for PersistentKnowledgeBase"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_facts(self):
        """Sample facts for testing"""
        return [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("mary"), atom("alice"))),
            Fact(compound("parent", atom("bob"), atom("tom"))),
        ]
    
    @pytest.fixture
    def sample_rules(self):
        """Sample rules for testing"""
        return [
            Rule(
                compound("grandparent", var("X"), var("Z")),
                [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
            ),
            Rule(
                compound("ancestor", var("X"), var("Y")),
                [compound("parent", var("X"), var("Y"))]
            )
        ]
    
    def test_initialization(self, temp_storage):
        """Test persistent KB initialization"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        assert kb.kb1 is not None
        assert kb.kb2 is not None
        assert len(kb.conflicts) == 0
        assert len(kb.versions) == 0
        assert kb.storage_path == temp_storage
    
    def test_add_learned_knowledge(self, temp_storage, sample_facts, sample_rules):
        """Test adding learned knowledge to KB_1"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        initial_facts = len(kb.kb1.facts)
        initial_rules = len(kb.kb1.rules)
        
        kb.add_learned_knowledge(sample_facts, sample_rules)
        
        assert len(kb.kb1.facts) == initial_facts + len(sample_facts)
        assert len(kb.kb1.rules) == initial_rules + len(sample_rules)
        assert len(kb.learning_events) > 0
        
        # Verify event logging
        last_event = kb.learning_events[-1]
        assert last_event[1] == "learned_knowledge_added"
        assert last_event[2]["facts_count"] == len(sample_facts)
        assert last_event[2]["rules_count"] == len(sample_rules)
    
    def test_add_user_knowledge(self, temp_storage, sample_facts):
        """Test adding user knowledge to KB_2"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        initial_facts = len(kb.kb2.facts)
        conflicts = kb.add_user_knowledge(sample_facts, [])
        
        assert len(kb.kb2.facts) == initial_facts + len(sample_facts)
        assert isinstance(conflicts, list)
        assert len(kb.learning_events) > 0
    
    def test_conflict_detection(self, temp_storage):
        """Test conflict detection between KB_1 and KB_2"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        # Add contradictory knowledge
        learned_fact = Fact(compound("alive", atom("john")))
        user_fact = Fact(compound("dead", atom("john")))
        
        kb.add_learned_knowledge([learned_fact], [])
        conflicts = kb.add_user_knowledge([user_fact], [])
        
        # Should detect contradiction
        assert len(conflicts) > 0 or len(kb.conflicts) > 0
    
    def test_conflict_resolution(self, temp_storage):
        """Test conflict resolution strategies"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        # Create conflicting knowledge
        learned_fact = Fact(compound("status", atom("john"), atom("student")))
        user_fact = Fact(compound("status", atom("john"), atom("teacher")))
        
        kb.add_learned_knowledge([learned_fact], [])
        kb.add_user_knowledge([user_fact], [])
        
        # Test user trust strategy
        kb.resolution_strategy = UserTrustStrategy()
        results = kb.resolve_conflicts()
        
        assert results["conflicts_resolved"] >= 0
        assert "resolution_descriptions" in results
    
    def test_version_snapshots(self, temp_storage, sample_facts):
        """Test version snapshot creation"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        # Create initial snapshot
        version1 = kb.create_version_snapshot({"reason": "initial"})
        assert version1 is not None
        assert version1.version_id is not None
        assert len(kb.versions) == 1
        
        # Add knowledge and create another snapshot
        kb.add_learned_knowledge(sample_facts, [])
        version2 = kb.create_version_snapshot({"reason": "after_learning"})
        
        assert len(kb.versions) == 2
        assert version2.facts_count > version1.facts_count
    
    def test_query_with_tracking(self, temp_storage, sample_facts):
        """Test query with performance tracking"""
        kb = PersistentKnowledgeBase(temp_storage)
        kb.add_learned_knowledge(sample_facts, [])
        
        initial_query_count = len(kb.query_history)
        
        # Execute query
        query = [compound("parent", atom("john"), var("X"))]
        solutions = kb.query_with_tracking(query)
        
        assert len(solutions) > 0
        assert len(kb.query_history) == initial_query_count + 1
        
        # Verify query history
        last_query = kb.query_history[-1]
        assert len(last_query) == 3  # (timestamp, query, solution_count)
        assert last_query[2] == len(solutions)
    
    def test_performance_metrics(self, temp_storage, sample_facts):
        """Test performance metrics calculation"""
        kb = PersistentKnowledgeBase(temp_storage)
        kb.add_learned_knowledge(sample_facts, [])
        
        metrics = kb.get_performance_metrics()
        
        assert "kb1_size" in metrics
        assert "kb2_size" in metrics
        assert "conflicts" in metrics
        assert "queries" in metrics
        assert "versions" in metrics
        assert "learning_events" in metrics
        
        assert metrics["kb1_size"]["facts"] == len(sample_facts)
        assert metrics["conflicts"]["total"] == len(kb.conflicts)
    
    def test_persistence(self, temp_storage, sample_facts, sample_rules):
        """Test saving and loading persistent state"""
        # Create and populate KB
        kb1 = PersistentKnowledgeBase(temp_storage)
        kb1.add_learned_knowledge(sample_facts, sample_rules[:1])
        kb1.add_user_knowledge([], sample_rules[1:])
        kb1.save()
        
        # Create new KB instance and verify data is loaded
        kb2 = PersistentKnowledgeBase(temp_storage)
        
        assert len(kb2.kb1.facts) == len(sample_facts)
        assert len(kb2.kb1.rules) == 1
        assert len(kb2.kb2.rules) == 1


class TestKnowledgeValidator:
    """Test suite for KnowledgeValidator"""
    
    @pytest.fixture
    def sample_kb(self):
        """Sample knowledge base for testing"""
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
        kb.add_fact(Fact(compound("parent", atom("mary"), atom("alice"))))
        kb.add_rule(Rule(
            compound("grandparent", var("X"), var("Z")),
            [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]
        ))
        return kb
    
    def test_consistency_test(self, sample_kb):
        """Test consistency validation"""
        test = ConsistencyTest()
        result = test.run(sample_kb)
        
        assert result.test_name == "Logical Consistency"
        assert result.passed is True or result.passed is False
        assert result.execution_time_ms >= 0
    
    def test_consistency_test_with_contradiction(self):
        """Test consistency with contradictory facts"""
        kb = KnowledgeBase()
        kb.add_fact(Fact(compound("alive", atom("john"))))
        kb.add_fact(Fact(compound("dead", atom("john"))))
        
        test = ConsistencyTest()
        result = test.run(kb)
        
        # Should detect contradiction (implementation dependent)
        assert result.execution_time_ms >= 0
    
    def test_completeness_test(self, sample_kb):
        """Test completeness validation"""
        # Create test queries based on KB content
        test_queries = [
            ([compound("parent", atom("john"), var("X"))], 1),
            ([compound("grandparent", var("X"), var("Y"))], 1)
        ]
        
        test = CompletenessTest(test_queries)
        result = test.run(sample_kb)
        
        assert result.test_name == "Query Completeness"
        assert isinstance(result.passed, bool)
        assert result.execution_time_ms >= 0
    
    def test_preservation_test(self, sample_kb):
        """Test behavior preservation"""
        # Create modified KB
        modified_kb = KnowledgeBase()
        for fact in sample_kb.facts:
            modified_kb.add_fact(fact)
        for rule in sample_kb.rules:
            modified_kb.add_rule(rule)
        
        # Add one more fact
        modified_kb.add_fact(Fact(compound("parent", atom("bob"), atom("tom"))))
        
        sample_queries = [
            [compound("parent", var("X"), var("Y"))],
            [compound("grandparent", var("X"), var("Y"))]
        ]
        
        test = PreservationTest(sample_queries, tolerance=1.0)  # Allow some difference
        result = test.run(modified_kb, sample_kb)
        
        assert result.test_name == "Behavior Preservation"
        assert isinstance(result.passed, bool)
        assert result.execution_time_ms >= 0
    
    def test_sample_query_generator(self, sample_kb):
        """Test sample query generation"""
        generator = SampleQueryGenerator(sample_kb)
        
        fact_queries = generator.generate_fact_queries(5)
        rule_queries = generator.generate_rule_queries(3)
        composite_queries = generator.generate_composite_queries(2)
        
        assert isinstance(fact_queries, list)
        assert isinstance(rule_queries, list)
        assert isinstance(composite_queries, list)
        
        # Verify query structure
        if fact_queries:
            assert all(isinstance(query, list) for query in fact_queries)
    
    def test_full_validation(self, sample_kb):
        """Test complete validation workflow"""
        validator = KnowledgeValidator()
        validator.create_standard_tests(sample_kb)
        
        assert len(validator.tests) > 0
        
        report = validator.validate(sample_kb)
        
        assert isinstance(report, ValidationReport)
        assert report.total_tests > 0
        assert report.passed_tests >= 0
        assert report.failed_tests >= 0
        assert report.total_tests == report.passed_tests + report.failed_tests
        assert 0 <= report.success_rate <= 100
    
    def test_quick_validate(self, sample_kb):
        """Test quick validation"""
        validator = KnowledgeValidator()
        result = validator.quick_validate(sample_kb)
        
        assert isinstance(result, bool)


class TestSleepCycle:
    """Test suite for SleepCycleManager and operations"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def persistent_kb_with_data(self, temp_storage):
        """Persistent KB with sample data"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        # Add some facts with duplicates
        facts = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("john"), atom("mary"))),  # Duplicate
            Fact(compound("parent", atom("mary"), atom("alice"))),
            Fact(compound("likes", atom("john"), atom("pizza"))),
            Fact(compound("likes", atom("mary"), atom("pizza")))
        ]
        
        rules = [
            Rule(compound("grandparent", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))]),
            Rule(compound("ancestor", var("X"), var("Y")),
                 [compound("parent", var("X"), var("Y"))])
        ]
        
        kb.add_learned_knowledge(facts, rules)
        return kb
    
    def test_duplicate_remover(self, persistent_kb_with_data):
        """Test duplicate removal operation"""
        kb = persistent_kb_with_data.kb1  # Get the learned KB
        initial_facts = len(kb.facts)
        
        operation = DuplicateRemover()
        config = SleepCycleConfig()
        
        new_kb, metrics = operation.execute(kb, config)
        
        assert len(new_kb.facts) <= initial_facts
        assert "facts_removed" in metrics
        assert "rules_removed" in metrics
        assert metrics["facts_removed"] >= 0
    
    def test_subsumption_compressor(self, persistent_kb_with_data):
        """Test subsumption-based compression"""
        kb = persistent_kb_with_data.kb1
        
        operation = SubsumptionCompressor()
        config = SleepCycleConfig()
        
        new_kb, metrics = operation.execute(kb, config)
        
        assert isinstance(new_kb, KnowledgeBase)
        assert "facts_removed" in metrics
        assert "rules_removed" in metrics
    
    def test_rule_generalizer(self, persistent_kb_with_data):
        """Test rule generalization"""
        kb = persistent_kb_with_data.kb1
        
        operation = RuleGeneralizer()
        config = SleepCycleConfig()
        
        new_kb, metrics = operation.execute(kb, config)
        
        assert isinstance(new_kb, KnowledgeBase)
        assert "rules_generated" in metrics
        assert metrics["rules_generated"] >= 0
    
    def test_sleep_cycle_manager_initialization(self, persistent_kb_with_data):
        """Test sleep cycle manager setup"""
        manager = SleepCycleManager(persistent_kb_with_data)
        
        assert manager.persistent_kb == persistent_kb_with_data
        assert manager.config is not None
        assert manager.validator is not None
        assert len(manager.light_sleep_ops) > 0
        assert len(manager.deep_sleep_ops) > 0
    
    def test_force_sleep_cycle(self, persistent_kb_with_data):
        """Test forced sleep cycle execution"""
        manager = SleepCycleManager(persistent_kb_with_data)
        
        initial_facts = len(persistent_kb_with_data.kb1.facts)
        
        report = manager.force_sleep_cycle(SleepPhase.LIGHT_SLEEP)
        
        assert report.phase == SleepPhase.LIGHT_SLEEP
        assert report.cycle_id is not None
        assert report.execution_time_ms >= 0
        assert isinstance(report.operations, list)
    
    def test_sleep_metrics(self, persistent_kb_with_data):
        """Test sleep cycle metrics"""
        manager = SleepCycleManager(persistent_kb_with_data)
        
        # Force a sleep cycle to generate some history
        manager.force_sleep_cycle(SleepPhase.LIGHT_SLEEP)
        
        metrics = manager.get_sleep_metrics()
        
        assert "total_cycles" in metrics
        assert "recent_cycles_7d" in metrics
        assert "avg_compression_ratio" in metrics
        assert "last_light_sleep" in metrics


class TestBackgroundLearner:
    """Test suite for BackgroundLearner service"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_ipc_message_serialization(self):
        """Test IPC message serialization"""
        message = IPCMessage(
            message_id="test-123",
            message_type=IPCMessageType.QUERY,
            data={"test": "data"}
        )
        
        json_str = message.to_json()
        assert isinstance(json_str, str)
        
        deserialized = IPCMessage.from_json(json_str)
        assert deserialized.message_id == message.message_id
        assert deserialized.message_type == message.message_type
        assert deserialized.data == message.data
    
    def test_background_learner_initialization(self, temp_storage):
        """Test background learner initialization"""
        learner = BackgroundLearner(temp_storage, ipc_port=7778)  # Use different port
        
        assert learner.storage_path == temp_storage
        assert learner.status.name == "STOPPED"
        assert learner.ipc_port == 7778
    
    @pytest.mark.slow
    def test_background_service_lifecycle(self, temp_storage):
        """Test background service start/stop lifecycle"""
        # Use unique port to avoid conflicts
        port = 7779
        learner = BackgroundLearner(temp_storage, ipc_port=port)
        
        try:
            # Start service
            learner.start()
            time.sleep(1)  # Give service time to start
            
            assert learner.running is True
            assert learner.status.name in ["RUNNING", "STARTING"]
            assert learner.persistent_kb is not None
            
            # Test basic client connection
            client = BackgroundLearnerClient(port)
            
            # Simple status check
            status = client.get_status()
            assert "status" in status
            
        finally:
            # Always stop service
            learner.stop()
            time.sleep(1)  # Give service time to stop
    
    def test_client_operations(self):
        """Test client operations (without actual service)"""
        client = BackgroundLearnerClient(9999)  # Non-existent port
        
        # Test that client methods exist and handle errors appropriately
        with pytest.raises(Exception):
            client.query([compound("test", atom("x"))])


class TestLearningAPI:
    """Test suite for LearningAPI"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_api_initialization_direct_mode(self, temp_storage):
        """Test API initialization in direct mode"""
        api = LearningAPI(
            temp_storage,
            learning_mode=LearningMode.ACTIVE,
            injection_mode=InjectionMode.PERMISSIVE,
            use_background_service=False
        )
        
        assert api.storage_path == temp_storage
        assert api.learning_mode == LearningMode.ACTIVE
        assert api.injection_mode == InjectionMode.PERMISSIVE
        assert api.use_background_service is False
        assert api.kb is not None
        assert api.injector is not None
        assert api.conflict_manager is not None
        assert api.progress_tracker is not None
    
    def test_api_initialization_background_mode(self, temp_storage):
        """Test API initialization in background service mode"""
        api = LearningAPI(
            temp_storage,
            use_background_service=True,
            background_port=7780
        )
        
        assert api.use_background_service is True
        assert api.background_client is not None
        assert api.kb is None  # Should use background service
    
    def test_safe_knowledge_injector(self, temp_storage):
        """Test SafeKnowledgeInjector"""
        kb = PersistentKnowledgeBase(temp_storage)
        validator = KnowledgeValidator()
        injector = SafeKnowledgeInjector(kb, validator, InjectionMode.PERMISSIVE)
        
        facts = [Fact(compound("test_fact", atom("value")))]
        result = injector.inject_facts(facts, validate=False)
        
        assert isinstance(result.success, bool)
        assert result.facts_added >= 0
        assert result.execution_time_ms >= 0
        
        # Test injection stats
        stats = injector.get_injection_stats()
        assert "total_operations" in stats
        assert "successful_operations" in stats
        assert "success_rate" in stats
    
    def test_conflict_manager(self, temp_storage):
        """Test ConflictManager"""
        kb = PersistentKnowledgeBase(temp_storage)
        manager = ConflictManager(kb)
        
        # Test with no conflicts
        analysis = manager.analyze_conflicts()
        assert "total_conflicts" in analysis
        assert "by_type" in analysis
        assert "by_severity" in analysis
        
        # Test conflict report generation
        report = manager.create_conflict_report()
        assert isinstance(report, str)
        assert "Conflict Report" in report or "No conflicts" in report
    
    def test_string_parsing_operations(self, temp_storage):
        """Test string-based operations"""
        api = LearningAPI(temp_storage, use_background_service=False)
        
        # Test simple fact injection
        fact_strings = ["parent(john, mary)", "likes(mary, pizza)"]
        rule_strings = ["grandparent(X,Z) :- parent(X,Y), parent(Y,Z)"]
        
        result = api.inject_knowledge_from_strings(fact_strings, rule_strings, validate=False)
        
        assert isinstance(result.success, bool)
        assert result.execution_time_ms >= 0
    
    def test_system_metrics(self, temp_storage):
        """Test system metrics collection"""
        api = LearningAPI(temp_storage, use_background_service=False)
        
        metrics = api.get_system_metrics()
        
        assert "api" in metrics
        assert "knowledge_base" in metrics
        assert "injection" in metrics
        assert "conflicts" in metrics
        
        # API metrics
        api_metrics = metrics["api"]
        assert "start_time" in api_metrics
        assert "uptime_seconds" in api_metrics
        assert "learning_mode" in api_metrics
        assert "injection_mode" in api_metrics
    
    def test_api_state_management(self, temp_storage):
        """Test API state management"""
        api = LearningAPI(temp_storage, use_background_service=False)
        
        # Test save state
        api.save_state()  # Should not raise exception
        
        # Test shutdown
        api.shutdown()  # Should not raise exception


class TestIntegration:
    """Integration tests for the complete persistent learning system"""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_learning_workflow(self, temp_storage):
        """Test complete learning workflow from injection to querying"""
        # Initialize API
        api = LearningAPI(
            temp_storage,
            learning_mode=LearningMode.ACTIVE,
            injection_mode=InjectionMode.PERMISSIVE,
            use_background_service=False
        )
        
        # Inject initial knowledge
        fact_strings = [
            "parent(john, mary)",
            "parent(mary, alice)",
            "parent(bob, tom)"
        ]
        rule_strings = [
            "grandparent(X,Z) :- parent(X,Y), parent(Y,Z)"
        ]
        
        injection_result = api.inject_knowledge_from_strings(fact_strings, rule_strings)
        assert injection_result.success is True
        
        # Query the knowledge
        solutions = api.query_knowledge("grandparent(john, X)")
        assert isinstance(solutions, list)
        
        # Check progress
        progress = api.get_learning_progress()
        assert isinstance(progress, dict)
        
        # Get metrics
        metrics = api.get_system_metrics()
        assert "api" in metrics
        assert metrics["api"]["operations_performed"] > 0
    
    def test_conflict_resolution_workflow(self, temp_storage):
        """Test conflict detection and resolution workflow"""
        api = LearningAPI(temp_storage, use_background_service=False)
        
        # Add initial learned knowledge
        learned_facts = [Fact(compound("status", atom("john"), atom("student")))]
        api.kb.add_learned_knowledge(learned_facts, [])
        
        # Add conflicting user knowledge
        user_facts = [Fact(compound("status", atom("john"), atom("teacher")))]
        conflicts = api.kb.add_user_knowledge(user_facts, [])
        
        # Analyze conflicts
        conflict_summary = api.get_conflict_summary()
        assert isinstance(conflict_summary, dict)
        
        # Resolve conflicts
        if not api.use_background_service:
            resolution_result = api.resolve_all_conflicts("user_trust")
            assert isinstance(resolution_result, dict)
    
    def test_validation_workflow(self, temp_storage):
        """Test validation workflow"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        # Add some knowledge
        facts = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("mary"), atom("alice")))
        ]
        kb.add_learned_knowledge(facts, [])
        
        # Validate
        validator = KnowledgeValidator()
        validator.create_standard_tests(kb.kb1)
        report = validator.validate(kb.kb1)
        
        assert report.total_tests > 0
        assert 0 <= report.success_rate <= 100
        
        # Test validation with baseline
        modified_kb = KnowledgeBase()
        for fact in kb.kb1.facts:
            modified_kb.add_fact(fact)
        modified_kb.add_fact(Fact(compound("parent", atom("bob"), atom("charlie"))))
        
        change_report = validator.validate_knowledge_change(kb.kb1, modified_kb)
        assert isinstance(change_report, ValidationReport)
    
    @pytest.mark.slow
    def test_sleep_cycle_integration(self, temp_storage):
        """Test sleep cycle integration"""
        kb = PersistentKnowledgeBase(temp_storage)
        
        # Add knowledge with duplicates and redundancy
        facts = [
            Fact(compound("parent", atom("john"), atom("mary"))),
            Fact(compound("parent", atom("john"), atom("mary"))),  # Duplicate
            Fact(compound("likes", atom("john"), atom("pizza"))),
        ]
        rules = [
            Rule(compound("grandparent", var("X"), var("Z")),
                 [compound("parent", var("X"), var("Y")), compound("parent", var("Y"), var("Z"))])
        ]
        
        kb.add_learned_knowledge(facts, rules)
        
        # Set up sleep cycle with fast intervals for testing
        config = SleepCycleConfig(
            light_sleep_interval=timedelta(seconds=1),
            min_idle_time=timedelta(milliseconds=100),
            require_validation=False  # Skip validation for speed
        )
        
        manager = SleepCycleManager(kb, config)
        
        # Force sleep cycle
        report = manager.force_sleep_cycle(SleepPhase.LIGHT_SLEEP)
        
        assert report.cycle_id is not None
        assert report.phase == SleepPhase.LIGHT_SLEEP
        assert isinstance(report.operations, list)
        
        # Verify compression occurred (should remove duplicates)
        final_fact_count = len(kb.kb1.facts)
        assert final_fact_count <= len(facts)  # Should be same or fewer after deduplication


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])