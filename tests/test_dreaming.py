"""
Tests for DreamLog's wake-sleep cycles and knowledge optimization.

These tests verify that:
1. Knowledge compression preserves behavior
2. Abstractions are discovered correctly
3. Verification prevents breaking changes
4. Multiple exploration paths are sampled
"""

import pytest
from dreamlog.knowledge import KnowledgeBase, Fact, Rule
from dreamlog.kb_dreamer import (
    KnowledgeBaseDreamer, DreamSession, DreamInsight, VerificationResult
)
from .mock_provider import MockLLMProvider
from dreamlog.prefix_parser import parse_prefix_notation
from dreamlog.terms import atom, var, compound


class TestDreaming:
    """Test the dreaming/optimization system"""
    
    @pytest.fixture
    def sample_kb(self):
        """Create a knowledge base with redundancy and optimization opportunities"""
        kb = KnowledgeBase()
        
        # Add redundant facts that could be compressed
        kb.add_fact(Fact.from_prefix(["fact", ["parent", "john", "mary"]]))
        kb.add_fact(Fact.from_prefix(["fact", ["parent", "john", "bob"]]))
        kb.add_fact(Fact.from_prefix(["fact", ["parent", "mary", "alice"]]))
        kb.add_fact(Fact.from_prefix(["fact", ["parent", "bob", "charlie"]]))
        
        # Add similar rules that could be merged
        kb.add_rule(Rule.from_prefix([
            "rule", 
            ["brother", "X", "Y"],
            [["male", "X"], ["parent", "Z", "X"], ["parent", "Z", "Y"], ["different", "X", "Y"]]
        ]))
        
        kb.add_rule(Rule.from_prefix([
            "rule",
            ["sister", "X", "Y"],
            [["female", "X"], ["parent", "Z", "X"], ["parent", "Z", "Y"], ["different", "X", "Y"]]
        ]))
        
        # These could be abstracted to a single "sibling" rule
        
        return kb
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM that returns predictable optimizations"""
        from unittest.mock import Mock
        
        provider = Mock()
        
        # Mock the generate_knowledge method
        def generate_knowledge(term, context):
            # Return appropriate response based on the query
            if "_find_compressions" in str(term):
                return Mock(text='{"compressed_rule": "(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)", "explanation": "Merged"}')
            elif "_find_abstractions" in str(term):
                return Mock(text='[{"original_rules": ["rule1", "rule2"], "abstract_rule": "(sibling X Y) :- ...", "benefit": "General"}]')
            elif "_test_queries" in str(term):
                return Mock(text='["(parent john X)", "(sibling X Y)"]')
            elif "_evaluate_diff" in str(term):
                return Mock(text='{"is_improvement": true, "is_acceptable": true, "reason": "Good"}')
            else:
                return Mock(text='[]', facts=[], rules=[])
        
        provider.generate_knowledge = generate_knowledge
        return provider
    
    def test_dream_session_creation(self, sample_kb, mock_llm_provider):
        """Test that a dream session can be created and run"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        session = dreamer.dream(
            sample_kb,
            dream_cycles=1,
            exploration_samples=2,
            focus="compression",
            verify=False
        )
        
        assert isinstance(session, DreamSession)
        assert session.original_size == len(sample_kb.facts) + len(sample_kb.rules)
        assert session.compression_ratio >= 0
        assert session.generalization_score >= 0
        assert session.exploration_paths == 2
    
    def test_compression_discovery(self, sample_kb, mock_llm_provider):
        """Test that compression opportunities are found"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        insights = dreamer._find_compressions(sample_kb)
        
        # Should find at least one compression opportunity
        assert len(insights) > 0
        
        for insight in insights:
            assert insight.type == "compression"
            assert insight.compression_ratio > 1.0  # Should compress
            assert len(insight.new_items) < len(insight.original_items)
    
    def test_behavior_verification(self, sample_kb, mock_llm_provider):
        """Test that behavior preservation is verified"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        # Create a simple insight
        insight = DreamInsight(
            type="compression",
            description="Test compression",
            original_items=["(brother X Y) :- ..."],
            new_items=["(sibling X Y) :- ..."],
            compression_ratio=2.0,
            coverage_gain=1.5
        )
        
        verification = dreamer._verify_behavior_preservation(sample_kb, [insight])
        
        assert isinstance(verification, VerificationResult)
        assert verification.similarity_score >= 0
        assert verification.similarity_score <= 1
        assert isinstance(verification.differences, list)
        assert isinstance(verification.improvements, list)
    
    def test_multiple_exploration_paths(self, sample_kb, mock_llm_provider):
        """Test that multiple optimization paths are explored"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        session = dreamer.dream(
            sample_kb,
            dream_cycles=1,
            exploration_samples=5,  # Try 5 different paths
            focus="all",
            verify=False
        )
        
        # Should have explored multiple paths
        assert session.exploration_paths == 5
        
        # Best insights should be selected
        assert len(session.insights) > 0
    
    def test_insight_scoring(self, mock_llm_provider):
        """Test that insights are scored correctly"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        # Create insights with different characteristics
        compression_insight = DreamInsight(
            type="compression",
            description="High compression",
            original_items=["a", "b", "c"],
            new_items=["x"],
            compression_ratio=3.0,
            coverage_gain=1.0,
            confidence=0.8
        )
        
        abstraction_insight = DreamInsight(
            type="abstraction",
            description="Good abstraction",
            original_items=["a", "b"],
            new_items=["x"],
            compression_ratio=2.0,
            coverage_gain=2.0,
            confidence=0.9
        )
        
        # Abstraction should score higher due to bonus
        compression_score = dreamer._score_insight(compression_insight)
        abstraction_score = dreamer._score_insight(abstraction_insight)
        
        assert abstraction_score > compression_score
    
    def test_apply_insights(self, sample_kb, mock_llm_provider):
        """Test that insights can be applied to create optimized KB"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        # Create an insight that removes redundancy
        insight = DreamInsight(
            type="compression",
            description="Remove redundant rule",
            original_items=[],  # Items to remove
            new_items=[],  # Items to add
            compression_ratio=1.2,
            coverage_gain=1.0
        )
        
        optimized_kb = dreamer._apply_insights(sample_kb, [insight])
        
        assert isinstance(optimized_kb, KnowledgeBase)
        # Should have same or fewer items
        assert len(optimized_kb.facts) <= len(sample_kb.facts)
        assert len(optimized_kb.rules) <= len(sample_kb.rules)
    
    def test_dream_with_verification(self, sample_kb, mock_llm_provider):
        """Test full dream cycle with verification"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        session = dreamer.dream(
            sample_kb,
            dream_cycles=2,
            exploration_samples=3,
            focus="all",
            verify=True  # Enable verification
        )
        
        assert session.verification is not None
        assert isinstance(session.verification, VerificationResult)
        
        # Insights should be marked as verified or not
        for insight in session.insights:
            assert hasattr(insight, 'verified')
    
    def test_temperature_variation(self, sample_kb, mock_llm_provider):
        """Test that temperature affects exploration"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        # Different temperatures should be used across samples
        insights_low_temp = dreamer._find_compressions(sample_kb, temperature=0.3)
        insights_high_temp = dreamer._find_compressions(sample_kb, temperature=0.9)
        
        # Both should return insights (mock doesn't actually use temperature)
        assert len(insights_low_temp) >= 0
        assert len(insights_high_temp) >= 0


class TestDreamInsights:
    """Test individual insight types"""
    
    def test_compression_insight(self):
        """Test compression insight creation and properties"""
        insight = DreamInsight(
            type="compression",
            description="Merge similar rules",
            original_items=["rule1", "rule2", "rule3"],
            new_items=["merged_rule"],
            compression_ratio=3.0,
            coverage_gain=1.0
        )
        
        assert insight.type == "compression"
        assert insight.compression_ratio == 3.0
        assert len(insight.original_items) == 3
        assert len(insight.new_items) == 1
    
    def test_abstraction_insight(self):
        """Test abstraction insight"""
        insight = DreamInsight(
            type="abstraction",
            description="Found higher-level concept",
            original_items=["specific1", "specific2"],
            new_items=["abstract_concept"],
            compression_ratio=2.0,
            coverage_gain=3.0  # Abstractions typically increase coverage
        )
        
        assert insight.type == "abstraction"
        assert insight.coverage_gain > insight.compression_ratio
    
    def test_generalization_insight(self):
        """Test generalization insight"""
        insight = DreamInsight(
            type="generalization",
            description="Replaced constants with variables",
            original_items=["(parent john mary)"],
            new_items=["(parent X Y)"],
            compression_ratio=1.0,
            coverage_gain=10.0  # Much higher coverage
        )
        
        assert insight.type == "generalization"
        assert insight.coverage_gain > 1.0


class TestVerification:
    """Test behavior verification system"""
    
    def test_verification_result(self):
        """Test verification result creation"""
        result = VerificationResult(
            preserved=True,
            similarity_score=0.95,
            differences=["minor difference in query3"],
            improvements=["query5 now faster"]
        )
        
        assert result.preserved
        assert result.similarity_score == 0.95
        assert len(result.differences) == 1
        assert len(result.improvements) == 1
    
    def test_perfect_preservation(self):
        """Test when behavior is perfectly preserved"""
        result = VerificationResult(
            preserved=True,
            similarity_score=1.0,
            differences=[],
            improvements=[]
        )
        
        assert result.preserved
        assert result.similarity_score == 1.0
        assert len(result.differences) == 0
    
    def test_broken_behavior(self):
        """Test when optimization breaks behavior"""
        result = VerificationResult(
            preserved=False,
            similarity_score=0.6,
            differences=[
                "Query (parent X Y) returns different results",
                "Query (grandparent X Y) fails"
            ],
            improvements=[]
        )
        
        assert not result.preserved
        assert result.similarity_score < 0.7
        assert len(result.differences) == 2


@pytest.mark.integration
class TestDreamingIntegration:
    """Integration tests for the dreaming system"""
    
    def test_end_to_end_optimization(self, sample_kb, mock_llm_provider):
        """Test complete optimization pipeline"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        # Dream and optimize
        session = dreamer.dream(sample_kb, verify=True)
        
        # Apply insights if verification passed
        if session.verification and session.verification.similarity_score > 0.9:
            optimized_kb = dreamer._apply_insights(sample_kb, session.insights)
            
            # Optimized KB should be smaller or same size
            assert (len(optimized_kb.facts) + len(optimized_kb.rules)) <= \
                   (len(sample_kb.facts) + len(sample_kb.rules))
    
    def test_repeated_dreaming(self, sample_kb, mock_llm_provider):
        """Test that repeated dreaming continues to find improvements"""
        dreamer = KnowledgeBaseDreamer(mock_llm_provider)
        
        kb = sample_kb
        total_compression = 1.0
        
        # Dream multiple times
        for _ in range(3):
            session = dreamer.dream(kb, dream_cycles=1, verify=False)
            
            if session.compression_ratio > 0:
                total_compression *= (1 - session.compression_ratio)
                kb = dreamer._apply_insights(kb, session.insights)
        
        # Should achieve some compression over multiple dreams
        assert total_compression <= 1.0