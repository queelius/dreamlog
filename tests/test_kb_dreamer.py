"""
Tests for KnowledgeBaseDreamer - Sleep phase knowledge optimization.

This test suite follows TDD principles to build the dreaming system that:
- Discovers patterns and redundancies in knowledge bases
- Compresses knowledge through abstraction
- Generalizes rules for broader applicability
- Verifies behavior preservation
"""

import pytest
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.knowledge import KnowledgeBase
from tests.mock_provider import MockLLMProvider


class TestKnowledgeBaseDreamerBasics:
    """Test basic initialization and structure."""

    def test_create_dreamer_instance(self):
        """Test that we can create a KnowledgeBaseDreamer with an LLM provider."""
        # Given: An LLM provider
        provider = MockLLMProvider()

        # When: We create a dreamer
        dreamer = KnowledgeBaseDreamer(provider)

        # Then: It should be created successfully
        assert dreamer is not None
        assert isinstance(dreamer, KnowledgeBaseDreamer)

    def test_dream_returns_session(self):
        """Test that calling dream() returns a dream session object."""
        # Given: A dreamer and a knowledge base
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        # When: We run a dream cycle
        session = dreamer.dream(kb)

        # Then: We should get a session object
        assert session is not None

    def test_session_has_required_attributes(self):
        """Test that dream session has all required attributes from the API."""
        # Given: A dreamer and knowledge base
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        # When: We run a dream cycle
        session = dreamer.dream(kb)

        # Then: Session should have all required attributes
        assert hasattr(session, 'exploration_paths'), "Session missing exploration_paths"
        assert hasattr(session, 'insights'), "Session missing insights"
        assert hasattr(session, 'compression_ratio'), "Session missing compression_ratio"
        assert hasattr(session, 'generalization_score'), "Session missing generalization_score"
        assert hasattr(session, 'verification'), "Session missing verification"

    def test_session_attribute_types(self):
        """Test that session attributes have correct types."""
        # Given: A dreamer and knowledge base
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        # When: We run a dream cycle
        session = dreamer.dream(kb)

        # Then: Attributes should have correct types
        assert isinstance(session.exploration_paths, int), "exploration_paths should be int"
        assert isinstance(session.insights, list), "insights should be list"
        assert isinstance(session.compression_ratio, float), "compression_ratio should be float"
        assert isinstance(session.generalization_score, float), "generalization_score should be float"
        assert session.verification is not None, "verification should not be None"


class TestPatternDetection:
    """Test pattern detection and insight discovery."""

    def test_detects_redundant_rules(self):
        """Test detection of redundant rule patterns like father/mother -> parent."""
        # Given: A KB with redundant patterns (father and mother both imply parent)
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        # Add redundant patterns: father(X,Y) -> parent(X,Y) and mother(X,Y) -> parent(X,Y)
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(mother ?X ?Y)")]
        ))

        # When: We run a dream cycle focused on compression
        session = dreamer.dream(kb, focus="compression")

        # Then: It should detect the redundant pattern
        assert len(session.insights) > 0, "Should discover at least one insight"
        compression_insights = [i for i in session.insights if i.type == "compression"]
        assert len(compression_insights) > 0, "Should find compression opportunity"

    def test_insight_structure(self):
        """Test that insights have the correct structure."""
        # Given: A KB with patterns to detect
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(mother ?X ?Y)")]
        ))

        # When: We run a dream cycle
        session = dreamer.dream(kb, focus="compression")

        # Then: Each insight should have required attributes
        if session.insights:
            insight = session.insights[0]
            assert hasattr(insight, 'type')
            assert hasattr(insight, 'description')
            assert hasattr(insight, 'compression_ratio')
            assert hasattr(insight, 'coverage_gain')
            assert hasattr(insight, 'verified')
            assert insight.type in ["abstraction", "compression", "generalization"]

    def test_empty_kb_returns_no_insights(self):
        """Test that an empty KB produces no insights."""
        # Given: An empty knowledge base
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        # When: We run a dream cycle
        session = dreamer.dream(kb)

        # Then: No insights should be found
        assert len(session.insights) == 0, "Empty KB should produce no insights"
        assert session.compression_ratio == 1.0, "No compression for empty KB"


class TestVerification:
    """Test behavior verification after optimization."""

    def test_verification_structure(self):
        """Test that verification results have correct structure."""
        # Given: A knowledge base with rules
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))

        # When: We run a dream cycle with verification
        session = dreamer.dream(kb, verify=True)

        # Then: Verification should have correct structure
        assert hasattr(session.verification, 'preserved')
        assert hasattr(session.verification, 'similarity_score')
        assert hasattr(session.verification, 'improvements')
        assert isinstance(session.verification.preserved, bool)
        assert isinstance(session.verification.similarity_score, float)
        assert isinstance(session.verification.improvements, list)

    def test_can_skip_verification(self):
        """Test that verification can be skipped."""
        # Given: A knowledge base
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))

        # When: We run without verification
        session = dreamer.dream(kb, verify=False)

        # Then: Should still have verification object (for API consistency)
        assert session.verification is not None
        assert session.verification.preserved is True


class TestDreamCycles:
    """Test dream cycle parameters and behavior."""

    def test_respects_focus_parameter(self):
        """Test that dream respects the focus parameter."""
        # Given: A knowledge base with patterns
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(mother ?X ?Y)")]
        ))

        # When: We focus on compression
        session_compression = dreamer.dream(kb, focus="compression")

        # Then: Should find compression insights
        compression_insights = [i for i in session_compression.insights if i.type == "compression"]
        assert len(compression_insights) > 0

        # When: We focus on generalization (not implemented yet, should find nothing)
        session_generalization = dreamer.dream(kb, focus="generalization")

        # Then: Should find no generalization insights (placeholder returns empty)
        generalization_insights = [i for i in session_generalization.insights if i.type == "generalization"]
        assert len(generalization_insights) == 0

    def test_exploration_paths_counted(self):
        """Test that exploration paths are counted correctly."""
        # Given: A knowledge base with patterns
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(mother ?X ?Y)")]
        ))

        # When: We run a dream cycle
        session = dreamer.dream(kb)

        # Then: Exploration paths should equal number of insights
        assert session.exploration_paths == len(session.insights)
        assert session.exploration_paths > 0

    def test_compression_ratio_calculation(self):
        """Test that compression ratio is calculated correctly."""
        # Given: A knowledge base with redundant patterns
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(mother ?X ?Y)")]
        ))

        # When: We run a dream cycle
        session = dreamer.dream(kb, focus="compression")

        # Then: Compression ratio should be between 0 and 1
        assert 0.0 <= session.compression_ratio <= 1.0
        # With 2 rules deriving same head, compression ratio should be 0.5 (1 - 1/2)
        assert session.compression_ratio == 0.5


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_family_relations_pattern_detection(self):
        """Test pattern detection in a realistic family relations KB."""
        # Given: A KB with family relationship rules
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        # Multiple rules with same head (parent relationship)
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(mother ?X ?Y)")]
        ))

        # Multiple rules with same head (child relationship)
        kb.add_rule(Rule(
            parse_s_expression("(child ?X ?Y)"),
            [parse_s_expression("(son ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(child ?X ?Y)"),
            [parse_s_expression("(daughter ?X ?Y)")]
        ))

        # When: We run dream analysis
        session = dreamer.dream(kb, focus="compression", verify=True)

        # Then: Should detect multiple compression patterns
        assert len(session.insights) >= 2, "Should detect patterns for both parent and child"
        compression_insights = [i for i in session.insights if i.type == "compression"]
        assert len(compression_insights) >= 2, "Should find at least 2 compression opportunities"

        # Should have meaningful descriptions
        for insight in compression_insights:
            assert "rules" in insight.description.lower()
            assert insight.compression_ratio > 0.0
            assert insight.coverage_gain > 0.0

    def test_session_matches_demo_api(self):
        """Test that the session object matches the wake_sleep_demo.py API expectations."""
        # Given: A knowledge base with patterns
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(father ?X ?Y)")]
        ))
        kb.add_rule(Rule(
            parse_s_expression("(parent ?X ?Y)"),
            [parse_s_expression("(mother ?X ?Y)")]
        ))

        # When: We run a dream cycle like the demo does
        session = dreamer.dream(
            kb,
            dream_cycles=2,
            exploration_samples=3,
            focus="all",
            verify=True
        )

        # Then: Session should have all properties the demo uses
        # These are accessed in the demo's sleep_phase function
        assert hasattr(session, 'exploration_paths')
        assert hasattr(session, 'insights')
        assert hasattr(session, 'compression_ratio')
        assert hasattr(session, 'generalization_score')
        assert hasattr(session, 'verification')

        # Verification should have these properties
        assert hasattr(session.verification, 'preserved')
        assert hasattr(session.verification, 'similarity_score')
        assert hasattr(session.verification, 'improvements')

        # Insights should have these properties
        if session.insights:
            insight = session.insights[0]
            assert hasattr(insight, 'type')
            assert hasattr(insight, 'description')
            assert hasattr(insight, 'compression_ratio')
            assert hasattr(insight, 'coverage_gain')
            assert hasattr(insight, 'verified')

    def test_multiple_pattern_types(self):
        """Test that different focus types can be detected separately."""
        # Given: A knowledge base
        from dreamlog.prefix_parser import parse_s_expression
        from dreamlog.knowledge import Rule
        provider = MockLLMProvider()
        dreamer = KnowledgeBaseDreamer(provider)
        kb = KnowledgeBase()

        kb.add_rule(Rule(
            parse_s_expression("(grandparent ?X ?Z)"),
            [parse_s_expression("(parent ?X ?Y)"), parse_s_expression("(parent ?Y ?Z)")]
        ))

        # When: We focus on different pattern types
        session_all = dreamer.dream(kb, focus="all")
        session_compression = dreamer.dream(kb, focus="compression")
        session_abstraction = dreamer.dream(kb, focus="abstraction")
        session_generalization = dreamer.dream(kb, focus="generalization")

        # Then: All sessions should return successfully
        assert session_all is not None
        assert session_compression is not None
        assert session_abstraction is not None
        assert session_generalization is not None

        # All should have valid structure
        for session in [session_all, session_compression, session_abstraction, session_generalization]:
            assert isinstance(session.insights, list)
            assert isinstance(session.compression_ratio, float)
            assert isinstance(session.generalization_score, float)
