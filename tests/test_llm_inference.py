"""
Tests for LLM-based knowledge generation when undefined predicates are encountered.

These tests verify that:
1. Undefined predicates trigger LLM generation
2. Generated knowledge is properly integrated
3. Context is correctly extracted and provided
4. Caching prevents redundant generation
"""

import pytest
from unittest.mock import Mock, patch
from dreamlog.pythonic import dreamlog
from dreamlog.knowledge import KnowledgeBase
from dreamlog.llm_hook import LLMHook
from dreamlog.llm_providers import LLMResponse
from .mock_provider import MockLLMProvider
from dreamlog import compound, atom, var
from dreamlog.evaluator import PrologEvaluator
from dreamlog.config import DreamLogConfig, LLMSamplingConfig


class TestLLMInference:
    """Test automatic knowledge generation for undefined predicates"""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider with predefined responses"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        # Add custom responses for specific test scenarios
        provider.add_response("healthy", 
            facts=[
                ["healthy", "alice"],
                ["exercises", "alice"], 
                ["eats_well", "alice"]
            ],
            rules=[
                [["healthy", "X"], [["exercises", "X"], ["eats_well", "X"]]]
            ]
        )
        
        provider.add_response("uncle",
            facts=[],
            rules=[
                [["uncle", "X", "Y"], [["brother", "X", "Z"], ["parent", "Z", "Y"]]],
                [["uncle", "X", "Y"], [["male", "X"], ["sibling", "X", "Z"], ["parent", "Z", "Y"]]]
            ]
        )
        
        return provider
    
    @pytest.fixture
    def kb_with_llm(self, mock_llm_provider):
        """Create a knowledge base with LLM support"""
        kb = dreamlog(llm_provider=mock_llm_provider)
        
        # Add some base facts
        kb.fact("parent", "john", "mary")
        kb.fact("parent", "mary", "alice")
        kb.fact("male", "john")
        kb.fact("male", "bob")
        kb.fact("brother", "john", "jane")
        
        return kb
    
    def test_undefined_predicate_triggers_generation(self, kb_with_llm):
        """Test that querying undefined predicate triggers LLM"""
        # Query for undefined "healthy" predicate
        results = list(kb_with_llm.query("healthy", "alice"))
        
        # Should have generated and found results
        assert len(results) > 0
        
        # Check that the generated facts were added
        assert kb_with_llm.ask("healthy", "alice")
        assert kb_with_llm.ask("exercises", "alice")
    
    def test_generated_rules_are_usable(self, kb_with_llm):
        """Test that generated rules can be used for inference"""
        # Query for uncle relationship (undefined)
        results = list(kb_with_llm.query("uncle", "john", "alice"))
        
        # Should have generated uncle rules
        # john is brother of jane, mary is parent of alice
        # So if jane were parent of alice, john would be uncle
        # But with current facts, might not find a match
        
        # Add fact to make the rule work
        kb_with_llm.fact("parent", "jane", "bob_child")
        
        # Now query should work
        results = list(kb_with_llm.query("uncle", "john", "bob_child"))
        assert len(results) > 0
    
    def test_context_extraction(self, mock_llm_provider):
        """Test that relevant context is extracted for LLM"""
        kb = KnowledgeBase()
        
        # Add various facts
        for i in range(30):  # Many facts to test sampling
            kb.add_fact(compound("fact", atom(f"item_{i}"), atom("value")))
        
        kb.add_fact(compound("parent", atom("john"), atom("mary")))
        kb.add_fact(compound("parent", atom("mary"), atom("alice")))
        
        hook = LLMHook(mock_llm_provider)
        
        # Test context extraction for a compound term
        term = compound("parent", var("X"), var("Y"))
        context = hook._extract_context(term, kb)
        
        assert "parent" in context
        assert "FACTS" in context or "RELATED FACTS" in context
        assert "KNOWLEDGE BASE STATS" in context
    
    def test_context_sampling_strategies(self):
        """Test different context sampling strategies"""
        config = DreamLogConfig()
        
        # Test related strategy
        config.sampling.strategy = "related"
        config.sampling.max_facts = 10
        assert config.sampling.strategy == "related"
        
        # Test random strategy
        config.sampling.strategy = "random"
        assert config.sampling.strategy == "random"
        
        # Test recent strategy
        config.sampling.strategy = "recent"
        assert config.sampling.strategy == "recent"
    
    def test_caching_prevents_redundant_generation(self, mock_llm_provider):
        """Test that LLM hook caches generated knowledge"""
        hook = LLMHook(mock_llm_provider, cache_enabled=True)
        kb = KnowledgeBase()
        
        # Create a mock evaluator
        evaluator = Mock()
        evaluator.kb = kb
        
        # First call should generate
        term = compound("healthy", atom("alice"))
        hook(term, evaluator)
        
        # Track calls to provider
        call_count_before = mock_llm_provider.call_count
        
        # Second call should use cache
        hook(term, evaluator)
        
        call_count_after = mock_llm_provider.call_count
        
        # Should not have called provider again  
        assert call_count_after == call_count_before
    
    def test_generation_limit(self, mock_llm_provider):
        """Test that generation has a limit to prevent infinite loops"""
        hook = LLMHook(mock_llm_provider, max_generations=2)
        kb = KnowledgeBase()
        evaluator = Mock()
        evaluator.kb = kb
        
        # Generate for two different terms
        hook(compound("pred1", atom("a")), evaluator)
        hook(compound("pred2", atom("b")), evaluator)
        
        # Third generation should be blocked
        hook(compound("pred3", atom("c")), evaluator)
        
        # Check generation count
        assert hook._generation_count == 2
    
    def test_fact_addition_from_llm(self, mock_llm_provider):
        """Test that facts from LLM are properly added"""
        kb = KnowledgeBase()
        evaluator = Mock()
        evaluator.kb = kb
        
        hook = LLMHook(mock_llm_provider)
        
        # Trigger generation for "healthy"
        term = compound("healthy", atom("alice"))
        hook(term, evaluator)
        
        # Check facts were added
        assert len(kb.facts) > 0
        
        # Check specific facts
        fact_strs = [str(f.term) for f in kb.facts]
        assert any("healthy" in s for s in fact_strs)
        assert any("exercises" in s for s in fact_strs)
    
    def test_rule_addition_from_llm(self, mock_llm_provider):
        """Test that rules from LLM are properly added"""
        kb = KnowledgeBase()
        evaluator = Mock()
        evaluator.kb = kb
        
        hook = LLMHook(mock_llm_provider)
        
        # Trigger generation for "uncle"
        term = compound("uncle", var("X"), var("Y"))
        hook(term, evaluator)
        
        # Check rules were added
        assert len(kb.rules) > 0
        
        # Check that uncle rules exist
        rule_heads = [str(r.head) for r in kb.rules]
        assert any("uncle" in h for h in rule_heads)


class TestLLMProviderIntegration:
    """Test integration with different LLM providers"""
    
    def test_mock_provider(self):
        """Test mock provider for deterministic testing"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        response = provider.generate_knowledge("parent", context="test context")
        
        assert isinstance(response, LLMResponse)
        assert len(response.facts) > 0
        # Check we got family domain data
        assert any("parent" in str(fact) or "grandparent" in str(fact) for fact in response.facts + response.rules)
    
    @pytest.mark.skip(reason="Requires API key")
    def test_openai_provider(self):
        """Test OpenAI provider (requires API key)"""
        from dreamlog.llm_providers import OpenAIProvider
        
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # This would make a real API call
        # response = provider.generate_knowledge("test", context="test")
    
    def test_provider_configuration(self):
        """Test provider configuration options"""
        from dreamlog.config import DreamLogConfig
        from dreamlog.llm_providers import create_provider
        
        # Test config-based provider creation
        config = DreamLogConfig()
        config.provider.provider = "mock"
        config.provider.temperature = 0.5
        config.provider.max_tokens = 100
        
        provider = create_provider(
            config.provider.provider,
            temperature=config.provider.temperature,
            max_tokens=config.provider.max_tokens
        )
        assert provider is not None


class TestPromptTemplates:
    """Test prompt template system"""
    
    def test_default_template(self):
        """Test default prompt template"""
        from dreamlog.llm_prompt_templates import PromptTemplateManager
        
        manager = PromptTemplateManager()
        
        prompt = manager.create_prompt(
            term="(healthy alice)",
            functor="healthy",
            arity=1,
            knowledge_base="(parent john mary)\n(age alice 25)"
        )
        
        assert "(healthy alice)" in prompt
        assert "healthy" in prompt
        assert "(parent john mary)" in prompt
    
    def test_custom_template(self):
        """Test custom prompt template"""
        from dreamlog.llm_prompt_templates import create_custom_template
        
        template = create_custom_template("""
        Generate facts for ${term}.
        Context: ${knowledge_base}
        """)
        
        prompt = template.create_prompt(
            term="test_term",
            knowledge_base="test_kb"
        )
        
        assert "test_term" in prompt
        assert "test_kb" in prompt
    
    def test_template_from_file(self):
        """Test loading template from file"""
        from dreamlog.llm_prompt_templates import PromptTemplateManager
        import tempfile
        
        # Create a temporary template file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test template for ${term}")
            temp_path = f.name
        
        try:
            manager = PromptTemplateManager(template_source=temp_path)
            prompt = manager.create_prompt(term="test")
            assert "Test template for test" == prompt
        finally:
            import os
            os.unlink(temp_path)


@pytest.mark.integration  
class TestEndToEndLLMIntegration:
    """End-to-end tests for LLM integration"""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider for integration tests"""
        return MockLLMProvider(knowledge_domain="family")
    
    def test_family_tree_completion(self, mock_llm_provider):
        """Test completing a family tree with LLM"""
        # Provide custom responses for family relationships
        mock_llm_provider.generate_knowledge = Mock(return_value=LLMResponse(
            text="Family rules",
            facts=[["brother", "bob", "alice"]],
            rules=[[["sibling", "X", "Y"], [["brother", "X", "Y"]]]]
        ))
        
        kb = dreamlog(llm_provider=mock_llm_provider)
        
        # Add partial family tree
        kb.fact("parent", "john", "alice")
        kb.fact("parent", "john", "bob")
        
        # Query for undefined sibling relationship
        results = list(kb.query("sibling", "bob", "alice"))
        
        # Should have generated and found relationship
        assert len(results) > 0
    
    def test_cascading_generation(self, mock_llm_provider):
        """Test that generated rules can trigger more generation"""
        responses = {
            "grandparent": LLMResponse(
                text="",
                facts=[],
                rules=[[["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
            ),
            "ancestor": LLMResponse(
                text="",
                facts=[],
                rules=[
                    [["ancestor", "X", "Y"], [["parent", "X", "Y"]]],
                    [["ancestor", "X", "Z"], [["parent", "X", "Y"], ["ancestor", "Y", "Z"]]]
                ]
            )
        }
        
        def generate(term, context):
            for key in responses:
                if key in str(term):
                    return responses[key]
            return LLMResponse("", [], [])
        
        mock_llm_provider.generate_knowledge = generate
        
        kb = dreamlog(llm_provider=mock_llm_provider)
        kb.fact("parent", "john", "mary")
        kb.fact("parent", "mary", "alice")
        
        # Query for undefined grandparent - should generate rule
        results = list(kb.query("grandparent", "john", "alice"))
        assert len(results) > 0
        
        # Query for undefined ancestor - should generate and use
        results = list(kb.query("ancestor", "john", "alice"))
        assert len(results) > 0