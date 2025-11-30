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
from dreamlog.tfidf_embedding_provider import TfIdfEmbeddingProvider
from dreamlog.prompt_template_system import RULE_EXAMPLES


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
        # Add base facts for healthy test
        kb.fact("exercises", "alice")
        kb.fact("eats_well", "alice")

        return kb

    def test_undefined_predicate_triggers_generation(self, kb_with_llm):
        """Test that querying undefined predicate triggers LLM rule generation"""
        # Query for undefined "healthy" predicate
        # LLM should generate: (healthy X) :- (exercises X), (eats_well X)
        results = list(kb_with_llm.query("healthy", "alice"))

        # Should have generated rule and found results using existing facts
        assert len(results) > 0, "Query should succeed after LLM generates rule"

        # Verify the rule was added, not facts
        engine = kb_with_llm.engine
        healthy_rules = [r for r in engine.kb.rules if hasattr(r.head, 'functor') and r.head.functor == 'healthy']
        assert len(healthy_rules) > 0, "LLM should have generated a rule for healthy"
    
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

        # Create embedding provider for RAG
        embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

        hook = LLMHook(mock_llm_provider, embedding_provider)
        
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
        # Create embedding provider for RAG
        embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

        hook = LLMHook(mock_llm_provider, embedding_provider, cache_enabled=True)
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
        # Create embedding provider for RAG
        embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

        hook = LLMHook(mock_llm_provider, embedding_provider, max_generations=2)
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
    
    def test_facts_and_rules_from_llm(self, mock_llm_provider):
        """Test that both facts and rules from LLM are accepted"""
        kb = KnowledgeBase()
        evaluator = Mock()
        evaluator.kb = kb

        # Create embedding provider for RAG
        embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

        hook = LLMHook(mock_llm_provider, embedding_provider)

        # Add a custom response that contains both facts and rules
        import json
        def custom_call_api(prompt, **kwargs):
            # Return both facts and rules
            return json.dumps([
                ["fact", ["should_be", "added"]],
                ["fact", ["also_added", "yes"]],
                ["rule", ["test_rule", "X"], [["base_predicate", "X"]]]
            ])

        mock_llm_provider._call_api = custom_call_api

        # Trigger generation
        term = compound("test_rule", var("X"))
        hook(term, evaluator)

        # Facts SHOULD be added (LLM can generate both facts and rules)
        assert len(kb.facts) == 2, "Facts from LLM should be added"
        fact_strs = [str(f.term) for f in kb.facts]
        assert any("should_be" in f for f in fact_strs)
        assert any("also_added" in f for f in fact_strs)

        # Rules SHOULD also be added
        assert len(kb.rules) > 0, "Rules from LLM should be added"
        rule_heads = [str(r.head) for r in kb.rules]
        assert any("test_rule" in h for h in rule_heads)
    
    def test_rule_addition_from_llm(self, mock_llm_provider):
        """Test that rules from LLM are properly added"""
        kb = KnowledgeBase()
        evaluator = Mock()
        evaluator.kb = kb

        # Create embedding provider for RAG
        embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

        hook = LLMHook(mock_llm_provider, embedding_provider)
        
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
        """Test completing a family tree with LLM-generated rules"""
        import json

        # Override _call_api to return a sibling rule that works with existing parent facts
        original_call_api = mock_llm_provider._call_api

        def custom_call_api(prompt, **kwargs):
            lines = prompt.split('\n')
            query_line = next((line for line in lines if line.startswith('Query:')), '')

            if 'sibling' in query_line.lower():
                # Return a sibling rule that uses parent facts
                # Note: We don't use (different X Y) here for simplicity
                return json.dumps([
                    ["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"]]]
                ])
            return original_call_api(prompt, **kwargs)

        mock_llm_provider._call_api = custom_call_api

        kb = dreamlog(llm_provider=mock_llm_provider)

        # Add partial family tree (base facts)
        kb.fact("parent", "john", "alice")
        kb.fact("parent", "john", "bob")

        # Query for undefined sibling relationship
        results = list(kb.query("sibling", "bob", "alice"))

        # Should have generated rule and found relationship using existing facts
        assert len(results) > 0
    
    def test_cascading_generation(self, mock_llm_provider):
        """Test that generated rules can trigger more generation"""
        import json

        # Mock responses as JSON strings (what _call_api returns)
        responses = {
            "grandparent": json.dumps([["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]),
            "ancestor": json.dumps([
                ["rule", ["ancestor", "X", "Y"], [["parent", "X", "Y"]]],
                ["rule", ["ancestor", "X", "Z"], [["parent", "X", "Y"], ["ancestor", "Y", "Z"]]]
            ])
        }

        # Store original _call_api
        original_call_api = mock_llm_provider._call_api

        def custom_call_api(prompt, **kwargs):
            # Check prompt for which predicate is being queried
            # Look specifically in the Query line to avoid false matches
            lines = prompt.split('\n')
            query_line = next((line for line in lines if line.startswith('Query:')), '')

            for key, response in responses.items():
                if key in query_line.lower():
                    return response
            # Fall back to original for anything else
            return original_call_api(prompt, **kwargs)

        # Override _call_api (not generate_knowledge, since LLMHook uses complete())
        mock_llm_provider._call_api = custom_call_api
        
        kb = dreamlog(llm_provider=mock_llm_provider)
        kb.fact("parent", "john", "mary")
        kb.fact("parent", "mary", "alice")
        
        # Query for undefined grandparent - should generate rule
        results = list(kb.query("grandparent", "john", "alice"))
        assert len(results) > 0
        
        # Query for undefined ancestor - should generate and use
        results = list(kb.query("ancestor", "john", "alice"))
        assert len(results) > 0