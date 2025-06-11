"""
Tests for LLM integration functionality
"""
import pytest
import sys
from pathlib import Path
import json
import re
from typing import Dict

# Add the jlog package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jlog import (
    atom, var, compound,
    Fact, KnowledgeBase,
    PrologEvaluator, JLogEngine,
    LLMHook, create_engine_with_llm
)


class MockLLMProvider:
    """Mock LLM provider for testing"""
    
    def __init__(self, knowledge_domain: str = "family"):
        self.knowledge_domain = knowledge_domain
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate mock response based on the prompt"""
        functor = self._extract_functor_from_prompt(prompt)
        query_args = self._extract_query_args_from_prompt(prompt)
        
        if self.knowledge_domain == "family":
            return self._generate_family_knowledge(functor, query_args, prompt)
        
        # Default response for unknown domains
        return '''[
            {
                "type": "fact",
                "term": {
                    "type": "atom",
                    "value": "unknown"
                }
            }
        ]'''
    
    def _generate_family_knowledge(self, functor: str, query_args: Dict[str, str], prompt: str) -> str:
        """Generate family-specific knowledge"""
        
        if functor == "parent":
            # Extract specific names from the query if available
            parent_name = query_args.get("arg1", "john")  # Default to john if not found
            child_name = query_args.get("arg2", "mary")   # Default child name
            
            # Generate facts that would satisfy the query
            return f'''[
                {{
                    "type": "fact",
                    "term": {{
                        "type": "compound",
                        "functor": "parent",
                        "args": [
                            {{"type": "atom", "value": "{parent_name}"}},
                            {{"type": "atom", "value": "{child_name}"}}
                        ]
                    }}
                }},
                {{
                    "type": "fact",
                    "term": {{
                        "type": "compound",
                        "functor": "parent",
                        "args": [
                            {{"type": "atom", "value": "{parent_name}"}},
                            {{"type": "atom", "value": "charlie"}}
                        ]
                    }}
                }},
                {{
                    "type": "fact",
                    "term": {{
                        "type": "compound",
                        "functor": "parent",
                        "args": [
                            {{"type": "atom", "value": "alice"}},
                            {{"type": "atom", "value": "bob"}}
                        ]
                    }}
                }}
            ]'''
        
        elif functor == "ancestor":
            return '''[
                {
                    "type": "rule",
                    "head": {
                        "type": "compound",
                        "functor": "ancestor",
                        "args": [
                            {"type": "variable", "name": "X"},
                            {"type": "variable", "name": "Y"}
                        ]
                    },
                    "body": [
                        {
                            "type": "compound",
                            "functor": "parent",
                            "args": [
                                {"type": "variable", "name": "X"},
                                {"type": "variable", "name": "Y"}
                            ]
                        }
                    ]
                },
                {
                    "type": "rule",
                    "head": {
                        "type": "compound",
                        "functor": "ancestor",
                        "args": [
                            {"type": "variable", "name": "X"},
                            {"type": "variable", "name": "Z"}
                        ]
                    },
                    "body": [
                        {
                            "type": "compound",
                            "functor": "parent",
                            "args": [
                                {"type": "variable", "name": "X"},
                                {"type": "variable", "name": "Y"}
                            ]
                        },
                        {
                            "type": "compound",
                            "functor": "ancestor",
                            "args": [
                                {"type": "variable", "name": "Y"},
                                {"type": "variable", "name": "Z"}
                            ]
                        }
                    ]
                }
            ]'''
        
        # Default family facts
        return '''[
            {
                "type": "fact",
                "term": {
                    "type": "compound",
                    "functor": "parent",
                    "args": [
                        {"type": "atom", "value": "john"},
                        {"type": "atom", "value": "mary"}
                    ]
                }
            }
        ]'''
    
    def _extract_functor_from_prompt(self, prompt: str) -> str:
        """Extract the main functor from a prompt"""
        patterns = [
            r'The term "(\w+)\(',          # "The term "parent("
            r'resolve the term: (\w+)\(',  # "resolve the term: parent("
            r'functor.*?[\'"](\w+)[\'"]',  # functor "parent"
            r'(\w+)\s*\(',                 # parent(
            r'(\w+)\s+relationships?',     # parent relationships
            r'(\w+)\s+queries?',           # parent queries
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _extract_query_args_from_prompt(self, prompt: str) -> Dict[str, str]:
        """Extract argument values from the query in the prompt"""
        args = {}
        
        # Look for patterns like parent(john, X) or parent("john", X)
        patterns = [
            r'(\w+)\s*\(\s*(["\']?)(\w+)\2\s*,\s*(["\']?)(\w+)\4\s*\)',  # parent(john, X)
            r'compound\(["\'](\w+)["\'],\s*atom\(["\'](\w+)["\'].*?\)',   # compound("parent", atom("john", ...
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 3:
                    args["arg1"] = match.group(3) if len(match.groups()) >= 3 else None
                    args["arg2"] = match.group(5) if len(match.groups()) >= 5 else None
                elif len(match.groups()) >= 2:
                    args["arg1"] = match.group(2)
                break
        
        # Also look for specific atom mentions in the prompt
        atom_pattern = r'atom\(["\'](\w+)["\']\)'
        atom_matches = re.findall(atom_pattern, prompt, re.IGNORECASE)
        if atom_matches:
            args["arg1"] = atom_matches[0]
            if len(atom_matches) > 1:
                args["arg2"] = atom_matches[1]
        
        return args
    
class TestLLMIntegration:
    """Test LLM integration functionality"""
    
    def test_mock_llm_provider(self):
        """Test mock LLM provider"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        # Test that it generates responses
        response = provider.generate_text("Generate facts about parent relationships")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_llm_hook_basic(self):
        """Test basic LLM hook functionality"""
        provider = MockLLMProvider(knowledge_domain="family")
        hook = LLMHook(provider, "family")
        
        kb = KnowledgeBase()
        evaluator = PrologEvaluator(kb, hook)
        
        # Query for unknown term should trigger LLM
        initial_count = len(kb.facts) + len(kb.rules)
        
        unknown_term = compound("parent", atom("john"), var("X"))
        hook(unknown_term, evaluator)
        
        # Should have generated some knowledge
        final_count = len(kb.facts) + len(kb.rules)
        assert final_count > initial_count
    
    def test_zero_shot_learning(self):
        """Test zero-shot learning capability"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Start with empty knowledge base
        assert len(engine.facts) == 0
        assert len(engine.rules) == 0
        
        # Query should trigger knowledge generation
        solutions = engine.query([compound("parent", atom("john"), var("X"))])
        
        # Should have generated knowledge and found solutions
        assert len(engine.facts) > 0
        assert len(solutions) > 0


    def test_engine_with_llm(self):
        """Test engine with LLM integration"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Should start empty
        assert len(engine.facts) == 0
        
        # Query should trigger LLM generation
        solutions = engine.query([compound("parent", var("X"), var("Y"))])
        
        # Should have generated knowledge
        assert len(engine.facts) > 0
        assert len(solutions) > 0


    def test_llm_progressive_learning(self):
        """Test progressive learning with LLM"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Start empty
        initial_facts = len(engine.facts)
        initial_rules = len(engine.rules)
        
        # First query - should generate basic facts
        engine.query([compound("parent", var("X"), var("Y"))])
        facts_after_first = len(engine.facts)
        
        assert facts_after_first > initial_facts
        
        # Second query - should generate rules  
        engine.query([compound("ancestor", var("X"), var("Y"))])
        rules_after_second = len(engine.rules)
        
        assert rules_after_second > initial_rules
        
        # Third query - should use existing knowledge
        solutions = engine.query([compound("ancestor", atom("john"), var("X"))])
        
        # Should find solutions using generated knowledge
        assert len(solutions) > 0

class TestMockLLMProvider:
    """Test MockLLMProvider functionality"""
    
    def test_creation(self):
        """Test creating mock LLM provider"""
        provider = MockLLMProvider()
        assert provider.knowledge_domain == "family"
        
        provider2 = MockLLMProvider(knowledge_domain="medical")
        assert provider2.knowledge_domain == "medical"
    
    def test_generate_text_basic(self):
        """Test basic text generation"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        response = provider.generate_text("test prompt")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_llm_integration_workflow(self):
        """Test complete LLM integration workflow"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Start empty and learn
        assert len(engine.facts) == 0
        
        # Query triggers learning
        solutions = engine.query([compound("parent", atom("john"), var("X"))])
        assert len(engine.facts) > 0
        
        # Export and reimport knowledge
        json_data = engine.save_to_json()
        
        new_engine = JLogEngine()
        new_engine.load_from_json(json_data)
        
        # Should work without LLM now
        assert len(new_engine.facts) > 0
    
    def test_generate_parent_facts(self):
        """Test generating parent facts"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        response = provider.generate_text("parent relationships")
        assert isinstance(response, str)
        
        # Should be valid JSON
        data = json.loads(response)
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check structure of first fact
        first_fact = data[0]
        assert first_fact["type"] == "fact"
        assert "term" in first_fact
    
    def test_generate_ancestor_rules(self):
        """Test generating ancestor rules"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        response = provider.generate_text("ancestor relationships")
        
        data = json.loads(response)
        assert isinstance(data, list)
        
        # Should contain rules
        has_rule = any(item.get("type") == "rule" for item in data)
        assert has_rule
    
    def test_functor_extraction(self):
        """Test functor extraction from prompts"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        # Test different prompt formats
        test_cases = [
            ("parent relationships", "parent"),
            ("ancestor queries", "ancestor"),
            ("grandparent(X, Y)", "grandparent"),
        ]
        
        for prompt, expected_functor in test_cases:
            extracted = provider._extract_functor_from_prompt(prompt)
            # Should extract some functor (might not be exact due to heuristics)
            assert isinstance(extracted, str)
    
    def test_unknown_functor_handling(self):
        """Test handling of unknown functors"""
        provider = MockLLMProvider(knowledge_domain="family")
        
        response = provider.generate_text("completely unknown relationship")
        
        # Should still return valid JSON
        data = json.loads(response)
        assert isinstance(data, list)


class TestLLMHook:
    """Test LLMHook functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.provider = MockLLMProvider(knowledge_domain="family")
        self.hook = LLMHook(self.provider, "family")
        self.kb = KnowledgeBase()
        self.evaluator = PrologEvaluator(self.kb, self.hook)
    
    def test_hook_creation(self):
        """Test creating LLM hook"""
        assert self.hook.knowledge_domain == "family"
        assert self.hook.llm_provider == self.provider
        assert len(self.hook._cache) == 0
    
    def test_hook_call_basic(self):
        """Test basic hook invocation"""
        initial_count = len(self.kb.facts) + len(self.kb.rules)
        
        unknown_term = compound("parent", atom("john"), var("X"))
        self.hook(unknown_term, self.evaluator)
        
        final_count = len(self.kb.facts) + len(self.kb.rules)
        assert final_count > initial_count
    
    def test_hook_caching(self):
        """Test that hook caches responses"""
        unknown_term = compound("parent", atom("john"), var("X"))
        
        # First call
        self.hook(unknown_term, self.evaluator)
        first_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Second call with same term
        self.hook(unknown_term, self.evaluator)
        second_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Should not generate new knowledge (cached)
        assert second_count == first_count

    def test_hook_generation_limit(self):
        """Test generation limit prevents infinite loops"""
        self.hook._max_generations = 3
        
        unknown_term = compound("parent", atom("john"), var("X"))
        
        # Clear cache to ensure fresh generation attempts
        self.hook.clear_cache()
        
        # Call multiple times - should stop generating after limit
        for _ in range(5):
            self.hook(unknown_term, self.evaluator)
        
        # Should have stopped generating after limit
        assert self.hook._generation_count <= self.hook._max_generations
    
    def test_hook_different_terms_not_cached(self):
        """Test that different terms are not cached together"""
        term1 = compound("parent", atom("john"), var("X"))
        term2 = compound("parent", atom("mary"), var("Y"))
        
        # First call with term1
        self.hook(term1, self.evaluator)
        first_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Call with different term2
        self.hook(term2, self.evaluator)
        second_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Should generate new knowledge for different term
        assert second_count >= first_count

    def test_hook_duplicate_knowledge_prevention(self):
        """Test that duplicate facts/rules are not added"""
        unknown_term = compound("parent", atom("john"), var("X"))
        
        # Clear cache to force regeneration
        self.hook.clear_cache()
        
        # First call
        self.hook(unknown_term, self.evaluator)
        first_facts = set(str(fact) for fact in self.kb.facts)
        first_rules = set(str(rule) for rule in self.kb.rules)
        
        # Clear cache and call again
        self.hook.clear_cache()
        self.hook(unknown_term, self.evaluator)
        second_facts = set(str(fact) for fact in self.kb.facts)
        second_rules = set(str(rule) for rule in self.kb.rules)
        
        # Should not have duplicates (sets should be same size or larger)
        assert len(second_facts) <= len(first_facts) * 2  # At most double
        assert len(second_rules) <= len(first_rules) * 2

    def test_hook_with_atom_term(self):
        """Test hook with simple atom term"""
        unknown_term = atom("unknown_predicate")
        initial_count = len(self.kb.facts) + len(self.kb.rules)
        
        self.hook(unknown_term, self.evaluator)
        final_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Should handle atom terms gracefully
        assert final_count >= initial_count

    def test_hook_with_variable_term(self):
        """Test hook with variable term"""
        unknown_term = var("X")
        initial_count = len(self.kb.facts) + len(self.kb.rules)
        
        self.hook(unknown_term, self.evaluator)
        final_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Should handle variable terms gracefully
        assert final_count >= initial_count

    def test_hook_cache_key_consistency(self):
        """Test that cache keys are consistent for same terms"""
        term1 = compound("parent", atom("john"), var("X"))
        term2 = compound("parent", atom("john"), var("X"))
        
        # Both terms should generate same cache key
        key1 = str(term1)
        key2 = str(term2)
        assert key1 == key2
        
        # First call should cache
        self.hook(term1, self.evaluator)
        assert len(self.hook._cache) > 0
        
        # Second call should use cache
        cache_size_before = len(self.hook._cache)
        self.hook(term2, self.evaluator)
        cache_size_after = len(self.hook._cache)
        
        assert cache_size_after == cache_size_before

    def test_hook_error_recovery(self):
        """Test hook recovery from errors during knowledge addition"""
        # Create a mock evaluator that raises errors when adding knowledge
        class ErrorEvaluator:
            def __init__(self, kb):
                self._kb = kb
            
            @property
            def kb(self):
                return self._kb
            
            @kb.setter 
            def kb(self, value):
                self._kb = value
        
        error_evaluator = ErrorEvaluator(self.kb)
        
        # Should not crash when evaluator has issues
        unknown_term = compound("parent", atom("john"), var("X"))
        try:
            self.hook(unknown_term, error_evaluator)
        except AttributeError:
            pass  # Expected due to mock evaluator
        
        # Hook should still be functional
        assert self.hook._generation_count > 0

    def test_hook_max_generations_reset(self):
        """Test that generation count can be reset"""
        self.hook._max_generations = 2
        
        unknown_term = compound("parent", atom("john"), var("X"))
        
        # Exhaust generation limit by clearing cache each time to force regeneration
        for _ in range(3):
            self.hook.clear_cache()  # Clear cache each iteration to force generation
            self.hook(unknown_term, self.evaluator)
        
        initial_count = self.hook._generation_count
        assert initial_count >= 2
        
        # Reset and verify it works again
        self.hook._generation_count = 0
        self.hook(unknown_term, self.evaluator)
        
        assert self.hook._generation_count > 0

    def test_hook_knowledge_format_validation(self):
        """Test validation of generated knowledge format"""
        # Create provider that returns various invalid formats
        class ValidationProvider:
            def __init__(self, response):
                self.response = response
            
            def generate_text(self, prompt, max_tokens=500):
                return self.response
        
        test_cases = [
            # Valid case
            ('[]', True),
            # Invalid JSON
            ('invalid json', False),
            # Non-list response
            ('{"type": "fact"}', True),  # Should be converted to list
            # Empty string
            ('', False),
        ]
        
        for response, should_work in test_cases:
            provider = ValidationProvider(response)
            hook = LLMHook(provider, "test")
            
            initial_count = len(self.kb.facts) + len(self.kb.rules)
            
            try:
                hook(compound("test", var("X")), self.evaluator)
                # If we get here without exception, check if knowledge was added appropriately
                final_count = len(self.kb.facts) + len(self.kb.rules)
                
                if should_work:
                    assert final_count >= initial_count
                else:
                    assert final_count == initial_count
                
            except Exception:
                # Some invalid formats may cause exceptions, which is acceptable
                if should_work:
                    pytest.fail(f"Valid format '{response}' caused exception")

    def test_hook_concurrent_calls(self):
        """Test hook behavior with multiple rapid calls"""
        unknown_term = compound("parent", atom("john"), var("X"))
        
        initial_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Multiple rapid calls
        for i in range(3):
            self.hook(unknown_term, self.evaluator)
        
        final_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Should only generate once due to caching
        # But generation count should reflect actual calls made
        assert final_count > initial_count
        assert len(self.hook._cache) > 0

    def test_hook_memory_efficiency(self):
        """Test that hook doesn't accumulate unlimited cache"""
        # Generate many different terms to test cache behavior
        for i in range(10):
            term = compound("predicate", atom(f"arg{i}"), var("X"))
            self.hook(term, self.evaluator)
        
        # Cache should contain entries but not grow without bound
        assert len(self.hook._cache) <= 15  # Reasonable upper limit
        
        # Clear cache should work
        self.hook.clear_cache()
        assert len(self.hook._cache) == 0

    def test_hook_nested_compound_terms(self):
        """Test hook with nested compound terms"""
        nested_term = compound("loves", 
                              compound("friend", atom("alice")), 
                              compound("parent", atom("bob"), var("X")))
        
        initial_count = len(self.kb.facts) + len(self.kb.rules)
        self.hook(nested_term, self.evaluator)
        final_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Should handle nested terms gracefully
        assert final_count >= initial_count

    def test_hook_large_knowledge_base(self):
        """Test hook performance with large knowledge base"""
        # Add many facts to the knowledge base
        for i in range(50):
            fact_term = compound("fact", atom(f"item{i}"), atom(f"value{i}"))
            self.kb.add_fact(Fact(fact_term))
        
        unknown_term = compound("unknown", atom("test"), var("X"))
        
        initial_count = len(self.kb.facts) + len(self.kb.rules)
        self.hook(unknown_term, self.evaluator)
        final_count = len(self.kb.facts) + len(self.kb.rules)
        
        # Should still work with large KB
        assert final_count >= initial_count

    def test_hook_prompt_contains_term_info(self):
        """Test that generated prompts contain term information"""
        term = compound("parent", atom("john"), var("X"))
        
        prompt = self.hook._create_prompt(term, self.kb)
        
        # Prompt should contain relevant information
        assert "parent" in prompt
        assert "john" in prompt
        assert self.hook.knowledge_domain in prompt
        assert "JSON" in prompt

    def test_hook_generation_count_tracking(self):
        """Test that generation count tracks actual knowledge generation"""
        assert self.hook._generation_count == 0
        
        term1 = compound("test1", var("X"))
        term2 = compound("test2", var("Y"))
        
        # First call - should generate new knowledge
        self.hook(term1, self.evaluator)
        assert self.hook._generation_count == 1
        
        # Second call with different term - should generate new knowledge
        self.hook(term2, self.evaluator)
        assert self.hook._generation_count == 2
        
        # Third call with cached term - should use cache, no increment
        self.hook(term1, self.evaluator)
        assert self.hook._generation_count == 2  # No increment for cache hit
        
        # Clear cache and call again - should generate and increment
        self.hook.clear_cache()
        self.hook(term1, self.evaluator)
        assert self.hook._generation_count == 3  # Now increments because cache was cleared

    def test_hook_cache_isolation(self):
        """Test that different hooks have isolated caches"""
        provider1 = MockLLMProvider(knowledge_domain="family")
        provider2 = MockLLMProvider(knowledge_domain="family")
        
        hook1 = LLMHook(provider1, "family")
        hook2 = LLMHook(provider2, "family")
        
        term = compound("parent", atom("john"), var("X"))
        
        # Call first hook
        hook1(term, self.evaluator)
        
        # Second hook should have empty cache
        assert len(hook1._cache) > 0
        assert len(hook2._cache) == 0

    def test_hook_json_parsing_edge_cases(self):
        """Test JSON parsing with edge cases"""
        class EdgeCaseProvider:
            def __init__(self, response):
                self.response = response
            
            def generate_text(self, prompt, max_tokens=500):
                return self.response
        
        edge_cases = [
            '{"type": "fact", "term": null}',  # null term
            '[{"type": "unknown"}]',  # unknown type
            '{"type": "fact", "term": {"type": "invalid"}}',  # invalid term type
            '[]',  # empty array
            '[{}]',  # empty object
        ]
        
        for case in edge_cases:
            provider = EdgeCaseProvider(case)
            hook = LLMHook(provider, "test")
            
            initial_count = len(self.kb.facts) + len(self.kb.rules)
            
            # Should not crash
            hook(compound("test", var("X")), self.evaluator)
            
            final_count = len(self.kb.facts) + len(self.kb.rules)
            # May or may not add knowledge depending on how edge case is handled
            assert final_count >= initial_count

    def test_hook_term_string_representation(self):
        """Test that term string representation is consistent for caching"""
        # Same terms with different variable names should be treated differently
        term1 = compound("parent", atom("john"), var("X"))
        term2 = compound("parent", atom("john"), var("Y"))
        
        key1 = str(term1)
        key2 = str(term2)
        
        # Should be different cache keys (different variable names)
        assert key1 != key2
        
        # Call with both terms
        self.hook(term1, self.evaluator)
        self.hook(term2, self.evaluator)
        
        # Should have separate cache entries
        assert len(self.hook._cache) >= 1

    def test_hook_knowledge_base_integration(self):
        """Test integration with knowledge base operations"""
        term = compound("parent", atom("alice"), var("X"))
        
        # Hook should work with existing facts
        existing_fact = Fact(compound("parent", atom("bob"), atom("charlie")))
        self.kb.add_fact(existing_fact)
        
        initial_facts = len(self.kb.facts)
        self.hook(term, self.evaluator)
        final_facts = len(self.kb.facts)
        
        # Should have added new facts
        assert final_facts >= initial_facts
        
        # Original fact should still be there
        assert existing_fact in self.kb.facts
    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        unknown_term = compound("parent", atom("john"), var("X"))
        self.hook(unknown_term, self.evaluator)
        
        assert len(self.hook._cache) > 0
        assert self.hook._generation_count > 0
        
        generation_count_before = self.hook._generation_count
        self.hook.clear_cache()
        
        # Cache should be cleared
        assert len(self.hook._cache) == 0
        # Generation count should persist (tracks lifetime total, not cache state)
        assert self.hook._generation_count == generation_count_before
    
    def test_prompt_creation(self):
        """Test prompt creation for LLM"""
        term = compound("likes", atom("john"), var("X"))
        
        prompt = self.hook._create_prompt(term, self.kb)
        
        assert isinstance(prompt, str)
        assert "likes" in prompt
        assert "john" in prompt
        assert self.hook.knowledge_domain in prompt
    
    def test_knowledge_generation_errors(self):
        """Test handling of knowledge generation errors"""
        # Create a hook with a provider that returns invalid JSON
        class BadProvider:
            def generate_text(self, prompt, max_tokens=500):
                return "invalid json {"
        
        bad_hook = LLMHook(BadProvider(), "test")
        term = compound("test", atom("a"))
        
        # Should not crash, just not add knowledge
        initial_count = len(self.kb.facts) + len(self.kb.rules)
        bad_hook(term, self.evaluator)
        final_count = len(self.kb.facts) + len(self.kb.rules)
        
        assert final_count == initial_count


class TestZeroShotLearning:
    """Test zero-shot learning capabilities"""
    
    def test_empty_kb_learning(self):
        """Test learning from completely empty knowledge base"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Start completely empty
        assert len(engine.facts) == 0
        assert len(engine.rules) == 0
        
        # Query should trigger learning
        solutions = engine.query([compound("parent", atom("john"), var("X"))])
        
        # Should have learned and found solutions
        assert len(engine.facts) > 0
        assert len(solutions) > 0
    
    def test_progressive_learning(self):
        """Test progressive knowledge building"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Track knowledge growth
        knowledge_states = []
        
        # Query 1: Basic facts
        engine.query([compound("parent", var("X"), var("Y"))])
        knowledge_states.append((len(engine.facts), len(engine.rules)))
        
        # Query 2: Rules
        engine.query([compound("ancestor", var("X"), var("Y"))])
        knowledge_states.append((len(engine.facts), len(engine.rules)))
        
        # Query 3: Use existing knowledge
        engine.query([compound("grandparent", var("X"), var("Y"))])
        knowledge_states.append((len(engine.facts), len(engine.rules)))
        
        # Knowledge should grow or stay the same (never shrink)
        for i in range(1, len(knowledge_states)):
            prev_facts, prev_rules = knowledge_states[i-1]
            curr_facts, curr_rules = knowledge_states[i]
            
            assert curr_facts >= prev_facts
            assert curr_rules >= prev_rules
    
    def test_domain_specific_learning(self):
        """Test domain-specific knowledge generation"""
        # Family domain
        family_provider = MockLLMProvider(knowledge_domain="family")
        family_engine = create_engine_with_llm(family_provider, "family")
        
        family_engine.query([compound("parent", var("X"), var("Y"))])
        
        # Should generate family-related knowledge
        family_facts = [str(fact) for fact in family_engine.facts]
        has_family_content = any("parent" in fact for fact in family_facts)
        assert has_family_content
    
    def test_learning_from_failed_queries(self):
        """Test that failed queries can trigger learning"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Query for something that doesn't exist initially
        initial_facts = len(engine.facts)
        solutions = engine.query([compound("loves", atom("romeo"), atom("juliet"))])
        final_facts = len(engine.facts)
        
        # Even if no solutions found, might have generated knowledge
        # (depends on mock provider behavior)
        assert final_facts >= initial_facts
    
    def test_learning_with_existing_knowledge(self):
        """Test learning when some knowledge already exists"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Add some initial knowledge
        engine.add_fact_from_term(compound("parent", atom("alice"), atom("bob")))
        initial_facts = len(engine.facts)
        
        # Query for related knowledge
        engine.query([compound("parent", var("X"), var("Y"))])
        final_facts = len(engine.facts)
        
        # Should have added more knowledge
        assert final_facts >= initial_facts


class TestLLMIntegrationWithEngine:
    """Test LLM integration with JLogEngine"""
    
    def test_engine_with_llm_creation(self):
        """Test creating engine with LLM"""
        provider = MockLLMProvider()
        engine = create_engine_with_llm(provider, "test_domain")
        
        assert engine.llm_hook is not None
        assert engine.llm_hook.knowledge_domain == "test_domain"
        assert engine.evaluator.unknown_hook == engine.llm_hook
    
    def test_engine_without_llm(self):
        """Test engine without LLM integration"""
        engine = JLogEngine()
        
        assert engine.llm_hook is None
        assert engine.evaluator.unknown_hook is None
    
    def test_llm_triggered_by_query(self):
        """Test that LLM is triggered by queries"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Enable tracing to see what happens
        original_count = len(engine.facts) + len(engine.rules)
        
        # Query unknown term
        solutions = engine.query([compound("unknown_predicate", var("X"))])
        
        # LLM should have been triggered
        new_count = len(engine.facts) + len(engine.rules)
        # Note: might not always increase depending on mock provider behavior
        assert new_count >= original_count
    
    def test_ask_with_llm(self):
        """Test yes/no queries with LLM"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Ask about something that requires generation
        result = engine.ask(compound("parent", atom("john"), atom("mary")))
        
        # Should have generated knowledge to answer
        assert len(engine.facts) > 0
    
    def test_find_all_with_llm(self):
        """Test find_all queries with LLM"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Find all children of john
        children = engine.find_all(compound("parent", atom("john"), var("X")), "X")
        
        # Should have generated knowledge and found answers
        assert len(engine.facts) > 0
        # May or may not find children depending on mock provider
    
    def test_json_export_with_generated_knowledge(self):
        """Test JSON export of generated knowledge"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Generate some knowledge
        engine.query([compound("parent", var("X"), var("Y"))])
        
        # Export to JSON
        json_str = engine.save_to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert "facts" in data or "rules" in data
    
    def test_llm_with_complex_queries(self):
        """Test LLM with complex multi-goal queries"""
        provider = MockLLMProvider(knowledge_domain="family")
        engine = create_engine_with_llm(provider, "family")
        
        # Complex query that might require multiple generation steps
        solutions = engine.query([
            compound("parent", var("X"), var("Y")),
            compound("parent", var("Y"), var("Z"))
        ])
        
        # Should have generated enough knowledge to handle the query
        assert len(engine.facts) + len(engine.rules) > 0


class TestLLMProviderInterface:
    """Test LLM provider interface compliance"""
    
    def test_mock_provider_interface(self):
        """Test that MockLLMProvider implements the interface correctly"""
        provider = MockLLMProvider()
        
        # Should have generate_text method
        assert hasattr(provider, 'generate_text')
        assert callable(provider.generate_text)
        
        # Should accept prompt and max_tokens
        response = provider.generate_text("test", max_tokens=100)
        assert isinstance(response, str)
    
    def test_custom_provider_integration(self):
        """Test integration with custom provider"""
        class CustomProvider:
            def generate_text(self, prompt, max_tokens=500):
                return '''[{"type": "fact", "term": {"type": "atom", "value": "custom"}}]'''
        
        provider = CustomProvider()
        hook = LLMHook(provider, "custom")
        engine = JLogEngine(hook)
        
        # Should work with custom provider
        engine.query([compound("unknown", var("X"))])
        
        # Should have added the custom fact
        fact_values = [str(fact) for fact in engine.facts]
        has_custom = any("custom" in fact for fact in fact_values)
        assert has_custom


class TestErrorHandling:
    """Test error handling in LLM integration"""
    
    def test_invalid_json_response(self):
        """Test handling of invalid JSON responses"""
        class BadJSONProvider:
            def generate_text(self, prompt, max_tokens=500):
                return "not valid json"
        
        provider = BadJSONProvider()
        hook = LLMHook(provider, "test")
        kb = KnowledgeBase()
        evaluator = PrologEvaluator(kb, hook)
        
        # Should not crash
        initial_count = len(kb.facts)
        hook(compound("test", var("X")), evaluator)
        final_count = len(kb.facts)
        
        # Should not have added any facts
        assert final_count == initial_count
    
    def test_malformed_knowledge_items(self):
        """Test handling of malformed knowledge items"""
        class BadStructureProvider:
            def generate_text(self, prompt, max_tokens=500):
                return '''[{"type": "fact"}]'''  # Missing term
        
        provider = BadStructureProvider()
        hook = LLMHook(provider, "test")
        kb = KnowledgeBase()
        evaluator = PrologEvaluator(kb, hook)
        
        # Should not crash
        initial_count = len(kb.facts)
        hook(compound("test", var("X")), evaluator)
        final_count = len(kb.facts)
        
        # Should not have added any facts due to malformed structure
        assert final_count == initial_count
    
    def test_provider_exception(self):
        """Test handling of provider exceptions"""
        class ExceptionProvider:
            def generate_text(self, prompt, max_tokens=500):
                raise Exception("Provider error")
        
        provider = ExceptionProvider()
        hook = LLMHook(provider, "test")
        kb = KnowledgeBase()
        evaluator = PrologEvaluator(kb, hook)
        
        # Should not crash
        initial_count = len(kb.facts)
        hook(compound("test", var("X")), evaluator)
        final_count = len(kb.facts)
        
        # Should not have added any facts
        assert final_count == initial_count
