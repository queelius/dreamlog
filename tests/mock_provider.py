"""
Mock LLM Provider for Testing

This mock provider returns deterministic responses for testing DreamLog
without requiring actual LLM API calls.
"""

import json
import re
from typing import Optional, Dict, Any
from dreamlog.llm_providers import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Mock provider for testing
    
    Generates deterministic responses based on the term.
    Used only in tests to avoid API calls and ensure reproducible results.
    """
    
    def __init__(self, knowledge_domain: str = "general", **kwargs):
        # Set defaults for mock provider
        defaults = {
            "model": "mock-model-v1.0",
            "temperature": 0.1,
            "max_tokens": 500,
            "deterministic": True,
            "knowledge_domain": knowledge_domain
        }
        
        # Merge defaults with provided kwargs
        merged_kwargs = {**defaults, **kwargs}
        
        # Initialize base provider with all parameters
        super().__init__(**merged_kwargs)
        
        # Mock-specific attributes
        self.custom_responses = {}  # For test-specific responses
        self.call_count = 0  # For tracking calls in tests
        self.domains = {
            "family": {
                "parent": [
                    ["fact", ["parent", "john", "mary"]],
                    ["fact", ["parent", "mary", "alice"]],
                    ["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]
                ],
                "sibling": [
                    ["fact", ["sibling", "alice", "bob"]],
                    ["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"], ["different", "X", "Y"]]]
                ],
                "grandparent": [
                    ["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]
                ],
                "healthy": [
                    ["fact", ["healthy", "alice"]],
                    ["fact", ["exercises", "alice"]],
                    ["fact", ["eats_well", "alice"]],
                    ["rule", ["healthy", "X"], [["exercises", "X"], ["eats_well", "X"]]]
                ],
                "uncle": [
                    ["rule", ["uncle", "X", "Y"], [["brother", "X", "Z"], ["parent", "Z", "Y"]]],
                    ["rule", ["uncle", "X", "Y"], [["male", "X"], ["sibling", "X", "Z"], ["parent", "Z", "Y"]]]
                ]
            },
            "academic": {
                "enrolled": [
                    ["fact", ["enrolled", "alice", "cs101"]],
                    ["fact", ["enrolled", "bob", "cs101"]],
                    ["rule", ["classmates", "X", "Y"], [["enrolled", "X", "C"], ["enrolled", "Y", "C"], ["different", "X", "Y"]]]
                ],
                "passing": [
                    ["fact", ["passing", "alice", "cs101"]],
                    ["rule", ["passing", "X", "C"], [["grade", "X", "C", "G"], ["greater", "G", "60"]]]
                ]
            }
        }
    
    def add_response(self, key: str, facts=None, rules=None):
        """Add a custom response for a specific key (for testing)"""
        response = []
        if facts:
            for fact in facts:
                response.append(["fact", fact])
        if rules:
            for rule in rules:
                response.append(["rule"] + rule)
        self.custom_responses[key.lower()] = json.dumps(response)
    
    def _call_api(self, prompt: str, **kwargs) -> str:
        """Simulate API call with deterministic response"""
        # Track calls for testing
        self.call_count += 1
        # Check custom responses first
        for key, response in self.custom_responses.items():
            if key in prompt.lower():
                return response
        
        # Check for grandparent specifically (common test case) - but only if it's in the query, not the example
        if 'query:' in prompt.lower() and 'grandparent' in prompt.lower().split('query:')[1].split('\n')[0]:
            return json.dumps([["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]])
        
        # Extract functor from prompt - try multiple patterns
        functor = None
        
        # Pattern 1: "Query: (functor ...)" or "Query: functor"
        query_match = re.search(r'query:\s*\(?(\w+)', prompt.lower())
        if query_match:
            functor = query_match.group(1)
        
        # Pattern 2: "term: functor" 
        if not functor:
            term_match = re.search(r'term:\s*(\w+)', prompt.lower())
            if term_match:
                functor = term_match.group(1)
        
        # Pattern 3: Look for domain-specific functors directly
        if not functor and self.knowledge_domain in self.domains:
            domain_data = self.domains[self.knowledge_domain]
            for key in domain_data.keys():
                if key in prompt.lower():
                    functor = key
                    break
        
        # Return domain data if we found a functor
        knowledge_domain = self.get_parameter("knowledge_domain", "general")
        if functor and knowledge_domain in self.domains:
            domain_data = self.domains[knowledge_domain]
            if functor in domain_data:
                return json.dumps(domain_data[functor])
        
        # Check for specific dream/optimization requests
        if "compression" in prompt.lower() or "optimize" in prompt.lower() or "_compress" in prompt.lower():
            return json.dumps({
                "compressed_rule": "(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)",
                "explanation": "Merged brother and sister rules into general sibling rule"
            })
        
        if "abstraction" in prompt.lower() or "_find_abstractions" in prompt.lower():
            return json.dumps([{
                "original_rules": ["(brother X Y) :- ...", "(sister X Y) :- ..."],
                "abstract_rule": "(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)",
                "benefit": "More general rule covers both cases"
            }])
        
        if "_test_queries" in prompt.lower():
            return json.dumps(["(parent john X)", "(sibling X Y)"])
        
        if "_evaluate_diff" in prompt.lower():
            return json.dumps({"is_improvement": True, "is_acceptable": True, "reason": "Good optimization"})
        
        # Default response
        return json.dumps([
            ["fact", ["predicate", "value1", "value2"]],
            ["fact", ["predicate", "value3", "value4"]]
        ])
    
    def get_metadata(self) -> Dict[str, Any]:
        """Enhanced metadata for mock provider"""
        base_metadata = super().get_metadata()
        base_metadata.update({
            "provider_type": "mock",
            "deterministic": self.get_parameter("deterministic", True),
            "knowledge_domain": self.get_parameter("knowledge_domain", "general"),
            "call_count": self.call_count,
            "available_domains": list(self.domains.keys()),
            "capabilities": ["knowledge_generation", "text_completion", "deterministic_responses"]
        })
        return base_metadata