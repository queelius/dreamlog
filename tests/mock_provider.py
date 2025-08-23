"""
Mock LLM Provider for Testing

This mock provider returns deterministic responses for testing DreamLog
without requiring actual LLM API calls.
"""

import json
import re
from typing import Optional
from dreamlog.llm_providers import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Mock provider for testing
    
    Generates deterministic responses based on the term.
    Used only in tests to avoid API calls and ensure reproducible results.
    """
    
    def __init__(self, knowledge_domain: str = "general", **kwargs):
        super().__init__(**kwargs)
        self.knowledge_domain = knowledge_domain
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
    
    def _call_api(self, prompt: str, **kwargs) -> str:
        """Simulate API call with deterministic response"""
        # Extract functor from prompt if possible
        functor_match = re.search(r'term:\s*(\w+)', prompt.lower())
        
        if functor_match and self.knowledge_domain in self.domains:
            functor = functor_match.group(1)
            domain_data = self.domains[self.knowledge_domain]
            
            if functor in domain_data:
                return json.dumps(domain_data[functor])
        
        # Check for specific dream/optimization requests
        if "compression" in prompt.lower() or "optimize" in prompt.lower():
            return json.dumps({
                "compressed_rule": "(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)",
                "explanation": "Merged brother and sister rules into general sibling rule"
            })
        
        if "abstraction" in prompt.lower():
            return json.dumps([{
                "original_rules": ["(brother X Y) :- ...", "(sister X Y) :- ..."],
                "abstract_rule": "(sibling X Y) :- (parent Z X), (parent Z Y), (different X Y)",
                "benefit": "More general rule covers both cases"
            }])
        
        # Default response
        return json.dumps([
            ["fact", ["predicate", "value1", "value2"]],
            ["fact", ["predicate", "value3", "value4"]]
        ])