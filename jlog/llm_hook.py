"""
LLM Integration for JLOG

Provides hooks for automatic knowledge generation when undefined terms are encountered.
"""

from typing import List, Dict, Any, Optional, Protocol, Callable
from abc import ABC, abstractmethod
import json
import re
from .terms import Term, Atom, Variable, Compound, term_from_json, atom, var, compound
from .knowledge import Fact, Rule
from .llm_providers import LLMProvider


class LLMHook:
    """
    Hook that integrates with the Prolog evaluator to generate knowledge on demand
    """
    
    def __init__(self, llm_provider: LLMProvider, knowledge_domain: str = "general"):
        self.llm_provider = llm_provider
        self.knowledge_domain = knowledge_domain
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._generation_count = 0
        self._max_generations = 10  # Prevent infinite generation loops
    
    def __call__(self, unknown_term: Term, evaluator) -> None:
        """
        Called when the evaluator encounters an unknown term
        
        Args:
            unknown_term: The term that couldn't be resolved
            evaluator: The PrologEvaluator instance
        """
        term_key = str(unknown_term)
        
        # Check cache first
        if term_key in self._cache:
            knowledge_items = self._cache[term_key]
        else:
            # Check generation limit before generating new knowledge
            if self._generation_count >= self._max_generations:
                print(f"Generation limit ({self._max_generations}) reached, skipping generation for {unknown_term}")
                return
            
            self._generation_count += 1
            knowledge_items = self._generate_knowledge(unknown_term, evaluator.kb)
            self._cache[term_key] = knowledge_items
    
        # Add generated knowledge to the knowledge base
        added_count = 0
        for item in knowledge_items:
            try:
                if item.get("type") == "fact" and "term" in item:
                    term_obj = term_from_json(item["term"])
                    fact = Fact(term_obj)
                    if fact not in evaluator.kb.facts:
                        evaluator.kb.add_fact(fact)
                        added_count += 1
                elif item.get("type") == "rule" and "head" in item and "body" in item:
                    head_term = term_from_json(item["head"])
                    body_terms = [term_from_json(body_item) for body_item in item["body"]]
                    rule = Rule(head_term, body_terms)
                    if rule not in evaluator.kb.rules:
                        evaluator.kb.add_rule(rule)
                        added_count += 1
            except Exception as e:
                print(f"Error adding knowledge item: {e}")
                continue

        if added_count > 0:
            print(f"LLM generated {added_count} new knowledge items for {unknown_term}")
    
    def _generate_knowledge(self, term: Term, kb) -> List[Dict[str, Any]]:
        """Generate knowledge for an unknown term using the LLM"""
        
        # Create prompt for the LLM
        prompt = self._create_prompt(term, kb)
        
        try:
            # Get response from LLM
            response = self.llm_provider.generate_text(prompt)
            
            # Parse the JSON response
            knowledge_items = json.loads(response)
            
            if not isinstance(knowledge_items, list):
                knowledge_items = [knowledge_items]
            
            return knowledge_items
            
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
            return []
        except Exception as e:
            print(f"Error generating knowledge: {e}")
            return []
    
    def _create_prompt(self, term: Term, kb) -> str:
        """Create a prompt for the LLM"""
        
        # Get current knowledge base state
        kb_json = kb.to_json()
        
        # Extract the functor from the term
        functor = None
        if isinstance(term, Compound):
            functor = term.functor
        elif isinstance(term, Atom):
            functor = term.value
        
        prompt = f"""You are an expert in the {self.knowledge_domain} domain. 

A Prolog-like system is trying to resolve the term: {term}

Current knowledge base:
{kb_json}

The term "{term}" is not currently defined. Please generate relevant facts and/or rules in JSON format that would help resolve queries about this term.

Focus on the functor "{functor}" if it's a compound term.

Return a JSON array containing fact and rule objects using this exact format:

For facts:
{{
  "type": "fact",
  "term": {{
    "type": "compound",
    "functor": "predicate_name",
    "args": [
      {{"type": "atom", "value": "constant1"}},
      {{"type": "atom", "value": "constant2"}}
    ]
  }}
}}

For rules:
{{
  "type": "rule", 
  "head": {{
    "type": "compound",
    "functor": "predicate_name",
    "args": [
      {{"type": "variable", "name": "X"}},
      {{"type": "variable", "name": "Y"}}
    ]
  }},
  "body": [
    {{
      "type": "compound",
      "functor": "condition_predicate",
      "args": [
        {{"type": "variable", "name": "X"}},
        {{"type": "variable", "name": "Z"}}
      ]
    }}
  ]
}}

Generate knowledge that is appropriate for the {self.knowledge_domain} domain. Ensure the JSON is valid and follows the exact format above.

Response (JSON array only):"""
        
        return prompt
    
    def clear_cache(self) -> None:
        """Clear the knowledge generation cache"""
        self._cache.clear()
