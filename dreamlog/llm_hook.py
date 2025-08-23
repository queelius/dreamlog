"""
Simplified LLM Hook for DreamLog

Integrates with the evaluator to generate knowledge on demand using the new LLM system.
"""

from typing import Dict, List, Optional
import random
from .terms import Term, Compound, Atom
from .knowledge import Fact, Rule
from .llm_providers import LLMProvider, LLMResponse
from .config import get_config
from .prompt_template_system import PromptTemplateLibrary, QueryContext


class LLMHook:
    """
    Hook for automatic knowledge generation when undefined terms are encountered
    """
    
    def __init__(self, 
                 provider: LLMProvider,
                 max_generations: int = 10,
                 cache_enabled: bool = True,
                 debug: bool = False):
        """
        Initialize LLM hook
        
        Args:
            provider: LLM provider to use for generation
            max_generations: Maximum number of generation calls per session
            cache_enabled: Whether to cache generated knowledge
        """
        self.provider = provider
        self.max_generations = max_generations
        self.cache_enabled = cache_enabled
        self.debug = debug
        self._cache: Dict[str, LLMResponse] = {}
        self._generation_count = 0
        
        # Initialize prompt template library
        # Extract model name from provider if available
        model_name = getattr(provider, 'model', 'unknown')
        self.template_library = PromptTemplateLibrary(model_name=model_name)
    
    def __call__(self, unknown_term: Term, evaluator) -> None:
        """
        Called when the evaluator encounters an unknown term
        
        Args:
            unknown_term: The term that couldn't be resolved
            evaluator: The PrologEvaluator instance
        """
        # Check generation limit
        if self._generation_count >= self.max_generations:
            return
        
        # Get or generate knowledge
        term_key = str(unknown_term)
        
        if self.cache_enabled and term_key in self._cache:
            response = self._cache[term_key]
        else:
            response = self._generate_knowledge(unknown_term, evaluator.kb)
            if self.cache_enabled:
                self._cache[term_key] = response
            self._generation_count += 1
        
        # Add knowledge to the knowledge base
        added_count = 0
        
        if self.debug:
            print(f"\n[DEBUG] LLM Response for '{unknown_term}':")
            print(f"  Facts: {response.facts}")
            print(f"  Rules: {response.rules}")
        
        for fact_data in response.facts:
            try:
                # Validate fact data
                if not isinstance(fact_data, (list, tuple)):
                    if self.debug:
                        print(f"  Skipping invalid fact data: {fact_data}")
                    continue
                    
                # Create fact from prefix notation
                fact = Fact.from_prefix(["fact", fact_data])
                # Use proper containment check
                if not any(f.term == fact.term for f in evaluator.kb.facts):
                    evaluator.kb.add_fact(fact)
                    added_count += 1
                    if self.debug:
                        print(f"  Added fact: {fact.term}")
            except Exception as e:
                print(f"Error adding fact {fact_data}: {e}")
        
        for rule_data in response.rules:
            try:
                # Validate rule data structure
                if not isinstance(rule_data, (list, tuple)) or len(rule_data) != 2:
                    if self.debug:
                        print(f"  Skipping invalid rule data: {rule_data}")
                    continue
                    
                head, body = rule_data
                if not isinstance(head, (list, tuple)) or not isinstance(body, (list, tuple)):
                    if self.debug:
                        print(f"  Skipping rule with invalid head/body: {rule_data}")
                    continue
                    
                rule = Rule.from_prefix(["rule", head, body])
                # Use proper containment check
                if not any(r.head == rule.head and r.body == rule.body for r in evaluator.kb.rules):
                    evaluator.kb.add_rule(rule)
                    added_count += 1
                    if self.debug:
                        print(f"  Added rule: {rule.head} :- {rule.body}")
            except Exception as e:
                if self.debug:
                    print(f"  Error adding rule {rule_data}: {e}")
        
        if added_count > 0:
            print(f"LLM generated {added_count} new knowledge items for {unknown_term}")
    
    def _generate_knowledge(self, term: Term, kb) -> LLMResponse:
        """Generate knowledge for an unknown term"""
        
        # Create query context for template manager
        query_context = QueryContext(
            term=str(term),
            kb_facts=[str(fact.term) for fact in kb.facts[:20]],  # Sample facts
            kb_rules=[(str(rule.head), [str(g) for g in rule.body]) for rule in kb.rules[:10]],  # Sample rules
            existing_functors=list(set(
                fact.term.functor for fact in kb.facts 
                if isinstance(fact.term, Compound)
            ))[:20]
        )
        
        # Get the best prompt from template library
        prompt, template_name = self.template_library.get_best_prompt(query_context)
        
        if self.debug:
            print(f"\n[DEBUG] Calling LLM for term: {str(term)}")
            print(f"[DEBUG] Using template: {template_name}")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            if len(prompt) < 1000:
                print(f"[DEBUG] Full prompt:\n{prompt}")
            else:
                print(f"[DEBUG] Prompt preview:\n{prompt[:500]}...\n...\n{prompt[-500:]}")
        
        # Call provider with the generated prompt
        # Note: We pass the prompt as the context since providers expect term + context
        response = self.provider.generate_knowledge(str(term), context=prompt)
        
        if self.debug and hasattr(response, 'raw_response'):
            print(f"[DEBUG] Raw LLM response: {response.raw_response[:500]}..." if len(response.raw_response) > 500 else f"[DEBUG] Raw LLM response: {response.raw_response}")
        
        # Track template performance
        success = len(response.facts) > 0 or len(response.rules) > 0
        self.template_library.record_performance(
            template_name=template_name,
            success=success,
            response_quality=1.0 if success else 0.0  # Simple metric for now
        )
        
        return response
    
    def _extract_context(self, term: Term, kb) -> str:
        """Extract relevant context for the LLM"""
        config = get_config()
        context_parts = []
        
        # Add term type information
        if isinstance(term, Compound):
            context_parts.append(f"Compound term with functor '{term.functor}' and {len(term.args)} arguments")
            
            # Get sampling parameters from config
            max_facts = config.sampling.max_facts
            max_rules = config.sampling.max_rules
            strategy = config.sampling.strategy
            
            # Find related facts
            related_facts = []
            all_facts = []
            all_functors = set()
            
            for fact in kb.facts:
                fact_str = str(fact.term)
                all_facts.append(fact_str)
                
                if isinstance(fact.term, Compound):
                    all_functors.add(fact.term.functor)
                    # Check if related
                    if strategy == "related":
                        if fact.term.functor == term.functor:
                            related_facts.append(fact_str)
                        elif any(isinstance(arg, Atom) and arg.value == term.functor for arg in fact.term.args):
                            related_facts.append(fact_str)
            
            # Sample facts based on strategy
            if strategy == "related" and related_facts:
                facts_to_show = self._sample_items(related_facts, max_facts, "FACTS")
            elif strategy == "random":
                facts_to_show = self._sample_items(all_facts, max_facts, "FACTS")
            elif strategy == "recent":
                # Most recent facts (end of list)
                facts_to_show = all_facts[-max_facts:] if all_facts else []
                if len(all_facts) > max_facts:
                    context_parts.append(f"RECENT FACTS (showing last {max_facts} of {len(all_facts)}):")
                elif facts_to_show:
                    context_parts.append("FACTS:")
            else:
                facts_to_show = self._sample_items(all_facts, max_facts, "FACTS")
            
            context_parts.extend(facts_to_show)
            
            # Sample rules
            if kb.rules:
                rule_strs = []
                for rule in kb.rules:
                    if strategy == "related" and isinstance(rule.head, Compound):
                        # Only include related rules
                        if rule.head.functor == term.functor or \
                           any(isinstance(g, Compound) and g.functor == term.functor for g in rule.body):
                            rule_strs.append(f"{rule.head} :- {', '.join(str(g) for g in rule.body)}")
                    else:
                        # Include all rules for other strategies
                        rule_strs.append(f"{rule.head} :- {', '.join(str(g) for g in rule.body)}")
                
                if rule_strs:
                    sampled_rules = self._sample_items(rule_strs, max_rules, "RULES")
                    context_parts.extend(sampled_rules)
            
            # Add other functors for context
            if all_functors:
                context_parts.append(f"OTHER PREDICATES IN KB: {', '.join(sorted(all_functors)[:20])}")
        
        elif isinstance(term, Atom):
            context_parts.append(f"Atom: {term.value}")
            
            # Find facts mentioning this atom
            related = []
            for fact in kb.facts:
                if term.value in str(fact.term):
                    related.append(str(fact.term))
                    if len(related) >= config.sampling.max_facts:
                        break
            
            if related:
                context_parts.append(f"FACTS MENTIONING '{term.value}':")
                context_parts.extend(related)
        
        # Add KB statistics
        context_parts.append(f"\nKNOWLEDGE BASE STATS: {len(kb.facts)} facts, {len(kb.rules)} rules")
        
        return "\n".join(context_parts)
    
    def _sample_items(self, items: List[str], max_items: int, label: str) -> List[str]:
        """Sample items with appropriate header"""
        if len(items) <= max_items:
            return [f"{label}:"] + items if items else []
        
        # Sample strategy: include first half and random sample from rest
        half = max_items // 2
        sampled = items[:half] + random.sample(items[half:], max_items - half)
        return [f"SAMPLE OF {label} ({len(items)} total, showing {max_items}):"] + sampled
    
    def clear_cache(self) -> None:
        """Clear the knowledge cache"""
        self._cache.clear()
        self._generation_count = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get hook statistics"""
        return {
            "generations": self._generation_count,
            "cached_terms": len(self._cache),
            "max_generations": self.max_generations
        }