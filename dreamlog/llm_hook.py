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
from .prompt_template_system import PromptTemplateLibrary, QueryContext, RULE_EXAMPLES
from .example_retriever import ExampleRetriever
from .rule_validator import RuleValidator
from .llm_judge import LLMJudge, VerificationPipeline
from .correction_retry import CorrectionBasedRetry
from .llm_response_parser import DreamLogResponseParser


class LLMHook:
    """
    Hook for automatic knowledge generation when undefined terms are encountered
    """
    
    def __init__(self,
                 provider: LLMProvider,
                 embedding_provider,
                 max_generations: int = 10,
                 max_retries: int = 5,
                 cache_enabled: bool = True,
                 debug: bool = False,
                 debug_callback: Optional[callable] = None,
                 enable_validation: bool = True,
                 enable_llm_judge: bool = False,
                 enable_correction_retry: bool = False):
        """
        Initialize LLM hook

        Args:
            provider: LLM provider to use for generation
            embedding_provider: EmbeddingProvider for RAG-based example selection (required)
            max_generations: Maximum number of generation calls per session
            max_retries: Maximum number of retry attempts for invalid responses (default 5)
            cache_enabled: Whether to cache generated knowledge
            debug: Enable debug output
            debug_callback: Optional callback for debug messages (for TUI integration)
            enable_validation: Enable structural/semantic validation (default True)
            enable_llm_judge: Enable LLM-as-judge verification (default False, expensive)
            enable_correction_retry: Enable correction-based retry (default False, requires LLM judge)
        """
        self.provider = provider
        self.max_generations = max_generations
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self.debug = debug
        self.debug_callback = debug_callback
        self.enable_validation = enable_validation
        self.enable_llm_judge = enable_llm_judge
        self.enable_correction_retry = enable_correction_retry
        self._cache: Dict[str, LLMResponse] = {}
        self._generation_count = 0

        # Initialize example retriever for RAG (always enabled)
        self._debug("[LLM] Initializing RAG-based example retriever...")
        example_retriever = ExampleRetriever(RULE_EXAMPLES, embedding_provider)
        self._debug(f"[LLM] Precomputed embeddings for {len(RULE_EXAMPLES)} examples")

        # Initialize prompt template library
        # Extract model name from provider if available
        model_name = getattr(provider, 'model', 'unknown')
        self.template_library = PromptTemplateLibrary(model_name=model_name, example_retriever=example_retriever)

        # Initialize validation components
        self.validator = None
        self.llm_judge = None
        self.correction_retry = None
        self.parser = DreamLogResponseParser()

        if self.enable_validation:
            self._debug("[LLM] Validation enabled")

        if self.enable_llm_judge:
            self.llm_judge = LLMJudge(provider, debug=debug)
            self._debug("[LLM] LLM judge enabled")

        if self.enable_correction_retry:
            if not self.enable_llm_judge:
                self._debug("[LLM] Warning: Correction retry requires LLM judge, enabling it")
                self.enable_llm_judge = True
                self.llm_judge = LLMJudge(provider, debug=debug)

            self.correction_retry = CorrectionBasedRetry(
                provider=provider,
                judge=self.llm_judge,
                parser=self.parser,
                max_attempts=3,
                debug=debug
            )
            self._debug("[LLM] Correction-based retry enabled")

    def _debug(self, message: str):
        """Output debug message through callback or print"""
        if self.debug:
            if self.debug_callback:
                self.debug_callback(message)
            else:
                print(message)
    
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

        # Skip LLM generation for predicates that are base facts
        # (already have facts in KB - should not be defined as rules)
        if isinstance(unknown_term, Compound):
            functor = unknown_term.functor
            has_facts = any(
                isinstance(f.term, Compound) and f.term.functor == functor
                for f in evaluator.kb.facts
            )
            if has_facts:
                self._debug(f"\n[LLM] Skipping '{functor}' - base predicate with existing facts")
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

        self._debug(f"\n[LLM] Response for '{unknown_term}':")
        self._debug(f"  Facts: {response.facts}")
        self._debug(f"  Rules: {response.rules}")

        # Skip facts - LLM should only generate rules, not facts
        # Facts should be manually added by users
        if response.facts:
            self._debug(f"  Note: Ignoring {len(response.facts)} facts from LLM (only rules are auto-generated)")
        
        for rule_data in response.rules:
            try:
                # Validate rule data structure
                if not isinstance(rule_data, (list, tuple)) or len(rule_data) != 2:
                    self._debug(f"  Skipping invalid rule data: {rule_data}")
                    continue

                head, body = rule_data
                if not isinstance(head, (list, tuple)) or not isinstance(body, (list, tuple)):
                    self._debug(f"  Skipping rule with invalid head/body: {rule_data}")
                    continue

                rule = Rule.from_prefix(["rule", head, body])

                # Validate the rule before adding
                if self.enable_validation:
                    # Initialize validator with current KB
                    if not self.validator:
                        self.validator = RuleValidator(evaluator.kb)
                    else:
                        # Update validator's KB reference
                        self.validator.semantic_validator.kb = evaluator.kb

                    # Run structural and semantic validation
                    validation_result = self.validator.validate(rule, structural=True, semantic=True)

                    if not validation_result.is_valid:
                        self._debug(f"  ✗ Rule failed validation: {validation_result.error_message}")
                        self._debug(f"    Rule: {rule}")
                        continue

                    if validation_result.warning_message:
                        self._debug(f"  ⚠ Warning: {validation_result.warning_message}")

                    # Run LLM judge verification if enabled
                    if self.enable_llm_judge and self.llm_judge:
                        functor = unknown_term.functor if isinstance(unknown_term, Compound) else str(unknown_term)

                        verification = VerificationPipeline(
                            knowledge_base=evaluator.kb,
                            llm_judge=self.llm_judge,
                            debug=self.debug
                        )

                        result = verification.verify_rule(
                            rule=rule,
                            query_functor=functor,
                            use_llm_judge=True
                        )

                        if not result["valid"]:
                            self._debug(f"  ✗ Rule failed verification:")
                            for error in result["errors"]:
                                self._debug(f"    - {error}")

                            if result["llm_judgement"] and result["llm_judgement"].suggested_correction:
                                self._debug(f"    Suggested: {result['llm_judgement'].suggested_correction}")

                            continue

                        if result["warnings"]:
                            for warning in result["warnings"]:
                                self._debug(f"  ⚠ {warning}")

                # Use proper containment check
                if not any(r.head == rule.head and r.body == rule.body for r in evaluator.kb.rules):
                    evaluator.kb.add_rule(rule)
                    added_count += 1
                    self._debug(f"  ✓ Added rule: {rule.head} :- {rule.body}")
            except Exception as e:
                self._debug(f"  Error adding rule {rule_data}: {e}")
        
        if added_count > 0:
            print(f"LLM generated {added_count} new knowledge items for {unknown_term}")
    
    def _generate_knowledge(self, term: Term, kb) -> LLMResponse:
        """Generate knowledge for an unknown term with retry logic"""

        # Create query context for template manager using JSON format
        query_context = QueryContext(
            term=term.to_prefix(),  # JSON format: ["functor", "arg1", ...]
            kb_facts=[fact.term.to_prefix() for fact in kb.facts[:20]],  # JSON facts
            kb_rules=[(rule.head.to_prefix(), [g.to_prefix() for g in rule.body]) for rule in kb.rules[:10]],  # JSON rules
            existing_functors=list(set(
                fact.term.functor for fact in kb.facts
                if isinstance(fact.term, Compound)
            ))[:20]
        )

        # Retry loop with different examples each time
        for attempt in range(self.max_retries):
            # Get the best prompt from template library (samples new examples each time)
            prompt, template_name = self.template_library.get_best_prompt(query_context)

            if attempt == 0:
                self._debug(f"\n[LLM] Calling LLM for undefined term: {str(term)}")
            else:
                self._debug(f"\n[LLM] Retry attempt {attempt + 1}/{self.max_retries}")

            self._debug(f"[LLM] Using template: {template_name}")
            self._debug(f"[LLM] Prompt length: {len(prompt)} characters")

            # Always show the full prompt when debug is enabled
            self._debug(f"\n[LLM] === PROMPT START ===")
            self._debug(prompt)
            self._debug(f"[LLM] === PROMPT END ===\n")

            # Call provider directly with the complete prompt from template library
            # Use complete() method since we already have a full prompt
            raw_response = self.provider.complete(prompt)

            # Parse the response into structured format
            from .llm_providers import LLMResponse

            # Always show raw response for debugging
            if len(raw_response) > 500:
                self._debug(f"[LLM] Raw response: {raw_response[:500]}...")
            else:
                self._debug(f"[LLM] Raw response: {raw_response}")

            response = LLMResponse.from_text(raw_response)

            # Check if response is valid (has facts or rules)
            is_valid = len(response.facts) > 0 or len(response.rules) > 0

            if is_valid:
                self._debug(f"[LLM] Valid response received on attempt {attempt + 1}")
                # Track template performance
                self.template_library.record_performance(
                    template_name=template_name,
                    success=True,
                    response_quality=1.0
                )
                return response
            else:
                self._debug(f"[LLM] Invalid response (no facts or rules), retrying...")
                # Track failure
                self.template_library.record_performance(
                    template_name=template_name,
                    success=False,
                    response_quality=0.0
                )

        # All retries exhausted, return last response even if invalid
        self._debug(f"[LLM] Max retries exhausted, returning last response")
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