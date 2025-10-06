"""
LLM Retry Wrapper with JSON validation and multiple sampling

Wraps LLM providers to add:
- Automatic retries on parse failures
- JSON format enforcement
- Multiple sampling strategies
- Beam search simulation

Uses DreamLog's native parser and validator for proper abstraction.
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import time

from .llm_providers import LLMProvider, LLMResponse
from .validation_feedback import OutputValidator
from .llm_response_parser import parse_llm_response


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    max_samples: int = 5  # For beam search
    temperature_increase: float = 0.1  # Increase temperature on retry
    timeout_per_call: int = 30
    enforce_json: bool = True
    json_format_hint: str = "Return ONLY valid JSON, no other text."
    verbose: bool = False


class RetryLLMProvider(LLMProvider):
    """
    Wrapper that adds retry logic and JSON validation to any LLM provider
    """
    
    def __init__(self, base_provider: LLMProvider, config: Optional[RetryConfig] = None):
        self.base_provider = base_provider
        self.config = config or RetryConfig()
        self.model = getattr(base_provider, 'model', 'unknown')
        self.validator = OutputValidator(verbose=self.config.verbose)
        
    def generate_knowledge(self, term: str, context: str = "") -> LLMResponse:
        """
        Generate knowledge with retries and JSON validation
        """
        # First attempt
        try:
            response = self.base_provider.generate_knowledge(term, context)
            analysis = self.validator.analyze_output(response.raw_response)
        except Exception as e:
            if self.config.verbose:
                print(f"[RETRY] First attempt failed: {e}")
            # Retry with strategies
            response = None
            analysis = {'valid': False, 'parsed': None, 'close_but_wrong': False, 'score': 0}
        
        if response and analysis['valid'] and analysis['parsed']:
            return LLMResponse(
                text=response.text,
                facts=analysis['parsed'].get('facts', []),
                rules=analysis['parsed'].get('rules', []),
                raw_response=response.raw_response
            )
        
        # If close but wrong, try with specific feedback
        if analysis['close_but_wrong'] and analysis['score'] > 0.5:
            if self.config.verbose:
                print(f"[RETRY] Output close but needs correction (score: {analysis['score']:.1f})")
            
            feedback = self.validator.generate_feedback_prompt(analysis)
            retry_response = self._retry_with_specific_feedback(term, context, feedback)
            if retry_response:
                return retry_response
        
        # Try different strategies
        strategies = [
            self._single_attempt,
            self._multi_sample,
            self._temperature_sweep,
            self._format_repair
        ]
        
        for strategy in strategies:
            response = strategy(term, context)
            if response and (response.facts or response.rules):
                return response
                
        # Final fallback - return empty response
        if self.config.verbose:
            print(f"[RETRY] All strategies failed for term: {term}")
        return LLMResponse(text="", facts=[], rules=[], raw_response="")
    
    def _single_attempt(self, term: str, context: str) -> Optional[LLMResponse]:
        """Single attempt with JSON enforcement"""
        if self.config.verbose:
            print(f"[RETRY] Strategy: Single attempt")
            
        # Add JSON format hint to context with clear examples
        enhanced_context = context
        if self.config.enforce_json:
            format_examples = """
Output must be a JSON array in EXACTLY this format:
[["rule", ["predicate", "X", "Y"], [["condition1", "X", "Z"], ["condition2", "Z", "Y"]]]]
or
[["fact", ["predicate", "value1", "value2"]]]

Return ONLY the JSON array, no other text."""
            enhanced_context = f"{context}\n{format_examples}"
        
        response = self.base_provider.generate_knowledge(term, enhanced_context)
        
        # Try to parse
        parsed = self._try_parse_response(response.raw_response)
        if parsed:
            return LLMResponse(
                text=response.text,
                facts=parsed.get('facts', []),
                rules=parsed.get('rules', []),
                raw_response=response.raw_response
            )
        return None
    
    def _retry_with_specific_feedback(self, term: str, context: str, feedback: str) -> Optional[LLMResponse]:
        """Retry with specific validation feedback"""
        if self.config.verbose:
            print(f"[RETRY] Retrying with validation feedback")
        
        enhanced_context = f"{context}\n\n{feedback}"
        response = self.base_provider.generate_knowledge(term, enhanced_context)
        
        # Validate the retry
        analysis = self.validator.analyze_output(response.raw_response)
        if analysis['valid'] and analysis['parsed']:
            return LLMResponse(
                text=response.text,
                facts=analysis['parsed'].get('facts', []),
                rules=analysis['parsed'].get('rules', []),
                raw_response=response.raw_response
            )
        return None
    
    def _retry_with_feedback(self, term: str, context: str, prev_response: str, error_msg: str) -> Optional[LLMResponse]:
        """Retry with feedback about what was wrong"""
        if self.config.verbose:
            print(f"[RETRY] Strategy: Retry with error feedback")
        
        # Create enhanced context with error feedback
        feedback_context = f"""{context}

Your previous response had errors:
{error_msg}

Correct format examples:
[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
[["fact", ["parent", "alice", "bob"]]]

Please provide a correctly formatted JSON array:"""
        
        response = self.base_provider.generate_knowledge(term, feedback_context)
        
        # Try to parse the corrected response
        parsed = self._try_parse_response(response.raw_response)
        if parsed:
            return LLMResponse(
                text=response.text,
                facts=parsed.get('facts', []),
                rules=parsed.get('rules', []),
                raw_response=response.raw_response
            )
        return None
    
    def _multi_sample(self, term: str, context: str) -> Optional[LLMResponse]:
        """Try multiple samples (beam search simulation)"""
        if self.config.verbose:
            print(f"[RETRY] Strategy: Multi-sample (n={self.config.max_samples})")
        
        responses = []
        # Use parameter methods for clean temperature management
        original_temp = self.base_provider.get_parameter('temperature', 0.7)
        
        # Generate multiple samples
        for i in range(self.config.max_samples):
            # Vary temperature slightly for diversity
            temp_variation = 0.1 * (i - self.config.max_samples // 2)
            varied_temp = max(0.1, original_temp + temp_variation)
            self.base_provider.set_parameter('temperature', varied_temp)
            
            response = self.base_provider.generate_knowledge(term, context)
            parsed = self._try_parse_response(response.raw_response)
            
            if parsed:
                # Score the response (simple heuristic)
                score = len(parsed.get('facts', [])) + len(parsed.get('rules', [])) * 2
                responses.append((score, parsed, response.raw_response))
                
                if self.config.verbose:
                    print(f"[RETRY]   Sample {i+1}: Success (score={score})")
                    
                # Early exit if we get a good response
                if score > 0:
                    break
            else:
                if self.config.verbose:
                    print(f"[RETRY]   Sample {i+1}: Parse failed")
        
        # Restore original temperature
        self.base_provider.set_parameter('temperature', original_temp)
        
        # Return best response
        if responses:
            responses.sort(reverse=True)  # Sort by score
            best = responses[0]
            return LLMResponse(
                text="",  # Empty text since we're using parsed data
                facts=best[1].get('facts', []),
                rules=best[1].get('rules', []),
                raw_response=best[2]
            )
        return None
    
    def _temperature_sweep(self, term: str, context: str) -> Optional[LLMResponse]:
        """Try different temperatures"""
        if self.config.verbose:
            print(f"[RETRY] Strategy: Temperature sweep")
            
        original_temp = self.base_provider.get_parameter('temperature', 0.7)
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.0]
        
        for temp in temperatures:
            self.base_provider.set_parameter('temperature', temp)
            response = self.base_provider.generate_knowledge(term, context)
            parsed = self._try_parse_response(response.raw_response)
            
            if parsed:
                if self.config.verbose:
                    print(f"[RETRY]   Temperature {temp}: Success")
                self.base_provider.set_parameter('temperature', original_temp)
                return LLMResponse(
                    text=response.text,
                    facts=parsed.get('facts', []),
                    rules=parsed.get('rules', []),
                    raw_response=response.raw_response
                )
                
        self.base_provider.set_parameter('temperature', original_temp)
        return None
    
    def _format_repair(self, term: str, context: str) -> Optional[LLMResponse]:
        """Try to repair malformed output using DreamLog's flexible parser"""
        if self.config.verbose:
            print(f"[RETRY] Strategy: Format repair")
            
        response = self.base_provider.generate_knowledge(term, context)
        
        # DreamLog's parser already handles extraction and repair internally
        # It tries multiple strategies including JSON extraction, S-expression parsing, etc.
        parsed = self._try_parse_response(response.raw_response)
        if parsed:
            return LLMResponse(
                text=response.text,
                facts=parsed.get('facts', []),
                rules=parsed.get('rules', []),
                raw_response=response.raw_response
            )
        return None
    
    
    def _try_parse_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Try to parse LLM response using DreamLog's native parser"""
        if not raw_response:
            return None
        
        # Use DreamLog's native parser with non-strict mode
        parsed_knowledge, validation_report = parse_llm_response(
            raw_response,
            strict=False,
            validate=True
        )
        
        # Check if we got any valid knowledge
        if parsed_knowledge.facts or parsed_knowledge.rules:
            # Convert to expected format
            return {
                'facts': [fact.term.to_prefix() for fact in parsed_knowledge.facts],
                'rules': [[rule.head.to_prefix(), [t.to_prefix() for t in rule.body]] 
                         for rule in parsed_knowledge.rules]
            }
        
        # Store parsing errors for potential retry with feedback
        if self.config.verbose and parsed_knowledge.errors:
            print(f"[RETRY] Parsing errors: {', '.join(parsed_knowledge.errors[:3])}")
        
        return None
    
    # LLMProvider protocol methods - delegate to base provider
    def complete(self, prompt: str, **kwargs) -> str:
        """General text completion with retry logic"""
        return self.base_provider.complete(prompt, **kwargs)
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter from base provider"""
        return self.base_provider.get_parameter(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Update configuration parameter on base provider"""
        self.base_provider.set_parameter(key, value)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Enhanced metadata including retry configuration"""
        base_metadata = self.base_provider.get_metadata()
        base_metadata.update({
            "wrapper": "RetryLLMProvider",
            "retry_config": {
                "max_retries": self.config.max_retries,
                "max_samples": self.config.max_samples,
                "temperature_increase": self.config.temperature_increase,
                "enforce_json": self.config.enforce_json
            }
        })
        return base_metadata
    
    def clone_with_parameters(self, **params) -> 'RetryLLMProvider':
        """Create copy with modified parameters"""
        new_base = self.base_provider.clone_with_parameters(**params)
        return RetryLLMProvider(new_base, self.config)


class OllamaRetryProvider(RetryLLMProvider):
    """
    Specialized retry provider for Ollama with JSON format enforcement
    """
    
    def __init__(self, base_provider: LLMProvider, config: Optional[RetryConfig] = None):
        super().__init__(base_provider, config)
        
        # Enhance Ollama to support JSON format if possible
        if hasattr(base_provider, 'base_url'):
            self._enhance_for_json()
    
    def _enhance_for_json(self):
        """Enhance Ollama provider for better JSON output"""
        # Monkey-patch the _call_api method to add format parameter
        original_call = self.base_provider._call_api
        
        def enhanced_call(prompt: str, **kwargs):
            # Add format hint to Ollama
            if 'format' not in kwargs:
                kwargs['format'] = 'json'  # Tell Ollama we want JSON
            
            # Also enhance the prompt
            if "JSON" not in prompt.upper():
                prompt = f"{prompt}\n\nOutput format: JSON array following this structure: [[\"rule\", [head], [body]]] or [[\"fact\", [predicate, args...]]]"
            
            return original_call(prompt, **kwargs)
        
        self.base_provider._call_api = enhanced_call


def create_retry_provider(provider: LLMProvider, 
                         max_retries: int = 3,
                         verbose: bool = False) -> RetryLLMProvider:
    """
    Factory function to create a retry-enabled provider
    
    Args:
        provider: Base LLM provider
        max_retries: Maximum number of retry attempts
        verbose: Whether to print debug information
    
    Returns:
        RetryLLMProvider wrapping the base provider
    """
    config = RetryConfig(
        max_retries=max_retries,
        verbose=verbose,
        enforce_json=True
    )
    
    # Use specialized Ollama wrapper if it's an Ollama provider
    if provider.__class__.__name__ == 'OllamaProvider':
        return OllamaRetryProvider(provider, config)
    
    return RetryLLMProvider(provider, config)