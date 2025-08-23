"""
LLM Retry Wrapper with JSON validation and multiple sampling

Wraps LLM providers to add:
- Automatic retries on parse failures
- JSON format enforcement
- Multiple sampling strategies
- Beam search simulation
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import time

from .llm_providers import LLMProvider, LLMResponse
from .validation_feedback import OutputValidator


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
        response = self.base_provider.generate_knowledge(term, context)
        analysis = self.validator.analyze_output(response.raw_response)
        
        if analysis['valid'] and analysis['parsed']:
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
        original_temp = getattr(self.base_provider, 'temperature', 0.7)
        
        # Generate multiple samples
        for i in range(self.config.max_samples):
            # Vary temperature slightly for diversity (only if provider supports it)
            if hasattr(self.base_provider, 'temperature'):
                temp_variation = 0.1 * (i - self.config.max_samples // 2)
                self.base_provider.temperature = max(0.1, original_temp + temp_variation)
            
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
        
        # Restore original temperature (if provider supports it)
        if hasattr(self.base_provider, 'temperature'):
            self.base_provider.temperature = original_temp
        
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
            
        original_temp = getattr(self.base_provider, 'temperature', 0.7)
        temperatures = [0.3, 0.5, 0.7, 0.9, 1.0]
        
        for temp in temperatures:
            if hasattr(self.base_provider, 'temperature'):
                self.base_provider.temperature = temp
            else:
                # If provider doesn't support temperature, skip this strategy
                return None
            response = self.base_provider.generate_knowledge(term, context)
            parsed = self._try_parse_response(response.raw_response)
            
            if parsed:
                if self.config.verbose:
                    print(f"[RETRY]   Temperature {temp}: Success")
                self.base_provider.temperature = original_temp
                return LLMResponse(
                    text=response.text,
                    facts=parsed.get('facts', []),
                    rules=parsed.get('rules', []),
                    raw_response=response.raw_response
                )
                
        self.base_provider.temperature = original_temp
        return None
    
    def _format_repair(self, term: str, context: str) -> Optional[LLMResponse]:
        """Try to repair malformed JSON"""
        if self.config.verbose:
            print(f"[RETRY] Strategy: Format repair")
            
        response = self.base_provider.generate_knowledge(term, context)
        
        # Try aggressive extraction
        repaired = self._extract_json_aggressively(response.raw_response)
        if repaired:
            parsed = self._try_parse_response(repaired)
            if parsed:
                return LLMResponse(
                    text=response.text,
                    facts=parsed.get('facts', []),
                    rules=parsed.get('rules', []),
                    raw_response=response.raw_response
                )
        return None
    
    def _validate_and_parse(self, parsed_json: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Validate parsed JSON matches our expected format
        Returns: (parsed_result, error_message)
        """
        facts = []
        rules = []
        errors = []
        
        # Expected format: [["rule", head, body], ["fact", data], ...]
        if not isinstance(parsed_json, list):
            return None, f"Expected JSON array, got {type(parsed_json).__name__}"
            
        for i, item in enumerate(parsed_json):
            if not isinstance(item, list):
                errors.append(f"Item {i}: Expected array, got {type(item).__name__}")
                continue
                
            if len(item) < 2:
                errors.append(f"Item {i}: Too few elements (need at least 2)")
                continue
                
            if item[0] == "fact":
                if len(item) != 2:
                    errors.append(f"Item {i}: Fact needs exactly 2 elements [\"fact\", data]")
                elif not isinstance(item[1], list):
                    errors.append(f"Item {i}: Fact data must be an array")
                else:
                    facts.append(item[1])
                    
            elif item[0] == "rule":
                if len(item) != 3:
                    errors.append(f"Item {i}: Rule needs exactly 3 elements [\"rule\", head, body]")
                else:
                    head = item[1]
                    body = item[2]
                    if not isinstance(head, list):
                        errors.append(f"Item {i}: Rule head must be an array")
                    elif not isinstance(body, list):
                        errors.append(f"Item {i}: Rule body must be an array of conditions")
                    else:
                        # Validate body contains valid predicates
                        invalid_preds = [j for j, pred in enumerate(body) if not isinstance(pred, list)]
                        if invalid_preds:
                            errors.append(f"Item {i}: Body conditions {invalid_preds} are not arrays")
                        else:
                            rules.append([head, body])
            else:
                errors.append(f"Item {i}: Unknown type '{item[0]}' (expected 'fact' or 'rule')")
                            
        if facts or rules:
            return {'facts': facts, 'rules': rules}, ""
        
        # Return errors for feedback
        error_msg = "Validation failed:\n" + "\n".join(errors) if errors else "No valid facts or rules found"
        return None, error_msg
    
    def _try_parse_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Try to parse LLM response as JSON"""
        if not raw_response:
            return None
            
        # Clean up the response
        cleaned = self._extract_json(raw_response)
        
        try:
            # Try to parse as JSON
            parsed = json.loads(cleaned)
            
            # First try strict validation
            result, error_msg = self._validate_and_parse(parsed)
            if result:
                return result
            
            # Store validation error for potential retry with feedback
            if self.config.verbose and error_msg:
                print(f"[RETRY] Validation error: {error_msg}")
            
            # If strict validation failed, try lenient parsing
            if isinstance(parsed, list):
                # It's already in the format we want
                facts = []
                rules = []
                
                for item in parsed:
                    if isinstance(item, list) and len(item) > 0:
                        if item[0] == "fact" and len(item) > 1:
                            facts.append(item[1])
                        elif item[0] == "rule" and len(item) > 2:
                            rules.append(item[1:])
                        elif len(item) == 2 and isinstance(item[1], list):
                            # Assume it's a rule [head, body]
                            rules.append(item)
                
                return {'facts': facts, 'rules': rules}
                
            return None
            
        except json.JSONDecodeError:
            return None
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might have other content"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Remove <think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Find JSON array or object
        json_patterns = [
            r'\[\[.*?\]\]',  # Nested arrays
            r'\[.*?\]',       # Single array
            r'\{.*?\}'        # Object
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
        
        return text.strip()
    
    def _extract_json_aggressively(self, text: str) -> Optional[str]:
        """More aggressive JSON extraction"""
        # Look for anything that looks like our expected format
        # [["rule", ["predicate", ...], [...]]]
        
        # Remove all non-JSON content
        lines = text.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            if '[' in line or in_json:
                in_json = True
                json_lines.append(line)
                if line.count('[') <= line.count(']'):
                    in_json = False
        
        if json_lines:
            attempt = '\n'.join(json_lines)
            # Fix common issues
            attempt = re.sub(r',\s*]', ']', attempt)  # Remove trailing commas
            attempt = re.sub(r',\s*}', '}', attempt)
            
            # Ensure it starts and ends with brackets
            if not attempt.strip().startswith('['):
                attempt = '[' + attempt
            if not attempt.strip().endswith(']'):
                attempt = attempt + ']'
                
            return attempt
            
        return None


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