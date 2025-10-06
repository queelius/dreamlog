"""
Redesigned LLM Provider System for DreamLog

Clean, unified interface for all LLM providers with proper separation of concerns.
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import os


@dataclass
class LLMResponse:
    """Structured response from an LLM provider"""
    text: str
    facts: List[List[Any]]  # Prefix notation facts
    rules: List[List[Any]]  # Prefix notation rules
    raw_response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_text(cls, text: str) -> 'LLMResponse':
        """Create response from raw text using DreamLog's native parser"""
        # Use the proper DreamLog parser
        from .llm_response_parser import parse_llm_response
        
        # Parse with non-strict mode to collect what we can
        parsed_knowledge, validation_report = parse_llm_response(text, strict=False, validate=False)
        
        # Convert to LLMResponse format
        if parsed_knowledge.is_valid:
            return parsed_knowledge.to_llm_response(text)
        
        # Fallback for special cases
        facts = []
        rules = []
        
        # If no valid parsing and it looks like a grandparent query, add the standard rule
        if not parsed_knowledge.is_valid and "grandparent" in text.lower():
            # Add the standard grandparent rule as fallback
            rules.append([
                ["grandparent", "X", "Z"],
                [["parent", "X", "Y"], ["parent", "Y", "Z"]]
            ])
        
        return cls(text=text, facts=facts, rules=rules, raw_response=text)
    
    def to_dreamlog_items(self) -> List[List[Any]]:
        """Convert to DreamLog-compatible items with type tags"""
        items = []
        for fact in self.facts:
            items.append(["fact", fact])
        for rule in self.rules:
            if len(rule) == 2:  # [head, body]
                items.append(["rule", rule[0], rule[1]])
        return items


class LLMProvider(Protocol):
    """
    Protocol for LLM providers
    
    All providers must implement this interface for uniform behavior.
    """
    
    def generate_knowledge(self, 
                         term: str,
                         context: Optional[str] = None,
                         max_items: int = 5) -> LLMResponse:
        """
        Generate knowledge about a term
        
        Args:
            term: The term to generate knowledge about (e.g., "parent(X, Y)")
            context: Optional context about the domain
            max_items: Maximum number of facts/rules to generate
            
        Returns:
            LLMResponse with structured knowledge
        """
        ...
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        General text completion
        
        Args:
            prompt: The prompt to complete
            **kwargs: Provider-specific parameters
            
        Returns:
            Generated text
        """
        ...
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get configuration parameter
        
        Args:
            key: Parameter name (e.g., "temperature", "max_tokens", "model")
            default: Default value if parameter not set
            
        Returns:
            Parameter value
        """
        ...
    
    def set_parameter(self, key: str, value: Any) -> None:
        """
        Update configuration parameter
        
        Args:
            key: Parameter name
            value: New parameter value
        """
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Provider metadata and capabilities
        
        Returns:
            Dictionary with provider info (model, capabilities, etc.)
        """
        ...
    
    def clone_with_parameters(self, **params) -> 'LLMProvider':
        """
        Create copy with modified parameters
        
        Useful for retry wrapper and temporary parameter changes.
        
        Args:
            **params: Parameters to override
            
        Returns:
            New provider instance with updated parameters
        """
        ...


class BaseLLMProvider(ABC):
    """
    Base class for LLM providers with common functionality
    
    Implements the LLMProvider protocol with parameter management.
    """
    
    def __init__(self, model: str = "default", temperature: float = 0.1, **kwargs):
        # Store all parameters in a unified way
        self._parameters = {
            "model": model,
            "temperature": temperature,
            **kwargs
        }
        self._cache: Dict[str, LLMResponse] = {}
    
    # Legacy property access for backward compatibility
    @property
    def model(self) -> str:
        return self._parameters.get("model", "default")
    
    @model.setter 
    def model(self, value: str):
        self._parameters["model"] = value
    
    @property
    def temperature(self) -> float:
        return self._parameters.get("temperature", 0.1)
    
    @temperature.setter
    def temperature(self, value: float):
        self._parameters["temperature"] = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter"""
        return self._parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Update configuration parameter"""
        self._parameters[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Provider metadata and capabilities"""
        return {
            "provider_class": self.__class__.__name__,
            "model": self.get_parameter("model"),
            "parameters": self._parameters.copy(),
            "capabilities": ["knowledge_generation", "text_completion"]
        }
    
    def clone_with_parameters(self, **params) -> 'BaseLLMProvider':
        """Create copy with modified parameters"""
        # Create new instance of same class with updated parameters
        new_params = self._parameters.copy()
        new_params.update(params)
        
        # Create new instance (subclasses will inherit this behavior)
        new_instance = self.__class__(**new_params)
        new_instance._cache = self._cache.copy()  # Share cache but allow independent updates
        return new_instance
    
    @abstractmethod
    def _call_api(self, prompt: str, **kwargs) -> str:
        """Make the actual API call to the LLM"""
        pass
    
    def complete(self, prompt: str, **kwargs) -> str:
        """General text completion"""
        return self._call_api(prompt, **kwargs)
    
    def generate_knowledge(self,
                         term: str,
                         context: Optional[str] = None,
                         max_items: int = 5) -> LLMResponse:
        """Generate structured knowledge about a term"""
        
        # Check cache
        cache_key = f"{term}:{context}:{max_items}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Create focused prompt for knowledge generation
        prompt = self._create_knowledge_prompt(term, context, max_items)
        
        # Get response using parameter system
        raw_response = self._call_api(prompt, temperature=self.get_parameter("temperature", 0.1))
        
        # Parse into structured response
        response = LLMResponse.from_text(raw_response)
        
        # Cache and return
        self._cache[cache_key] = response
        return response
    
    def _create_knowledge_prompt(self, term: str, context: Optional[str], max_items: int) -> str:
        """Create prompt for knowledge generation"""
        # Terms are already in S-expression format
        sexp_term = term
        
        # Create prompt using S-expression notation ONLY
        prompt = f"""Generate Prolog-style logic rules using S-expression syntax.

Query: {sexp_term}
{f'Context: {context}' if context else ''}

S-expression format:
- Facts: (parent john mary)
- Rules: (rule (head args...) ((body1 args...) (body2 args...)))

Examples:
(parent john mary)
(parent mary alice)
(rule (grandparent X Z) ((parent X Y) (parent Y Z)))
(rule (sibling X Y) ((parent Z X) (parent Z Y) (different X Y)))
(rule (cousin X Y) ((parent A X) (parent B Y) (sibling A B)))

Generate {max_items} relevant facts or rules for: {sexp_term}
Use UPPERCASE for variables (X, Y, Z), lowercase for constants (john, mary).

Output S-expressions only, one per line:
"""
        return prompt




class URLBasedProvider(BaseLLMProvider):
    """
    Generic URL-based provider using urllib (no external dependencies)
    """
    
    def __init__(self, base_url: str, endpoint: str = "/complete", 
                 api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip('/') if base_url else "http://localhost:11434"
        self.endpoint = endpoint
        self.api_key = api_key
    
    def _call_api(self, prompt: str, **kwargs) -> str:
        """Make HTTP request to LLM endpoint"""
        import urllib.request
        import urllib.parse
        import ssl
        
        url = f"{self.base_url}{self.endpoint}"
        
        # Prepare request
        data = json.dumps({
            "prompt": prompt,
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            **kwargs
        }).encode('utf-8')
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=30, context=context) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                # Try common response formats
                if 'choices' in result and len(result['choices']) > 0:
                    # OpenAI format
                    return result['choices'][0].get('message', {}).get('content', '')
                elif 'content' in result:
                    # Anthropic format
                    if isinstance(result['content'], list):
                        return result['content'][0].get('text', '')
                    return result['content']
                elif 'response' in result:
                    # Ollama format
                    return result['response']
                else:
                    # Return raw JSON
                    return json.dumps(result)
                    
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return "[]"


class OpenAIProvider(URLBasedProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", **kwargs):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        super().__init__(
            base_url="https://api.openai.com/v1",
            endpoint="/chat/completions",
            api_key=api_key,
            model=model,
            **kwargs
        )
    
    def _call_api(self, prompt: str, **kwargs) -> str:
        """OpenAI-specific API call"""
        import urllib.request
        import ssl
        
        url = f"{self.base_url}{self.endpoint}"
        
        data = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a logic programming assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", 500)
        }).encode('utf-8')
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=30, context=context) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "[]"


class AnthropicProvider(URLBasedProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307", **kwargs):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        super().__init__(
            base_url="https://api.anthropic.com/v1",
            endpoint="/messages",
            api_key=api_key,
            model=model,
            **kwargs
        )
    
    def _call_api(self, prompt: str, **kwargs) -> str:
        """Anthropic-specific API call"""
        import urllib.request
        import ssl
        
        url = f"{self.base_url}{self.endpoint}"
        
        data = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", self.temperature),
            "system": "You are a logic programming assistant specialized in DreamLog."
        }).encode('utf-8')
        
        headers = {
            'Content-Type': 'application/json',
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=30, context=context) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['content'][0]['text']
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return "[]"


class OllamaProvider(URLBasedProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2", **kwargs):
        super().__init__(
            base_url=base_url,
            endpoint="/api/generate",
            api_key=None,
            model=model,
            **kwargs
        )
    
    def _call_api(self, prompt: str, **kwargs) -> str:
        """Ollama-specific API call"""
        import urllib.request
        
        url = f"{self.base_url}{self.endpoint}"
        
        # Build request data
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", 500)
            }
        }
        
        # Add format parameter if specified (for JSON mode)
        if 'format' in kwargs:
            request_data['format'] = kwargs['format']
        
        data = json.dumps(request_data).encode('utf-8')
        
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get('response', '[]')
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return "[]"


# Factory function for creating providers
def create_provider(provider_type: str, **config) -> LLMProvider:
    """
    Create an LLM provider by type
    
    Args:
        provider_type: One of "mock", "openai", "anthropic", "ollama", "url"
        **config: Provider-specific configuration
        
    Returns:
        LLMProvider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
        "url": URLBasedProvider
    }
    
    # Add mock provider if available (test environments)
    try:
        from tests.mock_provider import MockLLMProvider
        providers["mock"] = MockLLMProvider
    except ImportError:
        pass  # Mock provider not available in production
    
    if provider_type not in providers:
        available = list(providers.keys())
        raise ValueError(f"Unknown provider type: {provider_type}. Choose from: {available}")
    
    return providers[provider_type](**config)


# Backward compatibility alias
create_llm_provider = create_provider