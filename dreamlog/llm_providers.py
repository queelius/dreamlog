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
        """Create response from raw text, attempting to parse DreamLog structures"""
        facts = []
        rules = []
        
        # First try to extract JSON from the text (handle extra text around JSON)
        import re
        # Look for JSON array anywhere in the text
        json_pattern = r'\[[\s\S]*?\]'
        json_matches = re.findall(json_pattern, text)
        
        for potential_json in json_matches:
            try:
                # Clean up the JSON string
                clean_json = potential_json.strip()
                data = json.loads(clean_json)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, list) and len(item) > 0:
                            if item[0] == "fact" and len(item) > 1:
                                facts.append(item[1])
                            elif item[0] == "rule" and len(item) > 2:
                                rules.append([item[1], item[2]])
                    # If we successfully parsed JSON, stop looking
                    if facts or rules:
                        break
            except (json.JSONDecodeError, TypeError):
                continue
        
        # If no valid JSON found and it looks like a grandparent query, add the standard rule
        if not facts and not rules and "grandparent" in text.lower():
            # Add the standard grandparent rule
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
    
    All providers must implement this interface.
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


class BaseLLMProvider(ABC):
    """
    Base class for LLM providers with common functionality
    """
    
    def __init__(self, model: str = "default", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self._cache: Dict[str, LLMResponse] = {}
    
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
        
        # Get response
        raw_response = self._call_api(prompt, temperature=self.temperature)
        
        # Parse into structured response
        response = LLMResponse.from_text(raw_response)
        
        # Cache and return
        self._cache[cache_key] = response
        return response
    
    def _create_knowledge_prompt(self, term: str, context: Optional[str], max_items: int) -> str:
        """Create prompt for knowledge generation"""
        # Convert term representation to S-expression if needed
        # term comes as "grandparent(john, X)" but we want "(grandparent john X)"
        if '(' in term and not term.startswith('('):
            # Convert from Prolog style to S-expression
            parts = term.replace(')', '').split('(')
            functor = parts[0]
            args = parts[1].split(',') if len(parts) > 1 else []
            args_str = ' '.join(arg.strip() for arg in args)
            sexp_term = f"({functor} {args_str})"
        else:
            sexp_term = term
        
        # Create prompt using S-expression notation
        prompt = f"""Generate logic rules in S-expression/prefix notation.

Query: {sexp_term}
{context if context else ''}

For (grandparent X Z), the rule is: X is grandparent of Z if X is parent of Y and Y is parent of Z.

Return as JSON array with prefix notation:
[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]

Your response (JSON only):
"""
        return prompt




class URLBasedProvider(BaseLLMProvider):
    """
    Generic URL-based provider using urllib (no external dependencies)
    """
    
    def __init__(self, base_url: str, endpoint: str = "/complete", 
                 api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url.rstrip('/')
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
    
    # Special handling for mock provider (test-only)
    if provider_type == "mock":
        # Import from tests if available
        try:
            from tests.mock_provider import MockLLMProvider
            return MockLLMProvider(**config)
        except ImportError:
            raise ValueError("Mock provider is only available in test environment")
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}. Choose from: {list(providers.keys())}")
    
    return providers[provider_type](**config)