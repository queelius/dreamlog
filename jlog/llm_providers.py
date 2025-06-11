from typing import Dict, Any, Optional, Protocol
import json
import requests
from abc import ABC, abstractmethod
import re

class LLMProvider(Protocol):
    """Protocol for LLM providers"""
    
    def generate_text(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Generate text based on a prompt"""
        ...

class HTTPLLMProvider(ABC):
    """Base class for HTTP-based LLM providers"""
    
    def __init__(self, base_url: str, model: str, **default_params):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.default_params = default_params
    
    @abstractmethod
    def _prepare_request(self, prompt: str, max_tokens: int, **kwargs) -> Dict[str, Any]:
        """Prepare the request payload for the specific provider"""
        pass
    
    @abstractmethod
    def _extract_response(self, response_data: Dict[str, Any]) -> str:
        """Extract the generated text from the provider's response"""
        pass
    
    @abstractmethod
    def _get_endpoint(self) -> str:
        """Get the API endpoint for this provider"""
        pass
    
    def generate_text(self, prompt: str, max_tokens: int = 500, **kwargs) -> str:
        """Generate text using HTTP API"""
        try:
            # Merge parameters
            params = {**self.default_params, **kwargs}
            
            # Prepare request
            payload = self._prepare_request(prompt, max_tokens, **params)
            
            # Make request
            response = requests.post(
                f"{self.base_url}{self._get_endpoint()}",
                json=payload,
                timeout=60  # Increased timeout for LLM responses
            )
            response.raise_for_status()
            
            # Extract response
            return self._extract_response(response.json())
            
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            return "[]"  # Return empty JSON array as fallback
        except Exception as e:
            print(f"Error generating text: {e}")
            return "[]"

class OllamaProvider(HTTPLLMProvider):
    """Ollama LLM provider"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2", **kwargs):
        super().__init__(base_url, model, **kwargs)
    
    def _get_endpoint(self) -> str:
        return "/api/generate"
    
    def _prepare_request(self, prompt: str, max_tokens: int, **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 0.9),
                **{k: v for k, v in kwargs.items() if k in ["top_k", "repeat_penalty"]}
            }
        }
    
    def _extract_response(self, response_data: Dict[str, Any]) -> str:
        return response_data.get("response", "[]")

