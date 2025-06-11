from typing import Dict, Any, Optional
import os
import json
from .llm_providers import LLMProvider, OllamaProvider, MockLLMProvider

class LLMConfig:
    """Configuration manager for LLM providers"""
    
    @staticmethod
    def create_provider(provider_type: str, **config) -> LLMProvider:
        """Factory method to create LLM providers"""
        
        if provider_type == "mock":
            return MockLLMProvider(
                knowledge_domain=config.get("knowledge_domain", "family")
            )
        
        elif provider_type == "ollama":
            return OllamaProvider(
                base_url=config.get("base_url", "http://localhost:11434"),
                model=config.get("model", "llama2"),
                **{k: v for k, v in config.items() if k not in ["base_url", "model"]}
            )
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def from_file(config_path: str) -> LLMProvider:
        """Create provider from JSON config file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        provider_type = config.pop("type")
        return LLMConfig.create_provider(provider_type, **config)
    
    @staticmethod
    def from_env() -> LLMProvider:
        """Create provider from environment variables"""
        provider_type = os.getenv("JLOG_LLM_PROVIDER", "mock")
        
        config = {}
        if provider_type == "ollama":
            config = {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "model": os.getenv("OLLAMA_MODEL", "llama2"),
                "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
            }
        
        return LLMConfig.create_provider(provider_type, **config)