"""
Thin LLM client wrapper around OpenAI/Anthropic SDKs.

Covers OpenAI, Anthropic, Ollama (via OpenAI-compatible endpoint),
and any OpenAI-compatible API by setting base_url.

Usage:
    client = LLMClient(model="gpt-4o-mini")                          # OpenAI
    client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(base_url="http://localhost:11434/v1")          # Ollama
"""

import os
from typing import Optional


class LLMClient:
    def __init__(self, provider="openai", base_url=None, api_key=None,
                 api_key_env=None, model=None, temperature=0.1,
                 max_tokens=500, timeout=30):
        self.provider = provider
        self.model = model or self._default_model(provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.base_url = base_url

        resolved_key = api_key or self._resolve_key(provider, api_key_env)

        if provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(
                api_key=resolved_key,
                timeout=timeout,
            )
        else:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=base_url,
                api_key=resolved_key or "dummy",
                timeout=timeout,
            )

    @staticmethod
    def _default_model(provider: str) -> str:
        return {
            "anthropic": "claude-haiku-4-5-20251001",
            "openai": "gpt-4o-mini",
            "ollama": "phi4-mini:latest",
        }.get(provider, "gpt-4o-mini")

    @staticmethod
    def _resolve_key(provider: str, api_key_env: Optional[str] = None) -> Optional[str]:
        if api_key_env:
            return os.getenv(api_key_env)
        defaults = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = defaults.get(provider)
        return os.getenv(env_var) if env_var else None

    def complete(self, prompt, **kwargs):
        """Send a prompt to the LLM and return the response text."""
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=model, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            return resp.content[0].text
        else:
            resp = self._client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
