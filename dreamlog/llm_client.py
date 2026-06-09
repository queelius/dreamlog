"""
Thin LLM client wrapper around OpenAI/Anthropic SDKs.

Covers OpenAI, Anthropic, Ollama (via OpenAI-compatible endpoint),
and any OpenAI-compatible API by setting base_url.

Usage:
    client = LLMClient(model="gpt-4o-mini")                          # OpenAI
    client = LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
    client = LLMClient(base_url="http://localhost:11434/v1")          # Ollama

    # After calls, check usage:
    print(client.usage)  # {'calls': 5, 'input_tokens': 4200, 'output_tokens': 1500}
    print(client.estimated_cost())  # 0.0094
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class LLMUsage:
    """Cumulative token usage and call count."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, input_tokens: int, output_tokens: int):
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def estimated_cost(self, pricing: Optional[Dict[str, float]] = None) -> float:
        """Estimate cost in USD. Pricing is {input_per_m, output_per_m}."""
        if pricing is None:
            pricing = {"input_per_m": 0.80, "output_per_m": 4.00}
        return (self.input_tokens / 1_000_000 * pricing["input_per_m"]
                + self.output_tokens / 1_000_000 * pricing["output_per_m"])

    def __str__(self):
        cost = self.estimated_cost()
        return (f"{self.calls} calls, {self.input_tokens:,} in / "
                f"{self.output_tokens:,} out (${cost:.4f})")


# Per-model pricing (USD per million tokens)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input_per_m": 0.80, "output_per_m": 4.00},
    "claude-sonnet-4-5-20241022": {"input_per_m": 3.00, "output_per_m": 15.00},
    "claude-opus-4-6": {"input_per_m": 15.00, "output_per_m": 75.00},
    "claude-sonnet-4-6": {"input_per_m": 3.00, "output_per_m": 15.00},
    "gpt-4o-mini": {"input_per_m": 0.15, "output_per_m": 0.60},
    "gpt-4o": {"input_per_m": 2.50, "output_per_m": 10.00},
}


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
        self.usage = LLMUsage()

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
            self.usage.record(resp.usage.input_tokens, resp.usage.output_tokens)
            return resp.content[0].text
        else:
            resp = self._client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            if resp.usage:
                self.usage.record(resp.usage.prompt_tokens,
                                  resp.usage.completion_tokens)
            return resp.choices[0].message.content

    def estimated_cost(self) -> float:
        """Estimated cost in USD based on model pricing."""
        pricing = MODEL_PRICING.get(self.model)
        return self.usage.estimated_cost(pricing)
