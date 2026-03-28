"""
Thin LLM client wrapper around OpenAI/Anthropic SDKs.

Covers OpenAI, Anthropic, Ollama (via OpenAI-compatible endpoint),
and any OpenAI-compatible API by setting base_url.

Usage:
    client = LLMClient(model="gpt-4o-mini")                          # OpenAI
    client = LLMClient(provider="anthropic", model="claude-3-haiku")  # Anthropic
    client = LLMClient(base_url="http://localhost:11434/v1")          # Ollama
"""


class LLMClient:
    def __init__(self, provider="openai", base_url=None, api_key=None,
                 model="gpt-4o-mini", temperature=0.1, max_tokens=500):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        if provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key or "dummy")

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
