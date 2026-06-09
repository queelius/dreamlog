"""Build an LLMClient pointed at the remote Ollama model.

The Ollama server exposes an OpenAI-compatible endpoint at /v1, which
dreamlog.llm_client.LLMClient already supports via base_url. The timeout is
generous because the GPU is shared with another session, so calls can be slow.
"""
from dreamlog.llm_client import LLMClient

OLLAMA_HOST = "192.168.0.204"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen2.5:3b"


def make_ollama_client(model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
                       port: int = OLLAMA_PORT, temperature: float = 0.3,
                       max_tokens: int = 800, timeout: int = 180) -> LLMClient:
    """LLMClient for the remote Ollama model. Only qwen2.5:3b by default, so the
    shared GPU keeps it warm (never request another model from this helper)."""
    return LLMClient(provider="ollama",
                     base_url=f"http://{host}:{port}/v1",
                     model=model, temperature=temperature,
                     max_tokens=max_tokens, timeout=timeout)
