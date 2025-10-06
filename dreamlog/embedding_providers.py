"""
Embedding Provider Abstractions for DreamLog

Provides a unified interface for different embedding services.
Similar architecture to LLM providers.
"""

from typing import List, Optional, Protocol, runtime_checkable
import requests

# Re-export TfIdfEmbeddingProvider for convenience
from .tfidf_embedding_provider import TfIdfEmbeddingProvider

__all__ = ['EmbeddingProvider', 'OllamaEmbeddingProvider', 'TfIdfEmbeddingProvider']


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol for embedding providers.

    All embedding providers must implement the embed() method.
    """

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for the given text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        ...

    @property
    def dimension(self) -> Optional[int]:
        """Return the dimensionality of embeddings from this provider, or None if unknown"""
        ...


class OllamaEmbeddingProvider:
    """
    Embedding provider using Ollama's embedding API.

    Compatible with models like:
    - nomic-embed-text (768 dims)
    - mxbai-embed-large (1024 dims)
    - snowflake-arctic-embed (1024 dims)
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "nomic-embed-text",
                 timeout: int = 30):
        """
        Initialize Ollama embedding provider.

        Args:
            base_url: Ollama server URL
            model: Embedding model name
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._dimension = None

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            embedding = result.get("embedding", [])

            # Cache dimension on first call
            if self._dimension is None and embedding:
                self._dimension = len(embedding)

            return embedding

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama embedding request timed out after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama embedding request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during embedding: {e}")

    @property
    def dimension(self) -> Optional[int]:
        """
        Return embedding dimensionality.

        If not cached, makes a test embed call to determine dimension.
        Subsequent calls return the cached value.
        """
        if self._dimension is None:
            self.embed("test")  # This sets self._dimension
        return self._dimension

    def __repr__(self) -> str:
        return f"OllamaEmbeddingProvider(model={self.model}, base_url={self.base_url})"
