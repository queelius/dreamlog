"""
Unit tests for embedding providers
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from dreamlog.embedding_providers import EmbeddingProvider, OllamaEmbeddingProvider, TfIdfEmbeddingProvider


# Integration test configuration from environment
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Skip integration tests if OLLAMA_INTEGRATION_TESTS is not set
skip_integration = pytest.mark.skipif(
    os.environ.get("OLLAMA_INTEGRATION_TESTS") != "1",
    reason="Set OLLAMA_INTEGRATION_TESTS=1 to run integration tests"
)


class TestEmbeddingProviderProtocol:
    """Test that the EmbeddingProvider protocol is correctly defined"""

    def test_protocol_has_embed_method(self):
        """Protocol should require embed() method"""
        # Create a mock that implements the protocol
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        assert callable(mock_provider.embed)
        assert isinstance(mock_provider.dimension, int)

    def test_protocol_compliance(self):
        """OllamaEmbeddingProvider should comply with protocol"""
        provider = OllamaEmbeddingProvider(model="test-model")
        assert isinstance(provider, EmbeddingProvider)


class TestOllamaEmbeddingProvider:
    """Test OllamaEmbeddingProvider implementation"""

    def test_initialization(self):
        """Test provider initialization with different parameters"""
        # Default initialization
        provider = OllamaEmbeddingProvider()
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "nomic-embed-text"
        assert provider.timeout == 30
        assert provider._dimension is None

        # Custom initialization
        provider = OllamaEmbeddingProvider(
            base_url="http://example.com:11434",
            model="mxbai-embed-large",
            timeout=60
        )
        assert provider.base_url == "http://example.com:11434"
        assert provider.model == "mxbai-embed-large"
        assert provider.timeout == 60

    def test_base_url_normalization(self):
        """Test that trailing slash is removed from base_url"""
        provider = OllamaEmbeddingProvider(base_url="http://localhost:11434/")
        assert provider.base_url == "http://localhost:11434"

    @patch('dreamlog.embedding_providers.requests.post')
    def test_embed_success(self, mock_post):
        """Test successful embedding generation"""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        mock_post.return_value = mock_response

        provider = OllamaEmbeddingProvider()
        result = provider.embed("test text")

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "nomic-embed-text"
        assert call_args[1]['json']['prompt'] == "test text"
        assert call_args[1]['timeout'] == 30

        # Verify result
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert provider._dimension == 5  # Should cache dimension

    @patch('dreamlog.embedding_providers.requests.post')
    def test_embed_caches_dimension(self, mock_post):
        """Test that dimension is cached after first embedding"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3]
        }
        mock_post.return_value = mock_response

        provider = OllamaEmbeddingProvider()

        # First call
        provider.embed("first")
        assert provider._dimension == 3

        # Second call should use cached dimension
        provider.embed("second")
        assert provider._dimension == 3

    @patch('dreamlog.embedding_providers.requests.post')
    def test_embed_timeout_error(self, mock_post):
        """Test handling of timeout errors"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        provider = OllamaEmbeddingProvider(timeout=10)

        with pytest.raises(RuntimeError, match="timed out after 10s"):
            provider.embed("test")

    @patch('dreamlog.embedding_providers.requests.post')
    def test_embed_request_error(self, mock_post):
        """Test handling of request errors"""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

        provider = OllamaEmbeddingProvider()

        with pytest.raises(RuntimeError, match="request failed"):
            provider.embed("test")

    @patch('dreamlog.embedding_providers.requests.post')
    def test_embed_http_error(self, mock_post):
        """Test handling of HTTP errors"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")
        mock_post.return_value = mock_response

        provider = OllamaEmbeddingProvider()

        with pytest.raises(RuntimeError, match="Unexpected error"):
            provider.embed("test")

    @patch('dreamlog.embedding_providers.requests.post')
    def test_dimension_property_lazy_initialization(self, mock_post):
        """Test that dimension property triggers embed if not cached"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        mock_post.return_value = mock_response

        provider = OllamaEmbeddingProvider()
        assert provider._dimension is None

        # Accessing dimension should trigger embed("test")
        dim = provider.dimension
        assert dim == 768
        assert provider._dimension == 768

        # Verify embed was called with "test"
        mock_post.assert_called_once()
        assert mock_post.call_args[1]['json']['prompt'] == "test"

    @patch('dreamlog.embedding_providers.requests.post')
    def test_dimension_property_uses_cache(self, mock_post):
        """Test that dimension property uses cached value"""
        provider = OllamaEmbeddingProvider()
        provider._dimension = 1024  # Manually set cached dimension

        # Should not trigger API call
        dim = provider.dimension
        assert dim == 1024
        mock_post.assert_not_called()

    def test_repr(self):
        """Test string representation"""
        provider = OllamaEmbeddingProvider(
            base_url="http://example.com:11434",
            model="nomic-embed-text"
        )

        repr_str = repr(provider)
        assert "OllamaEmbeddingProvider" in repr_str
        assert "nomic-embed-text" in repr_str
        assert "example.com" in repr_str


class TestTfIdfEmbeddingProvider:
    """Test TfIdfEmbeddingProvider implementation"""

    def test_initialization(self):
        """Test provider initialization with corpus"""
        corpus = [
            {"domain": "family", "prolog": "parent(X, Y) :- father(X, Y)."},
            {"domain": "family", "prolog": "grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."},
            {"domain": "programming", "prolog": "compiles(X) :- language(X, java)."},
        ]

        provider = TfIdfEmbeddingProvider(corpus)

        assert provider.corpus == corpus
        assert provider.dimension > 0
        assert len(provider.vocabulary) > 0
        assert len(provider.idf) > 0

    def test_tokenization(self):
        """Test text tokenization"""
        corpus = [{"domain": "test", "prolog": "foo(X)"}]
        provider = TfIdfEmbeddingProvider(corpus)

        tokens = provider._tokenize("Hello World! Test-123")
        assert tokens == ["hello", "world", "test", "123"]

        tokens = provider._tokenize("parent(X, Y)")
        assert "parent" in tokens
        assert "x" in tokens
        assert "y" in tokens

    def test_vocabulary_building(self):
        """Test that vocabulary is built from corpus"""
        corpus = [
            {"domain": "family", "prolog": "parent(john, mary)"},
            {"domain": "family", "prolog": "parent(mary, alice)"},
        ]

        provider = TfIdfEmbeddingProvider(corpus)

        # Check vocabulary contains expected words
        vocab_words = set(provider.vocabulary.keys())
        assert "family" in vocab_words
        assert "parent" in vocab_words
        assert "john" in vocab_words
        assert "mary" in vocab_words
        assert "alice" in vocab_words

    def test_embed_returns_correct_dimension(self):
        """Test that embeddings have correct dimension"""
        corpus = [
            {"domain": "test", "prolog": "foo(X)"},
            {"domain": "test", "prolog": "bar(Y)"},
        ]

        provider = TfIdfEmbeddingProvider(corpus)
        embedding = provider.embed("foo bar")

        assert len(embedding) == provider.dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_sparse_vector(self):
        """Test that embeddings are sparse (mostly zeros)"""
        corpus = [
            {"domain": "test", "prolog": "word1 word2 word3"},
            {"domain": "test", "prolog": "word4 word5 word6"},
        ]

        provider = TfIdfEmbeddingProvider(corpus)
        embedding = provider.embed("word1")

        # Most values should be zero (sparse)
        zero_count = sum(1 for x in embedding if x == 0.0)
        non_zero_count = sum(1 for x in embedding if x != 0.0)

        assert zero_count > non_zero_count  # Sparse vector

    def test_embed_similarity(self):
        """Test that similar texts have similar embeddings"""
        corpus = [
            {"domain": "family", "prolog": "parent(X, Y) :- father(X, Y)."},
            {"domain": "family", "prolog": "grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."},
            {"domain": "programming", "prolog": "compile(X) :- java(X)."},
        ]

        provider = TfIdfEmbeddingProvider(corpus)

        # Use phrases that include words from corpus
        emb_parent = provider.embed("parent father family")
        emb_grandparent = provider.embed("grandparent parent family")
        emb_compile = provider.embed("compile java programming")

        import numpy as np

        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return np.dot(a, b) / (norm_a * norm_b)

        # parent and grandparent should be more similar than parent and compile
        sim_family = cosine_sim(emb_parent, emb_grandparent)
        sim_different = cosine_sim(emb_parent, emb_compile)

        # Both should share "family" domain words
        assert sim_family > 0  # Should have some similarity
        assert sim_family > sim_different  # Family examples more similar to each other

    def test_embed_unknown_words(self):
        """Test embedding text with words not in vocabulary"""
        corpus = [
            {"domain": "test", "prolog": "known_word"},
        ]

        provider = TfIdfEmbeddingProvider(corpus)

        # Embed text with unknown words
        embedding = provider.embed("completely unknown words")

        # Should return zero vector or very sparse vector
        assert len(embedding) == provider.dimension
        # Most values should be zero since words aren't in vocabulary
        assert embedding.count(0.0) > len(embedding) * 0.8

    def test_idf_calculation(self):
        """Test that IDF is calculated correctly"""
        corpus = [
            {"domain": "test", "prolog": "common common rare1"},
            {"domain": "test", "prolog": "common common rare2"},
            {"domain": "test", "prolog": "common rare3 rare4"},
        ]

        provider = TfIdfEmbeddingProvider(corpus)

        # "common" appears in all 3 docs, so low IDF
        # "rare1" appears in only 1 doc, so higher IDF
        idf_common = provider.idf.get("common", 0)
        idf_rare = provider.idf.get("rare1", 0)

        assert idf_rare > idf_common

    def test_protocol_compliance(self):
        """Test that TfIdfEmbeddingProvider complies with protocol"""
        corpus = [{"domain": "test", "prolog": "test"}]
        provider = TfIdfEmbeddingProvider(corpus)

        assert isinstance(provider, EmbeddingProvider)
        assert callable(provider.embed)
        assert isinstance(provider.dimension, int)

    def test_repr(self):
        """Test string representation"""
        corpus = [
            {"domain": "test", "prolog": "foo"},
            {"domain": "test", "prolog": "bar"},
        ]

        provider = TfIdfEmbeddingProvider(corpus)
        repr_str = repr(provider)

        assert "TfIdfEmbeddingProvider" in repr_str
        assert "vocabulary_size" in repr_str
        assert "corpus_size=2" in repr_str


@skip_integration
class TestOllamaEmbeddingProviderIntegration:
    """
    Integration tests with real Ollama server.

    To run these tests, set environment variables:
      export OLLAMA_INTEGRATION_TESTS=1
      export OLLAMA_BASE_URL=http://192.168.0.225:11434  # optional, defaults to localhost
      export OLLAMA_EMBEDDING_MODEL=nomic-embed-text     # optional
    """

    def test_real_ollama_embedding(self):
        """Test with real Ollama server"""
        provider = OllamaEmbeddingProvider(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL
        )

        embedding = provider.embed("hello world")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
        assert provider.dimension == len(embedding)

    def test_real_ollama_dimension(self):
        """Test dimension property with real server"""
        provider = OllamaEmbeddingProvider(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL
        )

        dim = provider.dimension
        assert isinstance(dim, int)
        assert dim > 0

    def test_real_ollama_multiple_embeds(self):
        """Test multiple embeddings with dimension caching"""
        provider = OllamaEmbeddingProvider(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL
        )

        emb1 = provider.embed("first text")
        emb2 = provider.embed("second text")
        emb3 = provider.embed("third text")

        # All should have same dimension
        assert len(emb1) == len(emb2) == len(emb3)
        assert provider.dimension == len(emb1)

    def test_real_ollama_semantic_similarity(self):
        """Test that similar texts have similar embeddings"""
        import numpy as np

        provider = OllamaEmbeddingProvider(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_EMBEDDING_MODEL
        )

        # Similar texts
        emb1 = provider.embed("dog")
        emb2 = provider.embed("puppy")
        # Different text
        emb3 = provider.embed("computer")

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # dog and puppy should be more similar than dog and computer
        sim_dog_puppy = cosine_similarity(emb1, emb2)
        sim_dog_computer = cosine_similarity(emb1, emb3)

        assert sim_dog_puppy > sim_dog_computer
