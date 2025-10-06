"""
Unit and integration tests for RAG-based example retrieval with softmax+temperature
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, MagicMock
from dreamlog.example_retriever import ExampleRetriever


# Sample examples for testing
SAMPLE_EXAMPLES = [
    {
        "domain": "family",
        "prolog": "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
        "json": [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
    },
    {
        "domain": "family",
        "prolog": "sibling(X, Y) :- parent(Z, X), parent(Z, Y).",
        "json": [["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"]]]]
    },
    {
        "domain": "family",
        "prolog": "uncle(X, Y) :- sibling(X, Z), parent(Z, Y).",
        "json": [["rule", ["uncle", "X", "Y"], [["sibling", "X", "Z"], ["parent", "Z", "Y"]]]]
    },
    {
        "domain": "programming",
        "prolog": "compiles_to(X, Y) :- language(X, L), target(Y, L).",
        "json": [["rule", ["compiles_to", "X", "Y"], [["language", "X", "L"], ["target", "Y", "L"]]]]
    },
    {
        "domain": "programming",
        "prolog": "web_framework(X, Y) :- uses(Y, X), language(Y, Z).",
        "json": [["rule", ["web_framework", "X", "Y"], [["uses", "Y", "X"], ["language", "Y", "Z"]]]]
    },
    {
        "domain": "geography",
        "prolog": "on_continent(City, Continent) :- in_country(City, Country), in_continent(Country, Continent).",
        "json": [["rule", ["on_continent", "City", "Continent"], [["in_country", "City", "Country"], ["in_continent", "Country", "Continent"]]]]
    },
]


class TestExampleRetrieverInitialization:
    """Test ExampleRetriever initialization"""

    def test_init_with_mock_embedding_provider(self):
        """Test initialization with embedding provider"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        assert retriever.examples == SAMPLE_EXAMPLES
        assert retriever.embedding_provider is mock_provider
        assert len(retriever._example_embeddings) == len(SAMPLE_EXAMPLES)

        # Verify embed was called for each example
        assert mock_provider.embed.call_count == len(SAMPLE_EXAMPLES)

    def test_precompute_embeddings_handles_errors(self):
        """Test that embedding errors are handled gracefully"""
        mock_provider = Mock()
        mock_provider.dimension = 3

        # First call succeeds, second fails, third succeeds
        mock_provider.embed.side_effect = [
            [0.1, 0.2, 0.3],
            Exception("Embedding failed"),
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5],
        ]

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # Should have embeddings for all examples (failed one uses zero vector)
        assert len(retriever._example_embeddings) == len(SAMPLE_EXAMPLES)
        assert retriever._example_embeddings[0] == [0.1, 0.2, 0.3]
        assert retriever._example_embeddings[1] == [0.0, 0.0, 0.0]  # Zero vector fallback
        assert retriever._example_embeddings[2] == [0.4, 0.5, 0.6]

    def test_embedding_provider_required(self):
        """Test that embedding provider is required (not optional)"""
        # ExampleRetriever now requires an embedding provider
        # This test verifies the API change
        with pytest.raises(TypeError):
            ExampleRetriever(SAMPLE_EXAMPLES)  # Missing embedding_provider


class TestSoftmaxRetrieval:
    """Test softmax+temperature retrieval"""

    def test_retrieve_returns_requested_number(self):
        """Test that retrieve returns the requested number of examples"""
        mock_provider = Mock()
        mock_provider.embed.side_effect = [[0.1, 0.2]] * (len(SAMPLE_EXAMPLES) + 1)
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        for n in [1, 3, 5]:
            results = retriever.retrieve("test", num_examples=n, temperature=1.0)
            expected = min(n, len(SAMPLE_EXAMPLES))
            assert len(results) == expected

    def test_retrieve_samples_without_replacement(self):
        """Test that retrieval doesn't return duplicates"""
        mock_provider = Mock()
        # All examples get same embedding (uniform distribution)
        mock_provider.embed.side_effect = [[0.5, 0.5]] * (len(SAMPLE_EXAMPLES) + 1)
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        results = retriever.retrieve("test", num_examples=5, temperature=1.0)

        # Check no duplicates
        result_ids = [id(r) for r in results]
        assert len(result_ids) == len(set(result_ids))

    def test_low_temperature_peaks_distribution(self):
        """Test that low temperature selects most similar examples"""
        mock_provider = Mock()

        # Create embeddings where first example is most similar to query
        example_embeddings = [
            [1.0, 0.0],  # Very similar to query
            [0.5, 0.5],  # Less similar
            [0.0, 1.0],  # Not similar
            [0.1, 0.9],  # Not similar
            [0.2, 0.8],  # Not similar
            [0.3, 0.7],  # Not similar
        ]
        query_embedding = [0.99, 0.01]  # Very similar to first example

        mock_provider.embed.side_effect = example_embeddings + [query_embedding]
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # With very low temperature, should almost always pick the most similar
        results = retriever.retrieve("test", num_examples=3, temperature=0.01)

        # First result should be the most similar example
        assert results[0] == SAMPLE_EXAMPLES[0]

    def test_high_temperature_more_uniform(self):
        """Test that high temperature gives more uniform sampling"""
        mock_provider = Mock()

        # Create diverse embeddings
        example_embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.3, 0.7],
        ]
        query_embedding = [1.0, 0.0]

        mock_provider.embed.side_effect = example_embeddings + [query_embedding] * 100
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # With high temperature, should get diverse results across multiple samples
        all_selected = set()
        for _ in range(50):
            results = retriever.retrieve("test", num_examples=2, temperature=10.0)
            for r in results:
                all_selected.add(SAMPLE_EXAMPLES.index(r))

        # Should have selected from multiple examples, not just top-2
        assert len(all_selected) >= 4  # At least 4 different examples selected

    def test_softmax_computation(self):
        """Test softmax with temperature computation"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.5, 0.5]
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # Test softmax function
        scores = [1.0, 2.0, 3.0]

        # Temperature = 1.0 (standard softmax)
        probs = retriever._softmax(scores, temperature=1.0)
        assert abs(sum(probs) - 1.0) < 1e-6  # Probabilities sum to 1
        assert probs[2] > probs[1] > probs[0]  # Highest score gets highest prob

        # Temperature → 0 (approaches one-hot)
        probs_low = retriever._softmax(scores, temperature=0.1)
        assert probs_low[2] > 0.9  # Almost all weight on highest score

        # Temperature → ∞ (approaches uniform)
        probs_high = retriever._softmax(scores, temperature=10.0)
        assert abs(probs_high[0] - probs_high[2]) < 0.1  # Nearly uniform

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.5, 0.5]
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # Identical vectors
        sim = retriever._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

        # Orthogonal vectors
        sim = retriever._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert abs(sim - 0.0) < 1e-6

        # Opposite vectors
        sim = retriever._cosine_similarity([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-6

        # Zero vector
        sim = retriever._cosine_similarity([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 0.0

    def test_fallback_on_error(self):
        """Test fallback to random sampling on error"""
        mock_provider = Mock()
        # Precompute succeeds
        mock_provider.embed.side_effect = [[0.1, 0.2]] * len(SAMPLE_EXAMPLES) + [Exception("Embed failed")]
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # Query embedding fails, should fall back to random
        results = retriever.retrieve("test", num_examples=3, temperature=1.0)

        # Should still return 3 results
        assert len(results) == 3
        # Should be from the original examples
        assert all(r in SAMPLE_EXAMPLES for r in results)


class TestTfIdfIntegration:
    """Test ExampleRetriever with TfIdfEmbeddingProvider"""

    def test_with_tfidf_provider(self):
        """Test retrieval with TF-IDF embeddings"""
        from dreamlog.embedding_providers import TfIdfEmbeddingProvider

        provider = TfIdfEmbeddingProvider(SAMPLE_EXAMPLES)
        retriever = ExampleRetriever(SAMPLE_EXAMPLES, provider)

        # Query for family-related predicate
        results = retriever.retrieve("cousin", num_examples=3, temperature=0.5)

        assert len(results) == 3
        # Results should be from the database
        assert all(r in SAMPLE_EXAMPLES for r in results)

    def test_semantic_similarity_with_tfidf(self):
        """Test that TF-IDF gives reasonable semantic similarity"""
        from dreamlog.embedding_providers import TfIdfEmbeddingProvider

        provider = TfIdfEmbeddingProvider(SAMPLE_EXAMPLES)
        retriever = ExampleRetriever(SAMPLE_EXAMPLES, provider)

        # With low temperature, family queries should prefer family examples
        family_results = retriever.retrieve("grandchild", num_examples=3, temperature=0.1)

        # Count how many are from family domain
        family_count = sum(1 for r in family_results if r["domain"] == "family")

        # Should have at least some family examples (TF-IDF isn't perfect but should help)
        assert family_count >= 1


@pytest.mark.skipif(
    os.environ.get("OLLAMA_INTEGRATION_TESTS") != "1",
    reason="Set OLLAMA_INTEGRATION_TESTS=1 to run integration tests"
)
class TestExampleRetrieverIntegration:
    """
    Integration tests with real embedding provider.

    To run: export OLLAMA_INTEGRATION_TESTS=1
    """

    def test_real_embeddings_with_temperature(self):
        """Test temperature parameter with real embeddings"""
        from dreamlog.embedding_providers import OllamaEmbeddingProvider

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        provider = OllamaEmbeddingProvider(base_url=base_url, model="nomic-embed-text")

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, provider)

        # Test different temperatures
        results_low = retriever.retrieve("cousin", num_examples=3, temperature=0.1)
        results_high = retriever.retrieve("cousin", num_examples=3, temperature=5.0)

        assert len(results_low) == 3
        assert len(results_high) == 3

        # Low temperature should be more consistent across runs
        # High temperature should be more diverse

    def test_real_embeddings_full_database(self):
        """Test with full RULE_EXAMPLES database"""
        from dreamlog.embedding_providers import OllamaEmbeddingProvider
        from dreamlog.prompt_template_system import RULE_EXAMPLES

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        provider = OllamaEmbeddingProvider(base_url=base_url, model="nomic-embed-text")

        retriever = ExampleRetriever(RULE_EXAMPLES, provider)

        # Query for different domains
        test_queries = ["ancestor", "web_framework", "nearby"]

        for query in test_queries:
            results = retriever.retrieve(query, num_examples=5, temperature=1.0)
            assert len(results) == 5
            # Verify all results are from the database
            assert all(r in RULE_EXAMPLES for r in results)
