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

        # Verify embed was called for each example during initialization
        assert mock_provider.embed.call_count == len(SAMPLE_EXAMPLES)

    def test_precompute_embeddings_handles_errors(self):
        """Test that embedding errors are handled gracefully during initialization"""
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

        # Should initialize without error despite one embedding failure
        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # Verify all examples were processed (behavior: can still retrieve)
        results = retriever.retrieve("test", num_examples=3, temperature=1.0)
        assert len(results) == 3
        assert all(r in SAMPLE_EXAMPLES for r in results)

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

    def test_temperature_affects_sampling_behavior(self):
        """Test that temperature parameter affects sampling distribution"""
        mock_provider = Mock()

        # Create embeddings where first example is most similar to query
        example_embeddings = [
            [1.0, 0.0],  # Most similar
            [0.5, 0.5],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
        query_embedding = [1.0, 0.0]  # Very similar to first

        mock_provider.embed.side_effect = example_embeddings + [query_embedding] * 200
        mock_provider.dimension = 2

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # Low temperature should heavily favor most similar (deterministic)
        low_temp_counts = {}
        for _ in range(100):
            results = retriever.retrieve("test", num_examples=1, temperature=0.01)
            idx = SAMPLE_EXAMPLES.index(results[0])
            low_temp_counts[idx] = low_temp_counts.get(idx, 0) + 1

        # Should almost always pick example 0 (most similar)
        assert low_temp_counts.get(0, 0) > 90

        # High temperature should be more uniform (stochastic)
        high_temp_counts = {}
        for _ in range(100):
            results = retriever.retrieve("test", num_examples=1, temperature=10.0)
            idx = SAMPLE_EXAMPLES.index(results[0])
            high_temp_counts[idx] = high_temp_counts.get(idx, 0) + 1

        # Should select from multiple examples
        assert len(high_temp_counts) >= 3

    def test_semantic_similarity_ranking(self):
        """Test that retrieval ranks by semantic similarity"""
        mock_provider = Mock()

        # Create embeddings with clear similarity structure
        example_embeddings = [
            [1.0, 0.0, 0.0],  # Orthogonal to others
            [0.0, 1.0, 0.0],  # Identical to query
            [0.0, 0.9, 0.1],  # Very similar to query
            [0.0, 0.5, 0.5],  # Somewhat similar
            [0.0, 0.1, 0.9],  # Less similar
            [0.0, 0.0, 1.0],  # Orthogonal to query
        ]
        query_embedding = [0.0, 1.0, 0.0]  # Should match example 1 best

        mock_provider.embed.side_effect = example_embeddings + [query_embedding]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(SAMPLE_EXAMPLES, mock_provider)

        # With very low temperature, should prefer most similar
        results = retriever.retrieve("test", num_examples=3, temperature=0.001)

        # First result should be the most similar (example 1)
        assert results[0] == SAMPLE_EXAMPLES[1]

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
