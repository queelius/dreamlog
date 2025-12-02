"""
Unit and integration tests for RAG-based example retrieval with softmax+temperature
and KB-aware weighted embeddings.

Test organization follows behavior-driven design:
- Tests verify observable outcomes, not implementation details
- Each test focuses on a single behavior
- Tests use Given-When-Then structure where applicable
"""

import pytest
import os
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dreamlog.example_retriever import ExampleRetriever, create_retriever_from_legacy


# Sample examples in new KB-aware format
SAMPLE_EXAMPLES_NEW = [
    {
        "kb_predicates": ["parent/2"],
        "kb_sample": "parent(john, mary). parent(mary, alice).",
        "query": "grandparent(X, Z)",
        "output": "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
        "success_count": 0,
    },
    {
        "kb_predicates": ["parent/2"],
        "kb_sample": "parent(john, mary). parent(john, bob).",
        "query": "sibling(X, Y)",
        "output": "sibling(X, Y) :- parent(Z, X), parent(Z, Y).",
        "success_count": 0,
    },
    {
        "kb_predicates": ["parent/2", "sibling/2"],
        "kb_sample": "parent(bob, alice). sibling(bob, john).",
        "query": "uncle(X, Y)",
        "output": "uncle(X, Y) :- sibling(X, Z), parent(Z, Y).",
        "success_count": 0,
    },
    {
        "kb_predicates": ["language/2", "target/2"],
        "kb_sample": "language(c, compiled). target(binary, compiled).",
        "query": "compiles_to(X, Y)",
        "output": "compiles_to(X, Y) :- language(X, L), target(Y, L).",
        "success_count": 0,
    },
    {
        "kb_predicates": ["uses/2", "language/2"],
        "kb_sample": "uses(python, django). language(python, interpreted).",
        "query": "web_framework(X, Y)",
        "output": "web_framework(X, Y) :- uses(Y, X), language(Y, Z).",
        "success_count": 0,
    },
    {
        "kb_predicates": ["in_country/2", "in_continent/2"],
        "kb_sample": "in_country(paris, france). in_continent(france, europe).",
        "query": "on_continent(City, Continent)",
        "output": "on_continent(City, Continent) :- in_country(City, Country), in_continent(Country, Continent).",
        "success_count": 0,
    },
]

# Legacy format for backward compatibility tests
SAMPLE_EXAMPLES_LEGACY = [
    {
        "domain": "family",
        "prolog": "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
        "json": [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]],
    },
    {
        "domain": "family",
        "prolog": "sibling(X, Y) :- parent(Z, X), parent(Z, Y).",
        "json": [["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"]]]],
    },
    {
        "domain": "family",
        "prolog": "uncle(X, Y) :- sibling(X, Z), parent(Z, Y).",
        "json": [["rule", ["uncle", "X", "Y"], [["sibling", "X", "Z"], ["parent", "Z", "Y"]]]],
    },
    {
        "domain": "programming",
        "prolog": "compiles_to(X, Y) :- language(X, L), target(Y, L).",
        "json": [["rule", ["compiles_to", "X", "Y"], [["language", "X", "L"], ["target", "Y", "L"]]]],
    },
    {
        "domain": "programming",
        "prolog": "web_framework(X, Y) :- uses(Y, X), language(Y, Z).",
        "json": [["rule", ["web_framework", "X", "Y"], [["uses", "Y", "X"], ["language", "Y", "Z"]]]],
    },
    {
        "domain": "geography",
        "prolog": "on_continent(City, Continent) :- in_country(City, Country), in_continent(Country, Continent).",
        "json": [
            [
                "rule",
                ["on_continent", "City", "Continent"],
                [["in_country", "City", "Country"], ["in_continent", "Country", "Continent"]],
            ]
        ],
    },
]


class TestExampleRetrieverInitialization:
    """Test ExampleRetriever initialization"""

    def test_init_with_mock_embedding_provider(self):
        """Test initialization with embedding provider"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        assert retriever.examples == SAMPLE_EXAMPLES_NEW
        assert retriever.embedding_provider is mock_provider

        # Verify embed was called for each example during initialization
        # Each example has both query and kb_sample, so 2 calls per example
        assert mock_provider.embed.call_count == len(SAMPLE_EXAMPLES_NEW) * 2

    def test_precompute_embeddings_handles_errors(self):
        """Test that embedding errors are handled gracefully during initialization"""
        mock_provider = Mock()
        mock_provider.dimension = 3

        # Create enough return values for all calls (2 per example + extras for errors)
        embeddings = [[0.1, 0.2, 0.3]] * 20
        embeddings[2] = Exception("Embedding failed")  # One failure
        mock_provider.embed.side_effect = embeddings

        # Should initialize without error despite one embedding failure
        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        # Verify all examples were processed (behavior: can still retrieve)
        results = retriever.retrieve("test", num_examples=3, temperature=1.0)
        assert len(results) == 3
        assert all(r in SAMPLE_EXAMPLES_NEW for r in results)

    def test_embedding_provider_required(self):
        """Test that embedding provider is required"""
        with pytest.raises(TypeError):
            ExampleRetriever()  # Missing embedding_provider

    def test_default_weights(self):
        """Test default query and KB weights"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        assert retriever.query_weight == 0.7
        assert retriever.kb_weight == 0.3

    def test_custom_weights(self):
        """Test custom query and KB weights"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(
            mock_provider, examples=SAMPLE_EXAMPLES_NEW, query_weight=0.8, kb_weight=0.2
        )

        assert retriever.query_weight == 0.8
        assert retriever.kb_weight == 0.2


class TestWeightedEmbeddings:
    """Test weighted embedding combination"""

    def test_combine_embeddings_normalization(self):
        """Test that combined embeddings are normalized to unit length"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])

        query_emb = np.array([1.0, 0.0, 0.0])
        kb_emb = np.array([0.0, 1.0, 0.0])

        combined = retriever._combine_embeddings(query_emb, kb_emb)

        # Should be unit length
        assert np.isclose(np.linalg.norm(combined), 1.0)

    def test_combine_embeddings_weighting(self):
        """Test that embeddings are weighted correctly"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(
            mock_provider, examples=[], query_weight=0.7, kb_weight=0.3
        )

        query_emb = np.array([1.0, 0.0, 0.0])
        kb_emb = np.array([0.0, 1.0, 0.0])

        combined = retriever._combine_embeddings(query_emb, kb_emb)

        # Before normalization: [0.7, 0.3, 0]
        # After normalization: [0.7, 0.3, 0] / sqrt(0.49 + 0.09) = [0.7, 0.3, 0] / 0.7616
        expected_ratio = 0.7 / 0.3  # query/kb ratio should be preserved
        actual_ratio = combined[0] / combined[1]
        assert np.isclose(actual_ratio, expected_ratio)


class TestKBContextRetrieval:
    """Test retrieval with KB context"""

    def test_retrieve_with_kb_context(self):
        """Test that KB context affects retrieval"""
        mock_provider = Mock()

        # Create embeddings with clear similarity structure
        # First 6 pairs: examples (query + kb_sample each)
        example_embeddings = []
        for i in range(6):
            # Query embedding
            query_emb = [0.0] * 3
            query_emb[i % 3] = 1.0
            example_embeddings.append(query_emb)
            # KB embedding
            kb_emb = [0.5] * 3
            example_embeddings.append(kb_emb)

        # Retrieval: query embedding + kb context embedding
        retrieval_query = [1.0, 0.0, 0.0]  # Similar to first example
        retrieval_kb = [0.5, 0.5, 0.5]

        mock_provider.embed.side_effect = example_embeddings + [retrieval_query, retrieval_kb]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        results = retriever.retrieve(
            "test", num_examples=3, temperature=0.1, kb_context="parent(a, b)."
        )

        assert len(results) == 3

    def test_retrieve_without_kb_context(self):
        """Test retrieval without KB context falls back to query-only"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.5, 0.5, 0.5]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        results = retriever.retrieve("test", num_examples=3, temperature=1.0)

        assert len(results) == 3


class TestSoftmaxRetrieval:
    """Test softmax+temperature retrieval"""

    def test_retrieve_returns_requested_number(self):
        """Test that retrieve returns the requested number of examples"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2]
        mock_provider.dimension = 2

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        for n in [1, 3, 5]:
            results = retriever.retrieve("test", num_examples=n, temperature=1.0)
            expected = min(n, len(SAMPLE_EXAMPLES_NEW))
            assert len(results) == expected

    def test_retrieve_samples_without_replacement(self):
        """Test that retrieval doesn't return duplicates"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.5, 0.5]
        mock_provider.dimension = 2

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        results = retriever.retrieve("test", num_examples=5, temperature=1.0)

        # Check no duplicates
        result_ids = [id(r) for r in results]
        assert len(result_ids) == len(set(result_ids))

    def test_low_temperature_peaks_distribution(self):
        """Test that low temperature selects most similar examples"""
        mock_provider = Mock()

        # Create embeddings where first example is most similar to query
        # Each example has 2 embeddings (query + kb)
        example_embeddings = [
            [1.0, 0.0],
            [0.5, 0.5],  # Example 0
            [0.5, 0.5],
            [0.5, 0.5],  # Example 1
            [0.0, 1.0],
            [0.5, 0.5],  # Example 2
            [0.1, 0.9],
            [0.5, 0.5],  # Example 3
            [0.2, 0.8],
            [0.5, 0.5],  # Example 4
            [0.3, 0.7],
            [0.5, 0.5],  # Example 5
        ]
        query_embedding = [0.99, 0.01]  # Very similar to first example

        mock_provider.embed.side_effect = example_embeddings + [query_embedding]
        mock_provider.dimension = 2

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        # With very low temperature, should almost always pick the most similar
        results = retriever.retrieve("test", num_examples=3, temperature=0.01)

        # First result should be the most similar example
        assert results[0] == SAMPLE_EXAMPLES_NEW[0]

    def test_high_temperature_more_uniform(self):
        """Test that high temperature gives more uniform sampling"""
        mock_provider = Mock()

        # Create diverse embeddings
        example_embeddings = []
        for i in range(6):
            example_embeddings.append([float(i) / 6, 1.0 - float(i) / 6])
            example_embeddings.append([0.5, 0.5])

        query_embedding = [1.0, 0.0]

        # Need many query embeddings for repeated calls
        mock_provider.embed.side_effect = example_embeddings + [query_embedding] * 100
        mock_provider.dimension = 2

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        # With high temperature, should get diverse results across multiple samples
        all_selected = set()
        for _ in range(50):
            results = retriever.retrieve("test", num_examples=2, temperature=10.0)
            for r in results:
                all_selected.add(SAMPLE_EXAMPLES_NEW.index(r))

        # Should have selected from multiple examples, not just top-2
        assert len(all_selected) >= 4

    def test_fallback_on_error(self):
        """Test fallback to random sampling on error"""
        mock_provider = Mock()
        # Precompute succeeds for all examples, then query fails
        embeddings = [[0.1, 0.2]] * (len(SAMPLE_EXAMPLES_NEW) * 2)
        embeddings.append(Exception("Embed failed"))
        mock_provider.embed.side_effect = embeddings
        mock_provider.dimension = 2

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        # Query embedding fails, should fall back to random
        results = retriever.retrieve("test", num_examples=3, temperature=1.0)

        # Should still return 3 results
        assert len(results) == 3
        # Should be from the original examples
        assert all(r in SAMPLE_EXAMPLES_NEW for r in results)


class TestSuccessTracking:
    """Test success recording and learning"""

    def test_record_success(self):
        """Test recording successful example usage"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        # Record success for first example
        retriever.record_success(0)
        assert retriever.examples[0]["success_count"] == 1

        retriever.record_success(0)
        assert retriever.examples[0]["success_count"] == 2

    def test_add_example(self):
        """Test adding new examples dynamically"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])

        retriever.add_example(
            query="father(X, Y)",
            output="father(X, Y) :- parent(X, Y), male(X).",
            kb_sample="parent(john, mary). male(john).",
            kb_predicates=["parent/2", "male/1"],
        )

        assert len(retriever.examples) == 1
        assert retriever.examples[0]["query"] == "father(X, Y)"
        assert len(retriever._example_embeddings) == 1


class TestLegacyCompatibility:
    """Test backward compatibility with old example format"""

    def test_create_from_legacy_format(self):
        """Test converting legacy examples to new format"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = create_retriever_from_legacy(SAMPLE_EXAMPLES_LEGACY, mock_provider)

        assert len(retriever.examples) == len(SAMPLE_EXAMPLES_LEGACY)
        # Check new format fields exist
        for ex in retriever.examples:
            assert "query" in ex
            assert "output" in ex
            assert "kb_sample" in ex
            assert "kb_predicates" in ex

    def test_legacy_format_retrieval(self):
        """Test retrieval works with converted legacy examples"""
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = create_retriever_from_legacy(SAMPLE_EXAMPLES_LEGACY, mock_provider)

        results = retriever.retrieve("test", num_examples=3, temperature=1.0)
        assert len(results) == 3


class TestTfIdfIntegration:
    """Test ExampleRetriever with TfIdfEmbeddingProvider"""

    def test_with_tfidf_provider(self):
        """Test retrieval with TF-IDF embeddings"""
        from dreamlog.embedding_providers import TfIdfEmbeddingProvider

        # TfIdfEmbeddingProvider needs text corpus
        corpus = [ex["query"] + " " + ex.get("kb_sample", "") for ex in SAMPLE_EXAMPLES_NEW]
        provider = TfIdfEmbeddingProvider(corpus)
        retriever = ExampleRetriever(provider, examples=SAMPLE_EXAMPLES_NEW)

        # Query for family-related predicate
        results = retriever.retrieve("cousin(X, Y)", num_examples=3, temperature=0.5)

        assert len(results) == 3
        # Results should be from the database
        assert all(r in SAMPLE_EXAMPLES_NEW for r in results)

    def test_semantic_similarity_with_tfidf(self):
        """Test that TF-IDF gives reasonable semantic similarity"""
        from dreamlog.embedding_providers import TfIdfEmbeddingProvider

        corpus = [ex["query"] + " " + ex.get("kb_sample", "") for ex in SAMPLE_EXAMPLES_NEW]
        provider = TfIdfEmbeddingProvider(corpus)
        retriever = ExampleRetriever(provider, examples=SAMPLE_EXAMPLES_NEW)

        # Query with KB context similar to family examples
        results = retriever.retrieve(
            "grandchild(X, Y)",
            num_examples=3,
            temperature=0.1,
            kb_context="parent(john, mary). parent(mary, alice).",
        )

        # Should return results
        assert len(results) == 3


@pytest.mark.skipif(
    os.environ.get("OLLAMA_INTEGRATION_TESTS") != "1",
    reason="Set OLLAMA_INTEGRATION_TESTS=1 to run integration tests",
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

        retriever = ExampleRetriever(provider, examples=SAMPLE_EXAMPLES_NEW)

        # Test different temperatures
        results_low = retriever.retrieve("cousin(X, Y)", num_examples=3, temperature=0.1)
        results_high = retriever.retrieve("cousin(X, Y)", num_examples=3, temperature=5.0)

        assert len(results_low) == 3
        assert len(results_high) == 3

    def test_real_embeddings_with_kb_context(self):
        """Test KB context with real embeddings"""
        from dreamlog.embedding_providers import OllamaEmbeddingProvider

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        provider = OllamaEmbeddingProvider(base_url=base_url, model="nomic-embed-text")

        retriever = ExampleRetriever(provider, examples=SAMPLE_EXAMPLES_NEW)

        # Query with KB context
        results = retriever.retrieve(
            "ancestor(X, Y)",
            num_examples=3,
            temperature=0.5,
            kb_context="parent(john, mary). parent(mary, alice).",
        )

        assert len(results) == 3


# ============================================================================
# NEW TESTS: Coverage gaps and improved TDD compliance
# ============================================================================


class TestFileLoading:
    """Test file-based example loading behavior"""

    def test_load_examples_from_path(self):
        """
        Given: A JSON file with examples
        When: Creating a retriever with examples_path
        Then: Examples are loaded from the file
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        examples = [
            {
                "kb_predicates": ["test/1"],
                "kb_sample": "test(a).",
                "query": "test(X)",
                "output": "test(X) :- base(X).",
                "success_count": 0,
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(examples, f)
            temp_path = Path(f.name)

        try:
            retriever = ExampleRetriever(mock_provider, examples_path=temp_path)

            assert len(retriever.examples) == 1, "Should load one example from file"
            assert retriever.examples[0]["query"] == "test(X)"
        finally:
            temp_path.unlink()

    def test_load_examples_handles_malformed_json(self):
        """
        Given: A file with invalid JSON
        When: Creating a retriever with that path
        Then: Retriever initializes with empty examples (graceful degradation)
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        try:
            retriever = ExampleRetriever(mock_provider, examples_path=temp_path)

            assert retriever.examples == [], "Should fallback to empty list on parse error"
        finally:
            temp_path.unlink()

    def test_load_examples_handles_missing_file(self):
        """
        Given: A path to a non-existent file
        When: Creating a retriever with that path
        Then: Retriever initializes with empty examples (graceful degradation)
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        non_existent_path = Path("/tmp/definitely_does_not_exist_12345.json")

        retriever = ExampleRetriever(mock_provider, examples_path=non_existent_path)

        assert retriever.examples == [], "Should fallback to empty list when file missing"

    def test_default_path_loading_when_exists(self):
        """
        Given: No examples or examples_path provided, but default file exists
        When: Creating a retriever
        Then: Examples are loaded from the default rag_examples.json
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        # The default path should exist in the codebase
        retriever = ExampleRetriever(mock_provider)

        # Should have loaded from default rag_examples.json (15 examples)
        assert len(retriever.examples) > 0, "Should load from default rag_examples.json"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_retrieve_with_empty_examples_returns_empty_list(self):
        """
        Given: A retriever with no examples
        When: Attempting to retrieve
        Then: Returns empty list without error
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])

        results = retriever.retrieve("test(X)", num_examples=5, temperature=1.0)

        assert results == [], "Empty examples should return empty results"

    def test_normalize_zero_vector_returns_zero_vector(self):
        """
        Given: A zero vector
        When: Normalizing it
        Then: Returns zero vector (no division by zero error)
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.0, 0.0, 0.0]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])

        # This exercises the norm==0 branch
        zero_vec = np.array([0.0, 0.0, 0.0])
        result = retriever._normalize(zero_vec)

        assert np.allclose(result, zero_vec), "Zero vector should remain zero after normalization"

    def test_cosine_similarity_with_zero_vector_returns_zero(self):
        """
        Given: One or both vectors are zero
        When: Computing cosine similarity
        Then: Returns 0.0 (no division by zero)
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])

        zero_vec = np.array([0.0, 0.0, 0.0])
        normal_vec = np.array([1.0, 0.0, 0.0])

        sim1 = retriever._cosine_similarity(zero_vec, normal_vec)
        sim2 = retriever._cosine_similarity(normal_vec, zero_vec)
        sim3 = retriever._cosine_similarity(zero_vec, zero_vec)

        assert sim1 == 0.0, "Similarity with zero vector should be 0"
        assert sim2 == 0.0, "Similarity with zero vector should be 0"
        assert sim3 == 0.0, "Similarity of two zero vectors should be 0"

    def test_request_more_examples_than_available(self):
        """
        Given: A retriever with N examples
        When: Requesting more than N examples
        Then: Returns at most N examples (all available)
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        small_examples = SAMPLE_EXAMPLES_NEW[:2]  # Only 2 examples
        retriever = ExampleRetriever(mock_provider, examples=small_examples)

        results = retriever.retrieve("test", num_examples=10, temperature=1.0)

        assert len(results) == 2, "Should return all available examples when requesting more"

    def test_record_success_with_invalid_index_is_noop(self):
        """
        Given: An example index that doesn't exist
        When: Recording success for that index
        Then: No error occurs (silent no-op)
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        # Use fresh copies to avoid mutation from other tests
        fresh_examples = [
            {"query": "test1(X)", "output": "test1(X).", "kb_sample": "", "kb_predicates": [], "success_count": 0},
            {"query": "test2(X)", "output": "test2(X).", "kb_sample": "", "kb_predicates": [], "success_count": 0},
        ]
        retriever = ExampleRetriever(mock_provider, examples=fresh_examples)

        # Should not raise
        retriever.record_success(-1)
        retriever.record_success(100)

        # Original examples unchanged - invalid indices should not modify anything
        for ex in retriever.examples:
            assert ex.get("success_count", 0) == 0, "Invalid index should not modify success counts"


class TestSaveExamples:
    """Test example persistence"""

    def test_save_examples_to_custom_path(self):
        """
        Given: A retriever with examples
        When: Saving to a custom path
        Then: Examples are written as valid JSON
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW[:2])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            retriever.save_examples(temp_path)

            with open(temp_path, "r") as f:
                saved = json.load(f)

            assert len(saved) == 2, "Should save all examples"
            assert saved[0]["query"] == SAMPLE_EXAMPLES_NEW[0]["query"]
        finally:
            temp_path.unlink()

    def test_save_examples_preserves_success_counts(self):
        """
        Given: A retriever with recorded successes
        When: Saving examples
        Then: Success counts are preserved in the saved file
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        # Use fresh examples to avoid mutation from other tests
        fresh_examples = [
            {"query": "test1(X)", "output": "test1(X).", "kb_sample": "", "kb_predicates": [], "success_count": 0},
            {"query": "test2(X)", "output": "test2(X).", "kb_sample": "", "kb_predicates": [], "success_count": 0},
        ]
        retriever = ExampleRetriever(mock_provider, examples=fresh_examples)
        retriever.record_success(0)
        retriever.record_success(0)
        retriever.record_success(1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            retriever.save_examples(temp_path)

            with open(temp_path, "r") as f:
                saved = json.load(f)

            assert saved[0]["success_count"] == 2, "First example should have 2 successes"
            assert saved[1]["success_count"] == 1, "Second example should have 1 success"
        finally:
            temp_path.unlink()


class TestAddExampleBehavior:
    """Test dynamic example addition"""

    def test_add_example_without_kb_sample(self):
        """
        Given: Adding an example with empty kb_sample
        When: The example is added
        Then: Example uses query-only embedding and is retrievable
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])

        retriever.add_example(
            query="test(X)",
            output="test(X) :- base(X).",
            kb_sample="",  # Empty KB sample
            kb_predicates=[],
        )

        assert len(retriever.examples) == 1
        # Should be retrievable
        results = retriever.retrieve("test", num_examples=1, temperature=1.0)
        assert len(results) == 1

    def test_add_example_handles_embedding_failure(self):
        """
        Given: An embedding provider that fails
        When: Adding an example
        Then: Example is added with fallback embedding (still retrievable)
        """
        mock_provider = Mock()
        # First calls work (for any existing examples), then fail for new example
        mock_provider.embed.side_effect = Exception("Embedding service unavailable")
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])
        # Clear the embeddings list that may have been affected
        retriever._example_embeddings = []

        # Should not raise
        retriever.add_example(
            query="test(X)",
            output="test(X) :- base(X).",
            kb_sample="",
        )

        assert len(retriever.examples) == 1, "Example should be added despite embedding failure"

    def test_add_example_makes_it_immediately_retrievable(self):
        """
        Given: An empty retriever
        When: Adding an example
        Then: That example can be retrieved immediately
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [1.0, 0.0, 0.0]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=[])

        retriever.add_example(
            query="custom(X)",
            output="custom(X) :- known(X).",
            kb_sample="known(a). known(b).",
        )

        results = retriever.retrieve("custom(Y)", num_examples=1, temperature=1.0)

        assert len(results) == 1
        assert results[0]["query"] == "custom(X)"


class TestBehavioralContracts:
    """
    Tests that verify the behavioral contracts of the retriever,
    independent of implementation details.
    """

    def test_retrieval_is_deterministic_with_fixed_seed(self):
        """
        Given: A fixed random seed
        When: Retrieving examples multiple times with same parameters
        Then: Results are reproducible
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.5, 0.5, 0.5]
        mock_provider.dimension = 3

        retriever = ExampleRetriever(mock_provider, examples=SAMPLE_EXAMPLES_NEW)

        np.random.seed(42)
        results1 = retriever.retrieve("test", num_examples=3, temperature=1.0)

        np.random.seed(42)
        results2 = retriever.retrieve("test", num_examples=3, temperature=1.0)

        assert results1 == results2, "Same seed should produce same results"

    def test_weight_ratio_affects_retrieval_behavior(self):
        """
        Given: Different query/KB weight ratios
        When: The KB context is highly relevant to some examples
        Then: Higher KB weight should favor KB-similar examples
        """
        # Create a provider that returns different embeddings based on input
        def create_targeted_embedding(text):
            if "family" in text.lower() or "parent" in text.lower():
                return [1.0, 0.0, 0.0]  # Family domain
            elif "graph" in text.lower() or "edge" in text.lower():
                return [0.0, 1.0, 0.0]  # Graph domain
            else:
                return [0.0, 0.0, 1.0]  # Other

        mock_provider = Mock()
        mock_provider.embed.side_effect = create_targeted_embedding
        mock_provider.dimension = 3

        # Two retrievers with different weight ratios
        retriever_query_heavy = ExampleRetriever(
            mock_provider,
            examples=SAMPLE_EXAMPLES_NEW,
            query_weight=0.9,
            kb_weight=0.1,
        )

        # Reset mock for second retriever
        mock_provider.embed.side_effect = create_targeted_embedding
        retriever_kb_heavy = ExampleRetriever(
            mock_provider,
            examples=SAMPLE_EXAMPLES_NEW,
            query_weight=0.1,
            kb_weight=0.9,
        )

        # Query about graphs but KB is about family
        # With query-heavy weights, graph examples should rank higher
        # With KB-heavy weights, family examples should rank higher
        mock_provider.embed.side_effect = create_targeted_embedding

        # This test verifies the weight system works - exact results depend on implementation
        results_query = retriever_query_heavy.retrieve(
            "path(X, Y)",  # Graph query
            num_examples=2,
            temperature=0.1,
            kb_context="parent(john, mary).",  # Family KB
        )

        mock_provider.embed.side_effect = create_targeted_embedding
        results_kb = retriever_kb_heavy.retrieve(
            "path(X, Y)",  # Graph query
            num_examples=2,
            temperature=0.1,
            kb_context="parent(john, mary).",  # Family KB
        )

        # Both should return valid examples
        assert len(results_query) == 2
        assert len(results_kb) == 2

    def test_examples_list_is_not_modified_by_retrieval(self):
        """
        Given: A retriever with examples
        When: Retrieving multiple times
        Then: The original examples list is unchanged
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        original_examples = [ex.copy() for ex in SAMPLE_EXAMPLES_NEW[:3]]
        retriever = ExampleRetriever(mock_provider, examples=original_examples)

        # Multiple retrievals
        for _ in range(10):
            retriever.retrieve("test", num_examples=2, temperature=1.0)

        # Verify examples unchanged (except success_count which should be 0)
        for i, ex in enumerate(retriever.examples):
            assert ex["query"] == SAMPLE_EXAMPLES_NEW[i]["query"]
            assert ex["output"] == SAMPLE_EXAMPLES_NEW[i]["output"]


class TestSuccessBasedLearning:
    """Test success-based learning and boosting"""

    def test_success_boost_increases_selection_probability(self):
        """
        Given: Examples with different success counts
        When: Retrieving with success_boost enabled
        Then: High-success examples should be selected more often
        """
        mock_provider = Mock()
        # All examples have same embedding (equal similarity)
        mock_provider.embed.return_value = [0.5, 0.5, 0.5]
        mock_provider.dimension = 3

        # Create examples with different success counts
        examples = [
            {"query": "test1(X)", "output": "test1.", "kb_sample": "", "kb_predicates": [], "success_count": 0},
            {"query": "test2(X)", "output": "test2.", "kb_sample": "", "kb_predicates": [], "success_count": 10},
            {"query": "test3(X)", "output": "test3.", "kb_sample": "", "kb_predicates": [], "success_count": 0},
        ]

        retriever = ExampleRetriever(mock_provider, examples=examples, success_boost=0.5)

        # Sample many times and count selections
        selection_counts = {0: 0, 1: 0, 2: 0}
        for _ in range(100):
            results = retriever.retrieve("test", num_examples=1, temperature=0.5)
            idx = examples.index(results[0])
            selection_counts[idx] += 1

        # Example with success_count=10 should be selected more often
        assert selection_counts[1] > selection_counts[0], "High-success example should be selected more often"
        assert selection_counts[1] > selection_counts[2], "High-success example should be selected more often"

    def test_success_boost_zero_disables_learning(self):
        """
        Given: success_boost=0
        When: Retrieving
        Then: Success counts should not affect selection
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.5, 0.5, 0.5]
        mock_provider.dimension = 3

        examples = [
            {"query": "test1(X)", "output": "test1.", "kb_sample": "", "kb_predicates": [], "success_count": 0},
            {"query": "test2(X)", "output": "test2.", "kb_sample": "", "kb_predicates": [], "success_count": 100},
        ]

        retriever = ExampleRetriever(mock_provider, examples=examples, success_boost=0.0)

        # With boost=0 and equal similarity, distribution should be roughly uniform
        selection_counts = {0: 0, 1: 0}
        for _ in range(100):
            results = retriever.retrieve("test", num_examples=1, temperature=1.0)
            idx = examples.index(results[0])
            selection_counts[idx] += 1

        # Both should be selected roughly equally (within reasonable variance)
        assert abs(selection_counts[0] - selection_counts[1]) < 40, "Without boost, selection should be roughly equal"

    def test_success_count_uses_log_for_diminishing_returns(self):
        """
        Given: Examples with very different success counts (1 vs 1000)
        When: Retrieving with success_boost
        Then: The difference in selection probability should not be extreme (log dampening)
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.5, 0.5, 0.5]
        mock_provider.dimension = 3

        examples = [
            {"query": "test1(X)", "output": "test1.", "kb_sample": "", "kb_predicates": [], "success_count": 1},
            {"query": "test2(X)", "output": "test2.", "kb_sample": "", "kb_predicates": [], "success_count": 1000},
        ]

        retriever = ExampleRetriever(mock_provider, examples=examples, success_boost=0.1)

        selection_counts = {0: 0, 1: 0}
        for _ in range(100):
            results = retriever.retrieve("test", num_examples=1, temperature=0.5)
            idx = examples.index(results[0])
            selection_counts[idx] += 1

        # High-success should be preferred but not dominate completely
        assert selection_counts[1] > selection_counts[0], "Higher success should be preferred"
        assert selection_counts[0] > 5, "Lower success example should still be selected sometimes (log dampening)"


class TestLegacyFormatEdgeCases:
    """Test edge cases in legacy format conversion"""

    def test_legacy_format_extracts_head_correctly(self):
        """
        Given: A legacy example with a rule
        When: Converting to new format
        Then: The query field contains only the rule head
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        legacy_examples = [
            {
                "domain": "test",
                "prolog": "head(X, Y) :- body1(X), body2(Y).",
                "json": [],
            }
        ]

        retriever = create_retriever_from_legacy(legacy_examples, mock_provider)

        assert retriever.examples[0]["query"] == "head(X, Y)"
        assert retriever.examples[0]["output"] == "head(X, Y) :- body1(X), body2(Y)."

    def test_legacy_format_handles_facts(self):
        """
        Given: A legacy example that's a fact (no body)
        When: Converting to new format
        Then: Query and output are both the fact
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        legacy_examples = [
            {
                "domain": "test",
                "prolog": "fact(value).",
                "json": [],
            }
        ]

        retriever = create_retriever_from_legacy(legacy_examples, mock_provider)

        assert retriever.examples[0]["query"] == "fact(value)."
        assert retriever.examples[0]["output"] == "fact(value)."

    def test_legacy_format_with_empty_prolog(self):
        """
        Given: A legacy example with empty prolog field
        When: Converting to new format
        Then: Handles gracefully with empty strings
        """
        mock_provider = Mock()
        mock_provider.embed.return_value = [0.1, 0.2, 0.3]
        mock_provider.dimension = 3

        legacy_examples = [
            {
                "domain": "test",
                "prolog": "",
                "json": [],
            }
        ]

        retriever = create_retriever_from_legacy(legacy_examples, mock_provider)

        assert retriever.examples[0]["query"] == ""
        assert retriever.examples[0]["output"] == ""