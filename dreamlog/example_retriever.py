"""
RAG-based example retrieval for DreamLog

Uses embeddings with softmax+temperature sampling for semantic similarity.
"""

from typing import List, Dict, Any
import numpy as np


class ExampleRetriever:
    """
    Retrieves relevant examples using semantic similarity with softmax+temperature sampling.

    Temperature controls exploration vs exploitation:
    - Low temperature (0.1): Peaked distribution, select most similar
    - Medium temperature (1.0): Balanced sampling
    - High temperature (5.0): More uniform, diverse selection
    """

    def __init__(self, examples: List[Dict[str, Any]], embedding_provider):
        """
        Initialize the retriever with a database of examples.

        Args:
            examples: List of example dictionaries with 'domain', 'prolog', 'json' keys
            embedding_provider: EmbeddingProvider for semantic search (required)
        """
        self.examples = examples
        self.embedding_provider = embedding_provider
        self._example_embeddings = []

        # Precompute embeddings for all examples
        self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Pre-compute embeddings for all examples"""
        # Embed each example's domain + prolog representation
        texts = [f"{ex['domain']}: {ex['prolog']}" for ex in self.examples]

        for i, text in enumerate(texts):
            try:
                embedding = self.embedding_provider.embed(text)
                self._example_embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Failed to embed example '{text[:50]}...': {e}")
                # Use zero vector as fallback
                dim = self.embedding_provider.dimension or 768
                self._example_embeddings.append([0.0] * dim)

    def retrieve(self,
                 functor: str,
                 num_examples: int = 5,
                 temperature: float = 1.0) -> List[Dict[str, Any]]:
        """
        Retrieve examples using softmax+temperature sampling.

        Args:
            functor: The predicate functor to find examples for
            num_examples: Number of examples to retrieve
            temperature: Sampling temperature (lower = more peaked, higher = more uniform)
                        - 0.1: Nearly deterministic, top-k selection
                        - 1.0: Standard softmax (default)
                        - 5.0: Nearly uniform random

        Returns:
            List of example dictionaries sampled according to similarity scores
        """
        try:
            # Embed the query functor
            query_embedding = self.embedding_provider.embed(functor)

            # Compute cosine similarity with all examples
            similarities = []
            for ex_embedding in self._example_embeddings:
                similarity = self._cosine_similarity(query_embedding, ex_embedding)
                similarities.append(similarity)

            # Apply softmax with temperature
            probabilities = self._softmax(similarities, temperature)

            # Sample without replacement according to probabilities
            num_to_sample = min(num_examples, len(self.examples))
            indices = np.random.choice(
                len(self.examples),
                size=num_to_sample,
                replace=False,
                p=probabilities
            )

            return [self.examples[i] for i in indices]

        except Exception as e:
            print(f"Warning: Retrieval failed: {e}, falling back to random sampling")
            # Fallback to random sampling
            import random
            num_to_sample = min(num_examples, len(self.examples))
            return random.sample(self.examples, num_to_sample)

    def _softmax(self, scores: List[float], temperature: float) -> np.ndarray:
        """
        Apply softmax with temperature to scores.

        Args:
            scores: List of similarity scores
            temperature: Temperature parameter

        Returns:
            Probability distribution summing to 1.0
        """
        # Convert to numpy array
        scores_array = np.array(scores)

        # Apply temperature scaling
        scaled_scores = scores_array / temperature

        # Numerical stability: subtract max before exp
        scaled_scores = scaled_scores - np.max(scaled_scores)

        # Compute softmax
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        return probabilities

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors"""
        # Convert to numpy arrays if needed
        v1 = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
        v2 = np.array(vec2) if not isinstance(vec2, np.ndarray) else vec2

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
