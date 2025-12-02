"""
RAG-based example retrieval for DreamLog

Uses weighted embeddings combining query and KB context for semantic similarity.
The combined embedding is: w_query * embed(query) + w_kb * embed(kb), normalized to unit length.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import json
from pathlib import Path


class ExampleRetriever:
    """
    Retrieves relevant examples using KB-aware semantic similarity with softmax+temperature sampling.

    Examples include KB context, allowing retrieval to consider both:
    - Query similarity: How similar is the user's query to example queries?
    - KB similarity: How similar is the user's KB to example KBs?

    The combined embedding weights query more heavily (default 0.7 query, 0.3 KB).

    Temperature controls exploration vs exploitation:
    - Low temperature (0.1): Peaked distribution, select most similar
    - Medium temperature (1.0): Balanced sampling
    - High temperature (5.0): More uniform, diverse selection
    """

    def __init__(
        self,
        embedding_provider,
        examples: Optional[List[Dict[str, Any]]] = None,
        examples_path: Optional[Path] = None,
        query_weight: float = 0.7,
        kb_weight: float = 0.3,
        success_boost: float = 0.1,
    ):
        """
        Initialize the retriever with examples and embedding provider.

        Args:
            embedding_provider: EmbeddingProvider for semantic search (required)
            examples: List of example dictionaries (new format with kb_sample, query, output)
            examples_path: Path to JSON file with examples (alternative to examples list)
            query_weight: Weight for query embedding (default 0.7)
            kb_weight: Weight for KB embedding (default 0.3)
            success_boost: Weight for success count boost (default 0.1)
        """
        self.embedding_provider = embedding_provider
        self.query_weight = query_weight
        self.kb_weight = kb_weight
        self.success_boost = success_boost
        self._example_embeddings: List[np.ndarray] = []

        # Load examples
        if examples is not None:
            self.examples = examples
        elif examples_path is not None:
            self.examples = self._load_examples(examples_path)
        else:
            # Try default path
            default_path = Path(__file__).parent / "rag_examples.json"
            if default_path.exists():
                self.examples = self._load_examples(default_path)
            else:
                self.examples = []

        # Precompute embeddings for all examples
        if self.examples:
            self._precompute_embeddings()

    def _load_examples(self, path: Path) -> List[Dict[str, Any]]:
        """Load examples from JSON file"""
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load examples from {path}: {e}")
            return []

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _combine_embeddings(
        self, query_emb: np.ndarray, kb_emb: np.ndarray
    ) -> np.ndarray:
        """
        Combine query and KB embeddings with weights, normalized to unit length.

        combined = normalize(w_query * query_emb + w_kb * kb_emb)
        """
        combined = self.query_weight * query_emb + self.kb_weight * kb_emb
        return self._normalize(combined)

    def _precompute_embeddings(self):
        """Pre-compute weighted embeddings for all examples"""
        self._example_embeddings = []

        for ex in self.examples:
            try:
                # Get query text (new format uses 'query', old format uses 'prolog')
                query_text = ex.get("query", ex.get("prolog", ""))

                # Get KB text (new format uses 'kb_sample', old format has no KB)
                kb_text = ex.get("kb_sample", "")

                # Embed query and KB separately
                query_emb = np.array(self.embedding_provider.embed(query_text))

                if kb_text:
                    kb_emb = np.array(self.embedding_provider.embed(kb_text))
                    # Combine with weights
                    combined = self._combine_embeddings(query_emb, kb_emb)
                else:
                    # No KB context, just use normalized query embedding
                    combined = self._normalize(query_emb)

                self._example_embeddings.append(combined)

            except Exception as e:
                print(f"Warning: Failed to embed example: {e}")
                # Use zero vector as fallback
                dim = getattr(self.embedding_provider, "dimension", 768)
                self._example_embeddings.append(np.zeros(dim))

    def retrieve(
        self,
        query: str,
        num_examples: int = 5,
        temperature: float = 1.0,
        kb_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve examples using weighted KB-aware semantic similarity.

        Args:
            query: The predicate/query to find examples for (Prolog syntax)
            num_examples: Number of examples to retrieve
            temperature: Sampling temperature (lower = more peaked, higher = more uniform)
            kb_context: Current KB context as Prolog facts (optional but recommended)

        Returns:
            List of example dictionaries sampled according to similarity scores
        """
        if not self.examples or not self._example_embeddings:
            return []

        try:
            # Embed query
            query_emb = np.array(self.embedding_provider.embed(query))

            # Embed KB context if provided
            if kb_context:
                kb_emb = np.array(self.embedding_provider.embed(kb_context))
                search_emb = self._combine_embeddings(query_emb, kb_emb)
            else:
                search_emb = self._normalize(query_emb)

            # Compute cosine similarity with all examples, boosted by success count
            scores = []
            for i, ex_emb in enumerate(self._example_embeddings):
                similarity = self._cosine_similarity(search_emb, ex_emb)

                # Add success boost: log(1 + success_count) gives diminishing returns
                success_count = self.examples[i].get("success_count", 0)
                if success_count > 0 and self.success_boost > 0:
                    boost = self.success_boost * np.log1p(success_count)
                    similarity += boost

                scores.append(similarity)

            # Apply softmax with temperature
            probabilities = self._softmax(scores, temperature)

            # Sample without replacement according to probabilities
            num_to_sample = min(num_examples, len(self.examples))
            indices = np.random.choice(
                len(self.examples), size=num_to_sample, replace=False, p=probabilities
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
        scores_array = np.array(scores)

        # Apply temperature scaling
        scaled_scores = scores_array / temperature

        # Numerical stability: subtract max before exp
        scaled_scores = scaled_scores - np.max(scaled_scores)

        # Compute softmax
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        return probabilities

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def record_success(self, example_index: int):
        """
        Record that an example led to successful inference.

        This allows the system to learn which examples work best over time.
        """
        if 0 <= example_index < len(self.examples):
            self.examples[example_index]["success_count"] = (
                self.examples[example_index].get("success_count", 0) + 1
            )

    def save_examples(self, path: Optional[Path] = None):
        """Save examples (with updated success counts) to JSON file"""
        save_path = path or Path(__file__).parent / "rag_examples.json"
        with open(save_path, "w") as f:
            json.dump(self.examples, f, indent=2)

    def add_example(
        self,
        query: str,
        output: str,
        kb_sample: str = "",
        kb_predicates: Optional[List[str]] = None,
    ):
        """
        Add a new example to the database.

        Args:
            query: The query in Prolog syntax
            output: The generated output in Prolog syntax
            kb_sample: Sample facts from the KB context
            kb_predicates: List of predicates in the KB (e.g., ["parent/2", "male/1"])
        """
        example = {
            "kb_predicates": kb_predicates or [],
            "kb_sample": kb_sample,
            "query": query,
            "output": output,
            "success_count": 0,
        }
        self.examples.append(example)

        # Compute embedding for new example
        try:
            query_emb = np.array(self.embedding_provider.embed(query))
            if kb_sample:
                kb_emb = np.array(self.embedding_provider.embed(kb_sample))
                combined = self._combine_embeddings(query_emb, kb_emb)
            else:
                combined = self._normalize(query_emb)
            self._example_embeddings.append(combined)
        except Exception as e:
            print(f"Warning: Failed to embed new example: {e}")
            dim = getattr(self.embedding_provider, "dimension", 768)
            self._example_embeddings.append(np.zeros(dim))


# Backward compatibility: function to create retriever with old format examples
def create_retriever_from_legacy(
    examples: List[Dict[str, Any]], embedding_provider
) -> ExampleRetriever:
    """
    Create a retriever from legacy format examples (domain, prolog, json).

    Converts to new format with empty KB context.
    """
    converted = []
    for ex in examples:
        converted.append(
            {
                "kb_predicates": [],
                "kb_sample": "",
                "query": ex.get("prolog", "").split(":-")[0].strip(),  # Extract head
                "output": ex.get("prolog", ""),
                "success_count": 0,
            }
        )
    return ExampleRetriever(embedding_provider, examples=converted)
