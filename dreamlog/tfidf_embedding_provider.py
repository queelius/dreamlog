"""
TF-IDF Embedding Provider for DreamLog

Provides a local, dependency-free embedding provider using TF-IDF.
No external API or server required - always available as fallback.
"""

from typing import List, Dict, Optional
import math
from collections import Counter
import re


class TfIdfEmbeddingProvider:
    """
    TF-IDF based embedding provider.

    Uses Term Frequency-Inverse Document Frequency to create embeddings.
    Completely local, no external dependencies.
    """

    def __init__(self, corpus: List[Dict[str, str]]):
        """
        Initialize TF-IDF provider with a corpus.

        Args:
            corpus: List of example dictionaries with 'prolog' and/or 'domain' keys
        """
        self.corpus = corpus
        self.vocabulary: Dict[str, int] = {}  # word -> index
        self.idf: Dict[str, float] = {}  # word -> IDF score
        self._dimension: Optional[int] = None

        # Build vocabulary and compute IDF
        self._fit_corpus()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Simple tokenization: lowercase, split on non-alphanumeric.
        """
        # Convert to lowercase
        text = text.lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\w+', text)
        return tokens

    def _fit_corpus(self):
        """Fit the TF-IDF model on the corpus"""
        # Collect all documents
        documents = []
        for example in self.corpus:
            # Combine domain and prolog text
            doc_text = f"{example.get('domain', '')} {example.get('prolog', '')}"
            documents.append(doc_text)

        # Build vocabulary and document frequency
        doc_frequency: Counter = Counter()  # word -> number of documents containing it

        all_tokens = set()
        for doc in documents:
            tokens = set(self._tokenize(doc))  # Use set to count each word once per doc
            all_tokens.update(tokens)
            doc_frequency.update(tokens)

        # Build vocabulary (sorted for consistency)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_tokens))}
        self._dimension = len(self.vocabulary)

        # Compute IDF: log(N / df(t))
        num_docs = len(documents)
        for word, df in doc_frequency.items():
            self.idf[word] = math.log(num_docs / df)

    def embed(self, text: str) -> List[float]:
        """
        Generate TF-IDF embedding for text.

        Args:
            text: Text to embed

        Returns:
            Dense vector of TF-IDF scores (dimension = vocabulary size)
        """
        # Tokenize
        tokens = self._tokenize(text)

        # Compute term frequency
        tf = Counter(tokens)
        total_terms = len(tokens) if tokens else 1

        # Create embedding vector
        embedding = [0.0] * self._dimension

        for word, count in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                # TF-IDF = (count / total_terms) * IDF
                tf_score = count / total_terms
                idf_score = self.idf.get(word, 0.0)
                embedding[idx] = tf_score * idf_score

        return embedding

    @property
    def dimension(self) -> Optional[int]:
        """Return embedding dimensionality"""
        return self._dimension

    def __repr__(self) -> str:
        return f"TfIdfEmbeddingProvider(vocabulary_size={self._dimension}, corpus_size={len(self.corpus)})"
