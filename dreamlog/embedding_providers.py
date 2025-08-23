"""
Embedding Providers for DreamLog RAG System

Supports OpenAI and Ollama embedding APIs with optional caching wrapper.
The caching wrapper provides separation of concerns - any provider can be
wrapped with caching functionality.
"""

import json
import urllib.request
import ssl
import hashlib
from typing import List, Dict, Optional, Protocol
from abc import ABC, abstractmethod
import os


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers"""
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        ...


class OpenAIEmbeddingProvider:
    """OpenAI embedding provider"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-3-small",
                 base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for embeddings")
        self.model = model
        self.base_url = base_url.rstrip('/')
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed using OpenAI API"""
        url = f"{self.base_url}/embeddings"
        
        data = json.dumps({
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }).encode('utf-8')
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=30, context=context) as response:
                result = json.loads(response.read().decode('utf-8'))
                # Extract embeddings from response
                embeddings = [item['embedding'] for item in result['data']]
                return embeddings
        except Exception as e:
            print(f"Error calling OpenAI embeddings API: {e}")
            # Return zero vectors as fallback
            dim = 1536 if "small" in self.model or "ada" in self.model else 3072
            return [[0.0] * dim for _ in texts]


class NGramEmbeddingProvider:
    """
    N-gram based embedding provider for local/offline use.
    Uses character n-grams, word n-grams, and TF-IDF-like weighting.
    No external dependencies required.
    """
    
    def __init__(self,
                 char_ngram_range: tuple = (2, 4),
                 word_ngram_range: tuple = (1, 2),
                 vector_size: int = 512,
                 use_idf: bool = True):
        """
        Initialize n-gram embedding provider.
        
        Args:
            char_ngram_range: Range of character n-gram sizes (min, max)
            word_ngram_range: Range of word n-gram sizes (min, max)
            vector_size: Size of output embedding vector
            use_idf: Whether to use IDF weighting (requires building vocabulary)
        """
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.vector_size = vector_size
        self.use_idf = use_idf
        self.vocabulary = {}  # Feature -> IDF score
        self.doc_count = 0
    
    def _extract_char_ngrams(self, text: str) -> List[str]:
        """Extract character n-grams from text"""
        text = text.lower()
        ngrams = []
        for n in range(self.char_ngram_range[0], self.char_ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(f"char_{n}_{text[i:i+n]}")
        return ngrams
    
    def _extract_word_ngrams(self, text: str) -> List[str]:
        """Extract word n-grams from text"""
        import re
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        ngrams = []
        for n in range(self.word_ngram_range[0], self.word_ngram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                ngrams.append(f"word_{n}_{ngram}")
        return ngrams
    
    def _extract_logic_features(self, text: str) -> List[str]:
        """Extract logic programming specific features"""
        import re
        features = []
        
        # Extract predicates
        predicates = re.findall(r'\b(\w+)(?=\s*\(|\s+[A-Z]\b)', text)
        for pred in predicates:
            features.append(f"pred_{pred.lower()}")
        
        # Extract variables (single uppercase letters)
        variables = re.findall(r'\b([A-Z])\b', text)
        features.append(f"var_count_{len(set(variables))}")
        
        # Detect rule vs fact
        if ":-" in text or "rule" in text.lower():
            features.append("type_rule")
        elif "fact" in text.lower() or not any(c.isupper() for c in text):
            features.append("type_fact")
        
        # Detect recursion indicators
        if "ancestor" in text.lower() or "descendant" in text.lower():
            features.append("recursive_likely")
        
        # Detect transitivity
        if "grandparent" in text.lower() or "connected" in text.lower():
            features.append("transitive_likely")
        
        return features
    
    def _hash_features_to_vector(self, features: List[str], weights: Dict[str, float] = None) -> List[float]:
        """Hash features to fixed-size vector using feature hashing trick"""
        vector = [0.0] * self.vector_size
        
        for feature in features:
            # Get weight for this feature
            weight = 1.0
            if weights and feature in weights:
                weight = weights[feature]
            elif self.use_idf and feature in self.vocabulary:
                weight = self.vocabulary[feature]
            
            # Hash to multiple positions for robustness
            for i in range(3):  # Use 3 hash functions
                seed = f"{feature}_{i}"
                index = hash(seed) % self.vector_size
                # Use different signs for different hashes to reduce collisions
                sign = 1 if i == 0 else (-1 if i == 1 else 0.5)
                vector[index] += sign * weight
        
        # L2 normalization
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def update_vocabulary(self, texts: List[str]):
        """Update IDF vocabulary with new texts (optional, for IDF weighting)"""
        if not self.use_idf:
            return
        
        import math
        
        for text in texts:
            self.doc_count += 1
            features = set()
            features.update(self._extract_char_ngrams(text))
            features.update(self._extract_word_ngrams(text))
            features.update(self._extract_logic_features(text))
            
            for feature in features:
                if feature not in self.vocabulary:
                    self.vocabulary[feature] = 0
                self.vocabulary[feature] += 1
        
        # Convert counts to IDF scores
        for feature in self.vocabulary:
            doc_freq = self.vocabulary[feature]
            idf = math.log(self.doc_count / (1 + doc_freq))
            self.vocabulary[feature] = idf
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using n-grams and feature hashing"""
        # Extract all features
        features = []
        features.extend(self._extract_char_ngrams(text))
        features.extend(self._extract_word_ngrams(text))
        features.extend(self._extract_logic_features(text))
        
        # Convert to vector
        return self._hash_features_to_vector(features)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts"""
        # Optionally update vocabulary for IDF
        if self.use_idf and not self.vocabulary:
            self.update_vocabulary(texts)
        
        return [self.embed(text) for text in texts]


class OllamaEmbeddingProvider:
    """Ollama embedding provider for local models"""
    
    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "nomic-embed-text"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._dimension = None  # Will be determined from first response
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using Ollama"""
        url = f"{self.base_url}/api/embeddings"
        
        data = json.dumps({
            "model": self.model,
            "prompt": text
        }).encode('utf-8')
        
        headers = {'Content-Type': 'application/json'}
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                embedding = result.get('embedding', [])
                
                # Store dimension for fallback
                if embedding and self._dimension is None:
                    self._dimension = len(embedding)
                
                return embedding
        except Exception as e:
            print(f"Error calling Ollama embeddings: {e}")
            # Return zero vector as fallback
            dim = self._dimension or 768  # Common default
            return [0.0] * dim
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed using Ollama (processes one at a time)"""
        # Ollama doesn't support batch embedding in the same API call
        # So we process sequentially (could be parallelized with threads)
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings


class CachedEmbeddingProvider:
    """
    Wrapper that adds LRU caching to any embedding provider.
    This provides separation of concerns - the underlying provider handles
    the API calls, while this wrapper handles caching logic.
    """
    
    def __init__(self, provider: EmbeddingProvider, cache_size: int = 1000):
        self.provider = provider
        self.cache_size = cache_size
        self.cache: Dict[str, List[float]] = {}
        self.cache_order: List[str] = []  # Track order for LRU
        self._hits = 0
        self._misses = 0
    
    def embed(self, text: str) -> List[float]:
        """Generate or retrieve cached embedding"""
        if text in self.cache:
            # Cache hit
            self._hits += 1
            # Move to end (most recently used)
            self.cache_order.remove(text)
            self.cache_order.append(text)
            return self.cache[text]
        
        # Cache miss
        self._misses += 1
        
        # Compute embedding using underlying provider
        embedding = self.provider.embed(text)
        
        # Add to cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            # Remove least recently used
            lru_text = self.cache_order.pop(0)
            del self.cache[lru_text]
        
        self.cache[text] = embedding
        self.cache_order.append(text)
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed with caching"""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if text in self.cache:
                # Cache hit
                self._hits += 1
                # Move to end (most recently used)
                self.cache_order.remove(text)
                self.cache_order.append(text)
                results.append(self.cache[text])
            else:
                # Cache miss
                self._misses += 1
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute uncached embeddings
        if uncached_texts:
            new_embeddings = self.provider.embed_batch(uncached_texts)
            
            # Fill in results and update cache
            for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
                results[idx] = embedding
                
                # Add to cache with LRU eviction
                if len(self.cache) >= self.cache_size:
                    lru_text = self.cache_order.pop(0)
                    del self.cache[lru_text]
                
                self.cache[text] = embedding
                self.cache_order.append(text)
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        self.cache_order.clear()
        self._hits = 0
        self._misses = 0
    
    def cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.cache_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }


def create_embedding_provider(provider_type: str, **config) -> EmbeddingProvider:
    """
    Factory function to create embedding providers.
    
    Args:
        provider_type: One of "openai", "ollama", or "ngram"
        **config: Provider-specific configuration
            Common:
                - cache: Whether to wrap with caching (default: True)
                - cache_size: LRU cache size (default: 1000)
            For OpenAI:
                - api_key: API key (or uses OPENAI_API_KEY env var)
                - model: Model name (default: text-embedding-3-small)
                - base_url: API base URL (default: https://api.openai.com/v1)
            For Ollama:
                - base_url: Ollama server URL (default: http://localhost:11434)
                - model: Model name (default: nomic-embed-text)
            For N-gram:
                - char_ngram_range: Character n-gram range (default: (2, 4))
                - word_ngram_range: Word n-gram range (default: (1, 2))
                - vector_size: Embedding vector size (default: 512)
                - use_idf: Whether to use IDF weighting (default: True)
        
    Returns:
        EmbeddingProvider instance (possibly wrapped with caching)
    """
    # Create base provider
    if provider_type == "openai":
        provider = OpenAIEmbeddingProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "text-embedding-3-small"),
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
    elif provider_type == "ollama":
        provider = OllamaEmbeddingProvider(
            base_url=config.get("base_url", "http://localhost:11434"),
            model=config.get("model", "nomic-embed-text")
        )
    elif provider_type == "ngram":
        provider = NGramEmbeddingProvider(
            char_ngram_range=config.get("char_ngram_range", (2, 4)),
            word_ngram_range=config.get("word_ngram_range", (1, 2)),
            vector_size=config.get("vector_size", 512),
            use_idf=config.get("use_idf", True)
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}. Choose 'openai', 'ollama', or 'ngram'")
    
    # Wrap with caching if requested (default: yes)
    if config.get("cache", True):
        cache_size = config.get("cache_size", 1000)
        provider = CachedEmbeddingProvider(provider, cache_size)
    
    return provider


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns value between -1 and 1, where:
    - 1 means identical direction
    - 0 means orthogonal
    - -1 means opposite direction
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same dimension: {len(vec1)} != {len(vec2)}")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 * norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


# Additional enhancements we could add:

class BatchOptimizedProvider:
    """
    Wrapper that batches embedding requests for efficiency.
    Useful when many single embed() calls happen close together.
    """
    
    def __init__(self, provider: EmbeddingProvider, batch_size: int = 32, wait_ms: int = 10):
        self.provider = provider
        self.batch_size = batch_size
        self.wait_ms = wait_ms
        # TODO: Implement request batching with a small time window
        pass


class PersistentCacheProvider:
    """
    Wrapper that persists cache to disk for reuse across sessions.
    Uses SQLite or simple JSON file for storage.
    """
    
    def __init__(self, provider: EmbeddingProvider, cache_path: str):
        self.provider = provider
        self.cache_path = cache_path
        # TODO: Implement persistent caching
        pass


class MultiProviderRouter:
    """
    Routes requests to multiple providers based on text characteristics.
    For example, use a fast model for short texts and a better model for long texts.
    """
    
    def __init__(self, providers: Dict[str, EmbeddingProvider], routing_rules=None):
        self.providers = providers
        self.routing_rules = routing_rules
        # TODO: Implement intelligent routing
        pass