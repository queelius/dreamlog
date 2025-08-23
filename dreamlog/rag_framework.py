"""
Generic RAG Framework for DreamLog

A flexible RAG system that can be used for:
- Example retrieval (query -> similar examples)
- Prompt template selection (query -> successful templates)
- Meta-learning (tracking what works when)

The same framework and embedding providers work for all purposes.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Protocol, Generic, TypeVar
from dataclasses import dataclass, asdict, field
from datetime import datetime
from abc import ABC, abstractmethod

from .embedding_providers import EmbeddingProvider, cosine_similarity


T = TypeVar('T')  # Generic type for RAG items


@dataclass
class RAGItem(Generic[T]):
    """
    Generic item stored in RAG database.
    Can be an example, prompt template, or any other retrievable item.
    """
    id: str
    content: T  # The actual content (example, template, etc.)
    text_for_embedding: str  # Text used to generate embedding
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Learning/tracking fields
    use_count: int = 0
    success_count: int = 0
    last_used: Optional[str] = None
    source: str = "initial"  # "initial", "learned", "generated", "user"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count
    
    @property
    def confidence_score(self) -> float:
        """
        Confidence score based on usage and success.
        Higher usage with consistent success = higher confidence.
        """
        if self.use_count < 3:
            return 0.0  # Not enough data
        
        success_rate = self.success_rate
        # Penalize if too few uses
        usage_factor = min(1.0, self.use_count / 10.0)
        return success_rate * usage_factor


class RAGDatabase(Generic[T]):
    """
    Generic RAG database for storing and retrieving items.
    Uses embedding similarity for retrieval.
    """
    
    def __init__(self,
                 embedding_provider: EmbeddingProvider,
                 db_path: Optional[Path] = None,
                 auto_save: bool = True):
        """
        Initialize RAG database.
        
        Args:
            embedding_provider: Provider for generating embeddings
            db_path: Path to persist the database
            auto_save: Whether to auto-save after modifications
        """
        self.embedding_provider = embedding_provider
        self.db_path = Path(db_path) if db_path else None
        self.auto_save = auto_save
        self.items: List[RAGItem[T]] = []
        self._id_counter = 0
        
        # Load existing database if path provided
        if self.db_path and self.db_path.exists():
            self.load()
    
    def add_item(self, 
                 content: T,
                 text_for_embedding: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 source: str = "user") -> str:
        """
        Add a new item to the database.
        
        Returns:
            ID of the added item
        """
        # Generate ID
        item_id = f"{source}_{self._id_counter}"
        self._id_counter += 1
        
        # Generate embedding
        embedding = self.embedding_provider.embed(text_for_embedding)
        
        # Create item
        item = RAGItem(
            id=item_id,
            content=content,
            text_for_embedding=text_for_embedding,
            embedding=embedding,
            metadata=metadata or {},
            source=source
        )
        
        self.items.append(item)
        
        if self.auto_save and self.db_path:
            self.save()
        
        return item_id
    
    def retrieve(self,
                 query: str,
                 k: int = 10,
                 threshold: float = 0.0,
                 filter_fn: Optional[callable] = None) -> List[Tuple[RAGItem[T], float]]:
        """
        Retrieve top-k most similar items.
        
        Args:
            query: Query text
            k: Number of items to retrieve
            threshold: Minimum similarity threshold
            filter_fn: Optional filter function for items
            
        Returns:
            List of (item, similarity_score) tuples
        """
        if not self.items:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed(query)
        
        # Compute similarities
        similarities = []
        for item in self.items:
            if filter_fn and not filter_fn(item):
                continue
            
            if item.embedding:
                sim = cosine_similarity(query_embedding, item.embedding)
                if sim >= threshold:
                    similarities.append((item, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def retrieve_weighted(self,
                         query: str,
                         k: int = 10,
                         temperature: float = 1.0,
                         use_confidence: bool = True) -> List[RAGItem[T]]:
        """
        Retrieve items using weighted sampling based on similarity and confidence.
        
        Args:
            query: Query text
            k: Number of items to sample
            temperature: Temperature for sampling (higher = more random)
            use_confidence: Whether to factor in success/confidence scores
            
        Returns:
            List of sampled items
        """
        # Get similarities
        similarities = self.retrieve(query, k=min(50, len(self.items)))
        
        if not similarities:
            return []
        
        # Calculate weights
        items = []
        weights = []
        
        for item, sim in similarities:
            items.append(item)
            
            # Base weight is similarity
            weight = sim
            
            # Factor in confidence if requested
            if use_confidence:
                confidence = item.confidence_score
                # Blend similarity and confidence
                weight = 0.7 * sim + 0.3 * confidence
            
            weights.append(weight)
        
        # Apply temperature
        weights = np.array(weights)
        weights = weights / temperature
        
        # Softmax
        exp_weights = np.exp(weights - np.max(weights))
        probabilities = exp_weights / np.sum(exp_weights)
        
        # Sample
        k = min(k, len(items))
        indices = np.random.choice(
            len(items),
            size=k,
            replace=False,
            p=probabilities
        )
        
        return [items[i] for i in indices]
    
    def update_item_stats(self,
                         item_id: str,
                         success: bool,
                         metadata_update: Optional[Dict[str, Any]] = None):
        """
        Update usage statistics for an item.
        
        Args:
            item_id: ID of the item
            success: Whether the usage was successful
            metadata_update: Optional metadata to add/update
        """
        for item in self.items:
            if item.id == item_id:
                item.use_count += 1
                if success:
                    item.success_count += 1
                item.last_used = datetime.now().isoformat()
                
                if metadata_update:
                    item.metadata.update(metadata_update)
                
                if self.auto_save and self.db_path:
                    self.save()
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.items:
            return {
                "total_items": 0,
                "sources": {},
                "avg_success_rate": 0.0,
                "high_confidence_items": 0
            }
        
        sources = {}
        total_success = 0
        total_uses = 0
        high_confidence = 0
        
        for item in self.items:
            # Count by source
            sources[item.source] = sources.get(item.source, 0) + 1
            
            # Track success
            total_success += item.success_count
            total_uses += item.use_count
            
            # Count high confidence
            if item.confidence_score > 0.7:
                high_confidence += 1
        
        return {
            "total_items": len(self.items),
            "sources": sources,
            "avg_success_rate": total_success / total_uses if total_uses > 0 else 0.0,
            "high_confidence_items": high_confidence,
            "total_uses": total_uses
        }
    
    def prune_low_performers(self, min_uses: int = 5, max_failure_rate: float = 0.8):
        """
        Remove items that consistently fail.
        
        Args:
            min_uses: Minimum uses before considering pruning
            max_failure_rate: Maximum acceptable failure rate
        """
        original_count = len(self.items)
        self.items = [
            item for item in self.items
            if item.use_count < min_uses or item.success_rate > (1 - max_failure_rate)
        ]
        
        pruned = original_count - len(self.items)
        if pruned > 0:
            print(f"Pruned {pruned} low-performing items")
            if self.auto_save and self.db_path:
                self.save()
    
    def save(self):
        """Save database to disk"""
        if not self.db_path:
            return
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        data = {
            "items": [self._item_to_dict(item) for item in self.items],
            "id_counter": self._id_counter
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self):
        """Load database from disk"""
        if not self.db_path or not self.db_path.exists():
            return
        
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        self.items = [self._dict_to_item(d) for d in data.get("items", [])]
        self._id_counter = data.get("id_counter", 0)
    
    def _item_to_dict(self, item: RAGItem[T]) -> Dict[str, Any]:
        """Convert item to dictionary for serialization"""
        return {
            "id": item.id,
            "content": item.content,  # Assumes T is JSON-serializable
            "text_for_embedding": item.text_for_embedding,
            "embedding": item.embedding,
            "metadata": item.metadata,
            "use_count": item.use_count,
            "success_count": item.success_count,
            "last_used": item.last_used,
            "source": item.source,
            "created_at": item.created_at
        }
    
    def _dict_to_item(self, d: Dict[str, Any]) -> RAGItem[T]:
        """Convert dictionary to item"""
        return RAGItem(
            id=d["id"],
            content=d["content"],
            text_for_embedding=d["text_for_embedding"],
            embedding=d.get("embedding"),
            metadata=d.get("metadata", {}),
            use_count=d.get("use_count", 0),
            success_count=d.get("success_count", 0),
            last_used=d.get("last_used"),
            source=d.get("source", "unknown"),
            created_at=d.get("created_at", datetime.now().isoformat())
        )


class MetaLearningTracker:
    """
    Tracks meta-learning patterns across different RAG databases.
    Identifies what works when and helps improve the system over time.
    """
    
    def __init__(self, tracking_path: Optional[Path] = None):
        self.tracking_path = Path(tracking_path) if tracking_path else None
        self.patterns: List[Dict[str, Any]] = []
        
        if self.tracking_path and self.tracking_path.exists():
            self.load()
    
    def record_usage(self,
                    query: str,
                    query_type: str,
                    selected_items: List[str],  # Item IDs
                    success: bool,
                    context: Optional[Dict[str, Any]] = None):
        """
        Record a usage pattern.
        
        Args:
            query: The original query
            query_type: Type of query (e.g., "grandparent", "recursive", etc.)
            selected_items: IDs of items that were selected
            success: Whether the overall operation succeeded
            context: Additional context about the usage
        """
        pattern = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_type": query_type,
            "selected_items": selected_items,
            "success": success,
            "context": context or {}
        }
        
        self.patterns.append(pattern)
        
        # Keep only recent patterns (e.g., last 10000)
        if len(self.patterns) > 10000:
            self.patterns = self.patterns[-10000:]
        
        if self.tracking_path:
            self.save()
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze recorded patterns to find insights.
        
        Returns:
            Dictionary of insights and statistics
        """
        if not self.patterns:
            return {}
        
        # Analyze by query type
        by_type = {}
        for pattern in self.patterns:
            qtype = pattern["query_type"]
            if qtype not in by_type:
                by_type[qtype] = {"total": 0, "success": 0, "items": {}}
            
            by_type[qtype]["total"] += 1
            if pattern["success"]:
                by_type[qtype]["success"] += 1
            
            # Track which items work for this type
            for item_id in pattern["selected_items"]:
                if item_id not in by_type[qtype]["items"]:
                    by_type[qtype]["items"][item_id] = {"uses": 0, "successes": 0}
                by_type[qtype]["items"][item_id]["uses"] += 1
                if pattern["success"]:
                    by_type[qtype]["items"][item_id]["successes"] += 1
        
        # Calculate success rates
        insights = {
            "total_patterns": len(self.patterns),
            "by_query_type": {}
        }
        
        for qtype, stats in by_type.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            
            # Find best items for this type
            best_items = []
            for item_id, item_stats in stats["items"].items():
                if item_stats["uses"] >= 3:  # Minimum uses
                    item_success_rate = item_stats["successes"] / item_stats["uses"]
                    best_items.append((item_id, item_success_rate, item_stats["uses"]))
            
            best_items.sort(key=lambda x: x[1], reverse=True)
            
            insights["by_query_type"][qtype] = {
                "success_rate": success_rate,
                "total_queries": stats["total"],
                "best_items": best_items[:5]  # Top 5 items
            }
        
        return insights
    
    def get_recommendations(self, query_type: str) -> List[str]:
        """
        Get recommended item IDs for a query type based on historical success.
        
        Args:
            query_type: Type of query
            
        Returns:
            List of recommended item IDs
        """
        analysis = self.analyze_patterns()
        
        if query_type in analysis.get("by_query_type", {}):
            type_data = analysis["by_query_type"][query_type]
            return [item_id for item_id, _, _ in type_data.get("best_items", [])]
        
        return []
    
    def save(self):
        """Save tracking data to disk"""
        if not self.tracking_path:
            return
        
        self.tracking_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.tracking_path, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def load(self):
        """Load tracking data from disk"""
        if not self.tracking_path or not self.tracking_path.exists():
            return
        
        with open(self.tracking_path, 'r') as f:
            self.patterns = json.load(f)