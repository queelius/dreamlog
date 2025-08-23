"""
RAG-Enhanced LLM Hook for DreamLog

Integrates RAG (Retrieval-Augmented Generation) with the LLM hook
to provide better examples in prompts based on similarity search.
"""

from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass

from .llm_hook import LLMHook
from .llm_providers import LLMProvider, LLMResponse
from .rag_framework import RAGSystem, RAGExample, RAGConfig
from .embedding_providers import (
    EmbeddingProvider, 
    OllamaEmbeddingProvider, 
    NGramEmbeddingProvider,
    CachedEmbeddingProvider
)
from .terms import Term
from .knowledge import Fact, Rule
from .prompt_template_system import QueryContext, PromptTemplateLibrary


@dataclass
class RAGHookConfig:
    """Configuration for RAG-enhanced LLM hook"""
    # Embedding provider settings
    embedding_provider: str = "ngram"  # "ollama", "ngram", or "openai"
    embedding_model: Optional[str] = None  # For Ollama/OpenAI
    embedding_cache_size: int = 1000
    
    # RAG settings
    max_examples: int = 5
    similarity_threshold: float = 0.5
    probability_weighted: bool = True
    
    # Storage settings
    persist_examples: bool = True
    examples_file: str = "dreamlog_examples.json"
    
    # Learning settings
    track_success: bool = True
    min_confidence_to_store: float = 0.7


class RAGEnhancedLLMHook(LLMHook):
    """
    LLM Hook enhanced with RAG for better example selection in prompts
    """
    
    def __init__(self, 
                 provider: LLMProvider,
                 config: Optional[RAGHookConfig] = None,
                 max_generations: int = 10,
                 cache_enabled: bool = True,
                 debug: bool = False):
        """
        Initialize RAG-enhanced LLM hook
        
        Args:
            provider: Base LLM provider
            config: RAG configuration
            max_generations: Maximum LLM generation calls
            cache_enabled: Whether to cache generated knowledge
            debug: Enable debug output
        """
        super().__init__(provider, max_generations, cache_enabled, debug)
        
        self.rag_config = config or RAGHookConfig()
        
        # Initialize embedding provider
        self.embedding_provider = self._create_embedding_provider()
        
        # Initialize RAG system
        rag_config = RAGConfig(
            max_examples=self.rag_config.max_examples,
            similarity_threshold=self.rag_config.similarity_threshold,
            probability_weighted_sampling=self.rag_config.probability_weighted
        )
        self.rag_system = RAGSystem(
            embedding_provider=self.embedding_provider,
            config=rag_config
        )
        
        # Load existing examples if available
        if self.rag_config.persist_examples:
            self._load_examples()
        
        # Track successful generations for learning
        self.successful_examples: List[Tuple[str, Dict, LLMResponse]] = []
    
    def _create_embedding_provider(self) -> EmbeddingProvider:
        """Create the configured embedding provider"""
        provider_type = self.rag_config.embedding_provider.lower()
        
        if provider_type == "ngram":
            # N-gram provider for local, fast similarity
            base_provider = NGramEmbeddingProvider(n=3)
        elif provider_type == "ollama":
            # Ollama for local neural embeddings
            if not self.rag_config.embedding_model:
                raise ValueError("Ollama embedding provider requires embedding_model")
            base_provider = OllamaEmbeddingProvider(
                model=self.rag_config.embedding_model
            )
        elif provider_type == "openai":
            # OpenAI for high-quality embeddings
            raise NotImplementedError("OpenAI embedding provider not yet implemented")
        else:
            raise ValueError(f"Unknown embedding provider: {provider_type}")
        
        # Wrap with cache for performance
        return CachedEmbeddingProvider(
            base_provider,
            cache_size=self.rag_config.embedding_cache_size
        )
    
    def _generate_knowledge(self, term: Term, kb) -> LLMResponse:
        """
        Generate knowledge with RAG-enhanced context
        
        Overrides parent method to add RAG example retrieval
        """
        # Create query context
        query_context = QueryContext(
            term=str(term),
            kb_facts=[str(fact.term) for fact in kb.facts[:20]],
            kb_rules=[(str(rule.head), [str(g) for g in rule.body]) for rule in kb.rules[:10]],
            existing_functors=list(set(
                fact.term.functor for fact in kb.facts 
                if hasattr(fact.term, 'functor')
            ))[:20]
        )
        
        # Retrieve similar examples from RAG
        similar_examples = self._retrieve_similar_examples(term, query_context)
        
        # Enhance prompt with RAG examples
        enhanced_prompt = self._build_rag_enhanced_prompt(query_context, similar_examples)
        
        if self.debug:
            print(f"\n[DEBUG] RAG-Enhanced Generation for: {term}")
            print(f"[DEBUG] Retrieved {len(similar_examples)} similar examples")
            if similar_examples:
                print(f"[DEBUG] Best match similarity: {similar_examples[0].similarity:.2f}")
        
        # Generate with enhanced prompt
        response = self.provider.generate_knowledge(str(term), context=enhanced_prompt)
        
        # Track successful generation for future learning
        if response and (response.facts or response.rules):
            self._record_successful_example(term, query_context, response)
        
        return response
    
    def _retrieve_similar_examples(self, term: Term, context: QueryContext) -> List[RAGExample]:
        """Retrieve similar examples from RAG system"""
        # Create query string for similarity search
        query = f"{term}"
        if hasattr(term, 'functor'):
            query = f"functor:{term.functor} query:{term}"
        
        # Retrieve examples
        examples = self.rag_system.retrieve_examples(
            query=query,
            n=self.rag_config.max_examples
        )
        
        # Filter by similarity threshold
        filtered = [
            ex for ex in examples 
            if ex.similarity >= self.rag_config.similarity_threshold
        ]
        
        return filtered
    
    def _build_rag_enhanced_prompt(self, 
                                   context: QueryContext, 
                                   examples: List[RAGExample]) -> str:
        """Build prompt enhanced with RAG examples"""
        prompt_parts = []
        
        # Add context from KB
        if context.kb_facts:
            prompt_parts.append(f"Known facts: {', '.join(context.kb_facts[:5])}")
        if context.kb_rules:
            rule_strs = [f"{h} :- {', '.join(b)}" for h, b in context.kb_rules[:3]]
            prompt_parts.append(f"Known rules: {'; '.join(rule_strs)}")
        
        # Add RAG examples if available
        if examples:
            prompt_parts.append("\nSimilar successful examples:")
            for i, ex in enumerate(examples[:3], 1):
                if 'query' in ex.metadata and 'response' in ex.metadata:
                    prompt_parts.append(f"\nExample {i} (similarity: {ex.similarity:.2f}):")
                    prompt_parts.append(f"  Query: {ex.metadata['query']}")
                    prompt_parts.append(f"  Generated: {ex.metadata['response']}")
        
        # Add generation instructions
        prompt_parts.append("\nGenerate similar knowledge for the current query.")
        prompt_parts.append("Output format: [['rule', head, body]] or [['fact', data]]")
        
        return "\n".join(prompt_parts)
    
    def _record_successful_example(self, 
                                   term: Term, 
                                   context: QueryContext,
                                   response: LLMResponse):
        """Record successful generation as RAG example"""
        # Create example metadata
        metadata = {
            'query': str(term),
            'functor': term.functor if hasattr(term, 'functor') else None,
            'response': self._format_response_for_example(response),
            'kb_size': len(context.kb_facts),
            'success': True
        }
        
        # Add to RAG system
        example_text = f"Query: {term} Response: {metadata['response']}"
        self.rag_system.add_example(
            text=example_text,
            metadata=metadata,
            confidence=1.0  # High confidence for successful generation
        )
        
        if self.debug:
            print(f"[DEBUG] Recorded successful example for future RAG retrieval")
        
        # Persist if configured
        if self.rag_config.persist_examples:
            self._save_examples()
    
    def _format_response_for_example(self, response: LLMResponse) -> str:
        """Format LLM response for storage as example"""
        parts = []
        if response.facts:
            for fact in response.facts[:2]:  # Limit to 2 facts
                parts.append(f"fact:{json.dumps(fact)}")
        if response.rules:
            for rule in response.rules[:1]:  # Limit to 1 rule
                parts.append(f"rule:{json.dumps(rule)}")
        return " ".join(parts)
    
    def _load_examples(self):
        """Load persisted examples from file"""
        try:
            import os
            if os.path.exists(self.rag_config.examples_file):
                with open(self.rag_config.examples_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('examples', []):
                        self.rag_system.add_example(
                            text=item['text'],
                            metadata=item.get('metadata', {}),
                            confidence=item.get('confidence', 0.5)
                        )
                if self.debug:
                    print(f"[DEBUG] Loaded {len(data.get('examples', []))} examples from {self.rag_config.examples_file}")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Could not load examples: {e}")
    
    def _save_examples(self):
        """Save examples to file for persistence"""
        try:
            # Get all examples from RAG system
            examples_data = []
            for ex in self.rag_system.examples:
                examples_data.append({
                    'text': ex.text,
                    'metadata': ex.metadata,
                    'confidence': ex.confidence
                })
            
            # Save to file
            with open(self.rag_config.examples_file, 'w') as f:
                json.dump({'examples': examples_data}, f, indent=2)
                
            if self.debug:
                print(f"[DEBUG] Saved {len(examples_data)} examples to {self.rag_config.examples_file}")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Could not save examples: {e}")
    
    def get_rag_stats(self) -> Dict[str, int]:
        """Get statistics about RAG system"""
        return {
            'total_examples': len(self.rag_system.examples),
            'cached_embeddings': len(self.embedding_provider.cache) if hasattr(self.embedding_provider, 'cache') else 0,
            'successful_generations': len(self.successful_examples)
        }


def create_rag_enhanced_engine(llm_provider: LLMProvider,
                               embedding_provider: str = "ngram",
                               debug: bool = False) -> 'DreamLogEngine':
    """
    Create a DreamLog engine with RAG-enhanced LLM integration
    
    Args:
        llm_provider: Base LLM provider
        embedding_provider: Type of embedding provider ("ngram", "ollama")
        debug: Enable debug output
    
    Returns:
        DreamLogEngine with RAG enhancement
    """
    from .engine import DreamLogEngine
    
    # Configure RAG
    rag_config = RAGHookConfig(
        embedding_provider=embedding_provider,
        persist_examples=True,
        track_success=True
    )
    
    # Create RAG-enhanced hook
    rag_hook = RAGEnhancedLLMHook(
        provider=llm_provider,
        config=rag_config,
        debug=debug
    )
    
    # Create engine with RAG hook
    return DreamLogEngine(rag_hook)