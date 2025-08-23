"""
Extended Configuration System for DreamLog

Supports all the advanced features: RAG, dream cycles, user feedback, etc.
Backward compatible with the existing simple config.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import os
import json
import yaml
import re

from .config import DreamLogConfig, LLMProviderConfig, LLMSamplingConfig, QueryConfig


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration"""
    provider: str = "ngram"  # openai, ollama, ngram
    cache: bool = True
    cache_size: int = 1000
    
    # Provider-specific settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    
    openai_api_key: Optional[str] = None
    openai_model: str = "text-embedding-3-small"
    openai_base_url: str = "https://api.openai.com/v1"
    
    ngram_char_range: List[int] = field(default_factory=lambda: [2, 4])
    ngram_word_range: List[int] = field(default_factory=lambda: [1, 2])
    ngram_vector_size: int = 512
    ngram_use_idf: bool = True


@dataclass
class RAGDatabaseConfig:
    """Configuration for a RAG database"""
    enabled: bool = True
    db_path: str = "~/.dreamlog/rag.json"
    auto_save: bool = True
    retrieval_k: int = 8
    retrieval_threshold: float = 0.0
    retrieval_temperature: float = 1.0
    use_confidence: bool = True
    min_uses_for_confidence: int = 3
    prune_threshold: float = 0.2
    prune_min_uses: int = 5


@dataclass
class RAGConfig:
    """Complete RAG system configuration"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    examples: RAGDatabaseConfig = field(default_factory=lambda: RAGDatabaseConfig(
        db_path="~/.dreamlog/example_rag.json"
    ))
    templates: RAGDatabaseConfig = field(default_factory=lambda: RAGDatabaseConfig(
        db_path="~/.dreamlog/template_rag.json",
        retrieval_k=3,
        retrieval_temperature=0.7
    ))
    negative_examples: RAGDatabaseConfig = field(default_factory=lambda: RAGDatabaseConfig(
        db_path="~/.dreamlog/negative_rag.json",
        retrieval_k=3
    ))
    
    # Negative example generation
    generate_variations: bool = True
    max_variations_per_failure: int = 5
    include_negatives_in_prompts: bool = True
    max_negatives_in_prompt: int = 3
    
    # Meta-learning
    meta_tracking_path: str = "~/.dreamlog/meta_patterns.json"
    meta_pattern_buffer_size: int = 10000
    meta_min_examples: int = 10


@dataclass
class DreamCycleConfig:
    """Dream cycle configuration"""
    enabled: bool = False
    
    # Scheduling
    schedule_mode: str = "periodic"  # periodic, idle, manual, continuous
    period_minutes: int = 60
    idle_threshold_seconds: int = 300
    max_duration_seconds: int = 600
    checkpoint_interval: int = 60
    
    # Experience replay
    experience_replay_enabled: bool = True
    experience_batch_size: int = 32
    replay_buffer_size: int = 1000
    prioritized_replay: bool = True
    
    # LLM judge
    llm_judge_enabled: bool = True
    llm_judge_providers: List[str] = field(default_factory=lambda: ["ollama"])
    num_judges: int = 3
    consensus_threshold: float = 0.7
    allow_unsure: bool = True
    max_evaluations_per_cycle: int = 100
    
    # Success thresholds
    success_threshold: float = 0.7
    failure_threshold: float = 0.3
    high_confidence_threshold: float = 0.9
    
    # Knowledge operations
    compression_enabled: bool = True
    min_rules_to_compress: int = 3
    find_abstractions: bool = True
    find_redundancy: bool = True


@dataclass
class UserFeedbackConfig:
    """User feedback and active learning configuration"""
    # Basic collection
    collection_enabled: bool = False
    optional: bool = True
    prompt_style: str = "minimal"  # minimal, detailed, educational
    save_path: str = "~/.dreamlog/user_feedback.json"
    
    # Active learning
    active_learning_enabled: bool = False
    intrusiveness: str = "low"  # off, low, medium, high, adaptive
    questions_per_session: Dict[str, int] = field(default_factory=lambda: {
        "low": 1,
        "medium": 3,
        "high": 5
    })
    
    # Triggers
    trigger_on_uncertainty: bool = True
    trigger_on_disagreement: bool = True
    trigger_on_novelty: bool = True
    uncertainty_threshold: float = 0.4
    disagreement_threshold: float = 0.3
    novelty_threshold: float = 0.2
    
    # Trust scores
    trust_scores: Dict[str, float] = field(default_factory=lambda: {
        "user": 1.0,
        "llm_judge": 0.6,
        "automatic": 0.3,
        "inferred": 0.1
    })
    
    # Drift detection
    drift_detection_enabled: bool = True
    drift_check_interval: int = 100
    drift_threshold: float = 0.2
    auto_recalibrate: bool = True


@dataclass
class ExtendedDreamLogConfig(DreamLogConfig):
    """
    Extended configuration with all advanced features.
    Inherits from base DreamLogConfig for backward compatibility.
    """
    rag: RAGConfig = field(default_factory=RAGConfig)
    dream_cycle: DreamCycleConfig = field(default_factory=DreamCycleConfig)
    user_feedback: UserFeedbackConfig = field(default_factory=UserFeedbackConfig)
    
    # Experimental flags
    experiments: Dict[str, bool] = field(default_factory=lambda: {
        "use_context_embeddings": True,
        "use_clustering": False,
        "adaptive_temperature": True,
        "curriculum_learning": False,
        "adversarial_examples": False,
        "prompt_evolution": False
    })
    
    # Performance settings
    cache_query_results: bool = True
    cache_size: int = 100
    max_parallel_workers: int = 4
    batch_embeddings: bool = True
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "ExtendedDreamLogConfig":
        """Load extended configuration from file."""
        if path:
            return cls._load_from_file(Path(path))
        
        # Search paths (including parent class paths)
        search_paths = [
            Path.home() / ".dreamlog" / "config.yaml",
            Path.home() / ".dreamlog" / "config.yml",
            Path.home() / ".dreamlog" / "config.json",
            Path("dreamlog_config.yaml"),
            Path("dreamlog_config.yml"),
            Path("dreamlog_config.json"),
        ]
        
        for config_path in search_paths:
            if config_path.exists():
                return cls._load_from_file(config_path)
        
        return cls()
    
    @classmethod
    def _load_from_file(cls, path: Path) -> "ExtendedDreamLogConfig":
        """Load from specific file."""
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtendedDreamLogConfig":
        """Create from dictionary, handling nested configs."""
        # Start with base config
        config = super().from_dict(data)
        extended = cls(
            provider=config.provider,
            sampling=config.sampling,
            query=config.query,
            prompt_template=config.prompt_template,
            prompt_template_path=config.prompt_template_path,
            auto_save=config.auto_save,
            auto_save_path=config.auto_save_path,
            log_level=config.log_level,
            log_file=config.log_file
        )
        
        # Load RAG config
        if 'rag' in data:
            extended.rag = cls._load_rag_config(data['rag'])
        
        # Load dream cycle config
        if 'dream_cycle' in data:
            extended.dream_cycle = cls._load_dream_cycle_config(data['dream_cycle'])
        
        # Load user feedback config
        if 'user_feedback' in data:
            extended.user_feedback = cls._load_user_feedback_config(data['user_feedback'])
        
        # Load experiments
        if 'experiments' in data:
            extended.experiments.update(data['experiments'])
        
        # Load performance settings
        for key in ['cache_query_results', 'cache_size', 'max_parallel_workers', 'batch_embeddings']:
            if key in data:
                setattr(extended, key, data[key])
        
        return extended
    
    @staticmethod
    def _load_rag_config(data: Dict[str, Any]) -> RAGConfig:
        """Load RAG configuration from dict."""
        config = RAGConfig()
        
        if 'embedding' in data:
            config.embedding = EmbeddingConfig(**data['embedding'])
        
        for db_name in ['examples', 'templates', 'negative_examples']:
            if db_name in data:
                setattr(config, db_name, RAGDatabaseConfig(**data[db_name]))
        
        # Load other RAG settings
        for key in ['generate_variations', 'max_variations_per_failure', 
                   'include_negatives_in_prompts', 'max_negatives_in_prompt',
                   'meta_tracking_path', 'meta_pattern_buffer_size', 'meta_min_examples']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    @staticmethod
    def _load_dream_cycle_config(data: Dict[str, Any]) -> DreamCycleConfig:
        """Load dream cycle configuration from dict."""
        return DreamCycleConfig(**{k: v for k, v in data.items() 
                                   if k in DreamCycleConfig.__dataclass_fields__})
    
    @staticmethod
    def _load_user_feedback_config(data: Dict[str, Any]) -> UserFeedbackConfig:
        """Load user feedback configuration from dict."""
        return UserFeedbackConfig(**{k: v for k, v in data.items()
                                     if k in UserFeedbackConfig.__dataclass_fields__})
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding provider configuration for current provider."""
        embed = self.rag.embedding
        provider = embed.provider
        
        if provider == "openai":
            return {
                "api_key": embed.openai_api_key or os.getenv("OPENAI_API_KEY"),
                "model": embed.openai_model,
                "base_url": embed.openai_base_url,
                "cache_size": embed.cache_size if embed.cache else 0
            }
        elif provider == "ollama":
            return {
                "base_url": embed.ollama_base_url,
                "model": embed.ollama_model,
                "cache_size": embed.cache_size if embed.cache else 0
            }
        elif provider == "ngram":
            return {
                "char_ngram_range": tuple(embed.ngram_char_range),
                "word_ngram_range": tuple(embed.ngram_word_range),
                "vector_size": embed.ngram_vector_size,
                "use_idf": embed.ngram_use_idf
            }
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def validate(self) -> List[str]:
        """Validate configuration for issues."""
        issues = []
        
        # Check API keys
        if self.provider.provider == "openai" and not self.provider.get_api_key():
            issues.append("OpenAI LLM provider selected but no API key configured")
        
        if self.rag.embedding.provider == "openai" and not (
            self.rag.embedding.openai_api_key or os.getenv("OPENAI_API_KEY")
        ):
            issues.append("OpenAI embedding provider selected but no API key configured")
        
        # Check paths
        paths_to_check = [
            self.rag.examples.db_path,
            self.rag.templates.db_path,
            self.rag.negative_examples.db_path,
            self.user_feedback.save_path
        ]
        
        for path_str in paths_to_check:
            path = Path(path_str).expanduser()
            if not path.parent.exists():
                issues.append(f"Parent directory does not exist: {path.parent}")
        
        # Check logical consistency
        if self.dream_cycle.enabled and not self.rag.examples.enabled:
            issues.append("Dream cycle enabled but RAG examples disabled")
        
        if self.user_feedback.active_learning_enabled and not self.user_feedback.collection_enabled:
            issues.append("Active learning enabled but feedback collection disabled")
        
        return issues


# Global extended config instance
_extended_config: Optional[ExtendedDreamLogConfig] = None


def get_extended_config() -> ExtendedDreamLogConfig:
    """Get the global extended config instance."""
    global _extended_config
    if _extended_config is None:
        _extended_config = ExtendedDreamLogConfig.load()
    return _extended_config


def set_extended_config(config: ExtendedDreamLogConfig) -> None:
    """Set the global extended config instance."""
    global _extended_config
    _extended_config = config