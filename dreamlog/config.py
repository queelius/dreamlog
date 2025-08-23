"""
DreamLog Configuration System

Manages configuration for DreamLog including LLM settings, sampling parameters,
and other system settings. Supports both YAML and JSON formats.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict

# Optional YAML support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass 
class LLMSamplingConfig:
    """Configuration for how to sample context for LLM prompts"""
    max_facts: int = 20  # Maximum facts to include in prompt
    max_rules: int = 15  # Maximum rules to include in prompt
    strategy: str = "related"  # "related", "random", or "recent"
    include_stats: bool = True  # Include KB statistics


@dataclass
class LLMProviderConfig:
    """LLM provider configuration"""
    provider: str = "mock"
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None  # Env var containing API key
    temperature: float = 0.7
    max_tokens: int = 500
    base_url: Optional[str] = None
    timeout: int = 30
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment"""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        # Try standard env vars
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None


@dataclass
class QueryConfig:
    """Query evaluation configuration"""
    max_depth: int = 100
    max_iterations: int = 1000
    trace_enabled: bool = False
    cache_enabled: bool = True


@dataclass
class DreamLogConfig:
    """Main DreamLog configuration"""
    provider: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    sampling: LLMSamplingConfig = field(default_factory=LLMSamplingConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    
    # Prompt template configuration
    prompt_template: Optional[str] = None  # "default", "minimal", or path to custom template
    prompt_template_path: Optional[str] = None  # Path to custom template file
    
    # General settings
    auto_save: bool = False
    auto_save_path: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "DreamLogConfig":
        """
        Load configuration from file.
        
        Args:
            path: Path to config file. If None, searches for:
                  1. ~/.dreamlog/config.yaml (or .yml)
                  2. ~/.dreamlog/config.json
                  3. ./dreamlog_config.yaml (or .yml)
                  4. ./dreamlog_config.json
        
        Returns:
            DreamLogConfig instance
        """
        if path:
            return cls._load_from_file(Path(path))
        
        # Search for config in standard locations
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
        
        # Return default config if no file found
        return cls()
    
    @classmethod
    def _load_from_file(cls, path: Path) -> "DreamLogConfig":
        """Load config from specific file"""
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ImportError("PyYAML required for YAML config files. Install with: pip install pyyaml")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DreamLogConfig":
        """Create config from dictionary"""
        config = cls()
        
        # Load provider config
        if 'provider' in data:
            config.provider = LLMProviderConfig(**data['provider'])
        
        # Load sampling config  
        if 'sampling' in data:
            config.sampling = LLMSamplingConfig(**data['sampling'])
        
        # Load query config
        if 'query' in data:
            config.query = QueryConfig(**data['query'])
        
        # Load general settings
        for key in ['prompt_template', 'prompt_template_path', 'auto_save', 
                    'auto_save_path', 'log_level', 'log_file']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save config (extension determines format)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'provider': asdict(self.provider),
            'sampling': asdict(self.sampling),
            'query': asdict(self.query),
            'prompt_template': self.prompt_template,
            'prompt_template_path': self.prompt_template_path,
            'auto_save': self.auto_save,
            'auto_save_path': self.auto_save_path,
            'log_level': self.log_level,
            'log_file': self.log_file
        }
    
    def create_default_config_file(self) -> Path:
        """
        Create default config file in user's home directory.
        
        Returns:
            Path to created config file
        """
        config_dir = Path.home() / ".dreamlog"
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / "config.yaml"
        
        # Create with helpful comments
        config_content = """# DreamLog Configuration File
# 
# This file configures DreamLog's behavior including LLM integration,
# query evaluation, and system settings.

# LLM Provider Configuration
provider:
  # Provider type: "openai", "anthropic", "ollama", "mock", or "url"
  provider: mock
  
  # Model name (provider-specific)
  # model: gpt-4
  
  # API key (or use api_key_env for environment variable)
  # api_key: sk-...
  # api_key_env: OPENAI_API_KEY
  
  # Generation parameters
  temperature: 0.7
  max_tokens: 500
  
  # For local/custom providers
  # base_url: http://localhost:11434
  timeout: 30

# Context Sampling Configuration
sampling:
  # Maximum facts to include in LLM context
  max_facts: 20
  
  # Maximum rules to include in LLM context  
  max_rules: 15
  
  # How to sample: "related", "random", or "recent"
  strategy: related
  
  # Include KB statistics in context
  include_stats: true

# Query Evaluation Configuration
query:
  max_depth: 100
  max_iterations: 1000
  trace_enabled: false
  cache_enabled: true

# Prompt Template Configuration
# Use "default" for built-in, or path to custom template file
# prompt_template: default
# prompt_template_path: ~/.dreamlog/custom_prompt.txt

# General Settings
auto_save: false
# auto_save_path: ~/.dreamlog/autosave.dreamlog

# Logging
log_level: INFO
# log_file: ~/.dreamlog/dreamlog.log
"""
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path


# Global config instance
_config: Optional[DreamLogConfig] = None


def get_config() -> DreamLogConfig:
    """Get the global config instance"""
    global _config
    if _config is None:
        _config = DreamLogConfig.load()
    return _config


def set_config(config: DreamLogConfig) -> None:
    """Set the global config instance"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset to default config"""
    global _config
    _config = DreamLogConfig()