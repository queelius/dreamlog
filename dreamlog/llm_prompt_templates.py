"""
LLM Prompt Template Management for DreamLog

Handles loading and managing prompt templates from files or strings.
"""

from pathlib import Path
from string import Template
from typing import Optional, Dict, Any


class PromptTemplateManager:
    """Manages prompt templates for LLM knowledge generation"""
    
    def __init__(self, template_source: Optional[str] = None):
        """
        Initialize with a template.
        
        Args:
            template_source: Can be:
                - None or "default": Use default template
                - Path to a template file
                - Template string (if contains ${)
        """
        self.template = self._load_template(template_source)
    
    def _load_template(self, source: Optional[str]) -> Template:
        """Load template from various sources"""
        if source is None or source == "default":
            # Load default template from templates directory
            template_path = Path(__file__).parent / "templates" / "default.txt"
            if template_path.exists():
                with open(template_path, 'r') as f:
                    return Template(f.read())
            else:
                # Fallback to hardcoded template if file not found
                return Template(self._get_fallback_template())
        
        # Check if it's a file path
        if source and not "${" in source:
            path = Path(source).expanduser()
            if path.exists():
                with open(path, 'r') as f:
                    return Template(f.read())
        
        # Treat as template string
        if source and "${" in source:
            return Template(source)
        
        # Default fallback
        return Template(self._get_fallback_template())
    
    def _get_fallback_template(self) -> str:
        """Hardcoded fallback template"""
        return """Term: ${term}
Existing KB:
${knowledge_base}

Generate JSON array with facts/rules for "${term}":"""
    
    def create_prompt(self, **kwargs: Any) -> str:
        """
        Create a prompt by substituting template variables.
        
        Common variables:
            - term: The term being queried
            - functor: Main functor of the term
            - arity: Number of arguments
            - args: String representation of arguments
            - knowledge_base: Current KB contents (facts and rules)
            
        Additional custom variables can be provided via kwargs.
        """
        # Provide safe defaults for essential variables
        defaults = {
            'term': 'unknown',
            'functor': 'unknown',
            'arity': '0',
            'args': '',
            'knowledge_base': '(No existing knowledge)'
        }
        
        # Merge with provided kwargs
        params = {**defaults, **kwargs}
        
        # Use safe_substitute to avoid KeyError for missing variables
        return self.template.safe_substitute(**params)
    
    @classmethod
    def from_config(cls, config) -> "PromptTemplateManager":
        """
        Create template manager from config object.
        
        Args:
            config: DreamLogConfig object with prompt_template settings
        """
        if config.prompt_template_path:
            # Use specified template file
            return cls(config.prompt_template_path)
        elif config.prompt_template:
            # Use named template or string
            return cls(config.prompt_template)
        else:
            # Use default
            return cls()
    
    @classmethod
    def list_builtin_templates(cls) -> Dict[str, str]:
        """List available built-in templates"""
        templates_dir = Path(__file__).parent / "templates"
        templates = {}
        
        if templates_dir.exists():
            for template_file in templates_dir.glob("*.txt"):
                name = template_file.stem
                templates[name] = str(template_file)
        
        return templates


def create_custom_template(template_str: str) -> PromptTemplateManager:
    """
    Create a PromptTemplateManager with a custom template string.
    
    Example:
        template = create_custom_template('''
        Generate facts about ${term}.
        Context: ${knowledge_base}
        ''')
        prompt = template.create_prompt(term="parent", knowledge_base="...")
    """
    return PromptTemplateManager(template_source=template_str)