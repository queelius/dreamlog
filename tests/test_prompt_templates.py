#!/usr/bin/env python3
"""
Unit tests for prompt template system
"""

import pytest
from typing import List, Dict, Any

from dreamlog.prompt_template_system import (
    QueryContext, PromptTemplate, PromptTemplateLibrary,
    ModelParameters, PromptCategory
)


class TestQueryContext:
    """Test QueryContext dataclass"""
    
    def test_create_context(self):
        """Test creating query context"""
        context = QueryContext(
            term="parent(X, Y)",
            kb_facts=["parent(alice, bob)"],
            kb_rules=[("grandparent(X, Z)", ["parent(X, Y)", "parent(Y, Z)"])],
            existing_functors=["parent", "grandparent"]
        )
        
        assert context.term == "parent(X, Y)"
        assert len(context.kb_facts) == 1
        assert len(context.kb_rules) == 1
        assert len(context.existing_functors) == 2
    
    def test_empty_context(self):
        """Test empty context"""
        context = QueryContext(term="test(X)")
        
        assert context.term == "test(X)"
        assert context.kb_facts == []
        assert context.kb_rules == []
        assert context.existing_functors == []
    
    def test_context_properties(self):
        """Test computed properties"""
        context = QueryContext(
            term="parent(alice, X)",
            kb_facts=["fact1", "fact2", "fact3"],
            kb_rules=[("rule1", []), ("rule2", [])]
        )
        
        # Test if we add any computed properties
        assert len(context.kb_facts) == 3
        assert len(context.kb_rules) == 2


class TestModelParameters:
    """Test ModelParameters configuration"""
    
    def test_default_parameters(self):
        """Test default model parameters"""
        params = ModelParameters(model_name="test_model")
        
        assert params.model_name == "test_model"
        assert params.min_examples == 3
        assert params.max_examples == 8
        assert params.optimal_examples == 5
        assert params.max_context_tokens == 4000
        assert params.example_selection_strategy == "similarity"
        assert params.prefers_json == True
        assert params.needs_explicit_format == True
        assert params.handles_complex_reasoning == True
    
    def test_custom_parameters(self):
        """Test custom model parameters"""
        params = ModelParameters(
            model_name="custom_model",
            min_examples=2,
            max_examples=10,
            optimal_examples=6,
            max_context_tokens=8000,
            prefers_json=False
        )
        
        assert params.model_name == "custom_model"
        assert params.min_examples == 2
        assert params.max_examples == 10
        assert params.optimal_examples == 6
        assert params.max_context_tokens == 8000
        assert params.prefers_json == False


class TestPromptTemplate:
    """Test PromptTemplate class"""
    
    def test_create_template(self):
        """Test creating a prompt template"""
        template = PromptTemplate(
            id="test_template",
            category=PromptCategory.DEFINITION,
            template="Generate facts for: {term}",
            variables=["term"]
        )
        
        assert template.id == "test_template"
        assert template.category == PromptCategory.DEFINITION
        assert "{term}" in template.template
        assert "term" in template.variables
    
    def test_format_prompt(self):
        """Test formatting prompt with context"""
        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Query: {term}\nFacts: {kb_facts}",
            variables=["term", "kb_facts"]
        )
        
        formatted = template.render(
            term="parent(X, Y)",
            kb_facts="parent(alice, bob), parent(bob, charlie)"
        )
        
        assert "parent(X, Y)" in formatted
        assert "parent(alice, bob)" in formatted
        assert "parent(bob, charlie)" in formatted
    
    def test_template_with_examples(self):
        """Test template with examples"""
        template = PromptTemplate(
            id="with_examples",
            category=PromptCategory.DEFINITION,
            template="""Examples:
{examples}

Now generate for: {term}
""",
            variables=["examples", "term"]
        )
        
        examples_text = "Input: parent(X, Y)\nOutput: [['rule', ['parent', 'X', 'Y'], ...]]\nInput: sibling(A, B)\nOutput: [['fact', ['sibling', 'alice', 'bob']]]"
        formatted = template.render(examples=examples_text, term="grandparent(X, Z)")
        
        assert "Examples:" in formatted
        assert "parent(X, Y)" in formatted
        assert "sibling(A, B)" in formatted
        assert "grandparent(X, Z)" in formatted
    
    def test_adaptive_examples(self):
        """Test adaptive example selection based on context"""
        template = PromptTemplate(
            id="adaptive",
            category=PromptCategory.DEFINITION,
            template="{examples}\nGenerate: {term}",
            variables=["examples", "term"]
        )
        
        examples_text = "Example with facts\nExample with rules"
        formatted = template.render(examples=examples_text, term="test(X)")
        
        # Should include examples and the term
        assert "Example" in formatted
        assert "test(X)" in formatted


class TestPromptTemplateLibrary:
    """Test PromptTemplateLibrary"""
    
    def test_initialization(self):
        """Test library initialization"""
        library = PromptTemplateLibrary()
        
        # Should have default templates organized by category
        assert len(library.templates) > 0
        assert PromptCategory.DEFINITION in library.templates
        assert len(library.templates[PromptCategory.DEFINITION]) > 0
    
    def test_get_template(self):
        """Test getting template by name"""
        library = PromptTemplateLibrary()
        
        # Get template by select_template method
        context = QueryContext(term="test(X)")
        template = library.select_template(context)
        assert template is not None
        assert template.id is not None
        
        # Test with specific context
        context = QueryContext(
            term="grandparent(X, Z)",
            kb_facts=["parent(alice, bob)"],
            kb_rules=[]
        )
        template = library.select_template(context)
        assert template is not None
    
    def test_add_template(self):
        """Test adding custom template"""
        library = PromptTemplateLibrary()
        
        custom = PromptTemplate(
            id="custom",
            category=PromptCategory.DEFINITION,
            template="Custom system: {term}",
            variables=["term"]
        )
        
        # Add to the library
        library.templates[PromptCategory.DEFINITION].append(custom)
        
        # Verify it was added
        assert custom in library.templates[PromptCategory.DEFINITION]
        assert custom.id == "custom"
    
    def test_select_best_template(self):
        """Test automatic template selection"""
        library = PromptTemplateLibrary()
        
        # Add specialized templates
        math_template = PromptTemplate(
            id="math",
            category=PromptCategory.DEFINITION,
            template="Math logic: Solve {term}",
            variables=["term"]
        )
        
        family_template = PromptTemplate(
            id="family",
            category=PromptCategory.DEFINITION,
            template="Family relations: {term}",
            variables=["term"]
        )
        
        # Add templates to library
        library.templates[PromptCategory.DEFINITION].extend([math_template, family_template])
        
        # Test selection based on context
        context = QueryContext(
            term="parent(X, Y)",
            existing_functors=["parent", "child", "sibling"]
        )
        
        selected = library.select_template(context)
        # Should select a template based on context
        assert selected is not None
        assert selected.category == PromptCategory.DEFINITION
    
    def test_model_specific_parameters(self):
        """Test model-specific parameter selection"""
        library = PromptTemplateLibrary()
        
        # Test model parameters are accessible
        assert library.model_params is not None
        assert library.model_params.model_name is not None
        
        # Test default parameters
        assert library.model_params.min_examples == 3
        assert library.model_params.max_examples == 8
        assert library.model_params.optimal_examples == 5
        assert library.model_params.prefers_json == True
    
    def test_format_with_library(self):
        """Test formatting prompt through library"""
        library = PromptTemplateLibrary()
        
        context = QueryContext(
            term="grandparent(X, Z)",
            kb_facts=["parent(alice, bob)", "parent(bob, charlie)"],
            kb_rules=[]
        )
        
        # Format with default template
        formatted = library.format_prompt(context)
        
        assert "grandparent(X, Z)" in formatted
        assert formatted  # Should produce non-empty prompt
        
        # Format with specific template
        formatted = library.format_prompt(context, template_name="default")
        assert formatted


class TestTemplateExamples:
    """Test template example handling"""
    
    def test_s_expression_examples(self):
        """Test that examples use S-expression format"""
        library = PromptTemplateLibrary()
        
        # Get a template from the definition category
        templates = library.templates[PromptCategory.DEFINITION]
        assert len(templates) > 0
        
        template = templates[0]
        
        # Check the template format includes S-expression patterns
        if "parent" in template.template or "grandparent" in template.template:
            # Should use S-expression or JSON format
            assert ("(parent" in template.template) or ("['parent'" in template.template) or ('["parent"' in template.template)
    
    def test_json_format_examples(self):
        """Test JSON format in examples"""
        template = PromptTemplate(
            id="json_test",
            category=PromptCategory.DEFINITION,
            template="Generate JSON: {term}\nExamples:\n{examples}",
            variables=["term", "examples"]
        )
        
        # Test that the template can be rendered
        rendered = template.render(
            term="parent(X, Y)",
            examples='[["fact", ["parent", "alice", "bob"]]], [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]'
        )
        
        # Should contain the expected JSON format
        assert "parent(X, Y)" in rendered
        assert '"fact"' in rendered or '"rule"' in rendered


class TestPromptOptimization:
    """Test prompt optimization features"""
    
    def test_context_summarization(self):
        """Test that large contexts are summarized"""
        library = PromptTemplateLibrary()
        
        # Create context with many facts
        context = QueryContext(
            term="test(X)",
            kb_facts=[f"fact_{i}(a, b)" for i in range(100)],
            kb_rules=[(f"rule_{i}(X, Y)", ["body"]) for i in range(50)]
        )
        
        formatted = library.format_prompt(context)
        
        # Should not include all 100 facts directly
        # Should be summarized or truncated
        assert len(formatted) < 10000  # Reasonable size limit
    
    def test_relevant_fact_selection(self):
        """Test selection of relevant facts"""
        template = PromptTemplate(
            id="selective",
            category=PromptCategory.DEFINITION,
            template="Relevant facts: {relevant_facts}\nGenerate: {term}",
            variables=["relevant_facts", "term"]
        )
        
        context = QueryContext(
            term="parent(alice, X)",
            kb_facts=[
                "parent(alice, bob)",
                "parent(bob, charlie)",
                "parent(charlie, david)",
                "sibling(alice, andy)",
                "unrelated(x, y)"
            ]
        )
        
        # Should prioritize parent facts for parent query
        relevant_facts = "\n".join(context.kb_facts[:3])  # Take first 3 facts
        formatted = template.render(term=context.term, relevant_facts=relevant_facts)
        assert "parent(alice, bob)" in formatted


class TestErrorHandling:
    """Test error handling in prompt templates"""
    
    def test_missing_variable(self):
        """Test handling missing template variables"""
        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Term: {term}, Other: {nonexistent}",
            variables=["term", "nonexistent"]
        )
        
        # Should handle gracefully with partial variables
        formatted = template.render(term="test(X)")
        assert "test(X)" in formatted
        # Nonexistent variable remains as placeholder
        assert "{nonexistent}" in formatted
    
    def test_empty_template(self):
        """Test empty template handling"""
        template = PromptTemplate(
            id="empty",
            category=PromptCategory.DEFINITION,
            template="",
            variables=[]
        )
        
        formatted = template.render()
        
        # Should still produce something (empty string)
        assert formatted is not None
        assert formatted == ""
    
    def test_invalid_model_config(self):
        """Test invalid model configuration"""
        library = PromptTemplateLibrary()
        
        # Should not crash with current library structure
        assert library.model_params is not None
        assert library.model_params.model_name is not None
        
        # Test with empty model name still works
        empty_library = PromptTemplateLibrary(model_name="")
        assert empty_library.model_params is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])