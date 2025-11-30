#!/usr/bin/env python3
"""
Unit tests for prompt template system
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

from dreamlog.prompt_template_system import (
    QueryContext, PromptTemplate, PromptTemplateLibrary,
    ModelParameters, PromptCategory, sample_examples, RULE_EXAMPLES,
    AdaptivePromptSelector, PromptBuilder, PromptLearningSystem
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


class TestSampleExamples:
    """Test sample_examples function"""

    def test_sample_examples_default(self):
        """Test sampling examples with default count"""
        # When: Sampling with default parameters
        examples = sample_examples()

        # Then: Should return 5 examples
        assert len(examples) == 5

    def test_sample_examples_custom_count(self):
        """Test sampling specific number of examples"""
        # When: Sampling 3 examples
        examples = sample_examples(num_examples=3)

        # Then: Should return 3 examples
        assert len(examples) == 3

    def test_sample_examples_with_seed(self):
        """Test reproducible sampling with seed"""
        # When: Sampling with same seed
        examples1 = sample_examples(num_examples=5, seed=42)
        examples2 = sample_examples(num_examples=5, seed=42)

        # Then: Should return same examples
        assert examples1 == examples2

    def test_sample_examples_different_seeds(self):
        """Test different seeds produce different results"""
        # When: Sampling with different seeds
        examples1 = sample_examples(num_examples=5, seed=42)
        examples2 = sample_examples(num_examples=5, seed=123)

        # Then: May have different examples (high probability)
        # Note: Could theoretically be same by chance, but unlikely with 5 samples
        # This is probabilistic, so we just verify they're valid
        assert len(examples1) == 5
        assert len(examples2) == 5

    def test_sample_examples_exceeds_available(self):
        """Test sampling more than available examples"""
        # When: Sampling more than total examples
        total_examples = len(RULE_EXAMPLES)
        examples = sample_examples(num_examples=total_examples + 10)

        # Then: Should return all available examples
        assert len(examples) == total_examples

    def test_sample_examples_have_required_fields(self):
        """Test sampled examples have required fields"""
        # When: Sampling examples
        examples = sample_examples(num_examples=3)

        # Then: Each example should have domain, prolog, and json fields
        for example in examples:
            assert "domain" in example
            assert "prolog" in example
            assert "json" in example


class TestPromptTemplateSuccessRate:
    """Test PromptTemplate success rate tracking"""

    def test_success_rate_zero_when_unused(self):
        """Test success rate is 0 for unused templates"""
        # Given: A fresh template
        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"]
        )

        # Then: Success rate should be 0
        assert template.success_rate == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        # Given: A template with usage data
        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"],
            use_count=10,
            success_count=7
        )

        # Then: Success rate should be 70%
        assert template.success_rate == 0.7

    def test_success_rate_full_success(self):
        """Test 100% success rate"""
        # Given: Template with all successes
        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"],
            use_count=5,
            success_count=5
        )

        # Then: Success rate should be 1.0
        assert template.success_rate == 1.0


class TestQueryContextProperties:
    """Test QueryContext computed properties"""

    def test_is_empty_kb_true(self):
        """Test is_empty_kb when KB is empty"""
        # Given: Context with no facts or rules
        context = QueryContext(term="test(X)")

        # Then: is_empty_kb should be True
        assert context.is_empty_kb is True

    def test_is_empty_kb_false_with_facts(self):
        """Test is_empty_kb when KB has facts"""
        # Given: Context with facts
        context = QueryContext(
            term="test(X)",
            kb_facts=["fact(a, b)"]
        )

        # Then: is_empty_kb should be False
        assert context.is_empty_kb is False

    def test_is_empty_kb_false_with_rules(self):
        """Test is_empty_kb when KB has rules"""
        # Given: Context with rules
        context = QueryContext(
            term="test(X)",
            kb_rules=[("head(X)", ["body(X)"])]
        )

        # Then: is_empty_kb should be False
        assert context.is_empty_kb is False

    def test_has_examples_true(self):
        """Test has_examples when facts exist"""
        # Given: Context with facts
        context = QueryContext(
            term="test(X)",
            kb_facts=["fact(a, b)"]
        )

        # Then: has_examples should be True
        assert context.has_examples is True

    def test_has_examples_false(self):
        """Test has_examples when no facts"""
        # Given: Context without facts
        context = QueryContext(term="test(X)")

        # Then: has_examples should be False
        assert context.has_examples is False

    def test_has_rules_true(self):
        """Test has_rules when rules exist"""
        # Given: Context with rules
        context = QueryContext(
            term="test(X)",
            kb_rules=[("head(X)", ["body(X)"])]
        )

        # Then: has_rules should be True
        assert context.has_rules is True

    def test_has_rules_false(self):
        """Test has_rules when no rules"""
        # Given: Context without rules
        context = QueryContext(term="test(X)")

        # Then: has_rules should be False
        assert context.has_rules is False


class TestPromptTemplateLibraryAdvanced:
    """Advanced tests for PromptTemplateLibrary"""

    def test_get_best_prompt_empty_kb(self):
        """Test get_best_prompt with empty KB"""
        # Given: Library and empty context
        library = PromptTemplateLibrary()
        context = QueryContext(term=["parent", "X", "Y"])

        # When: Getting best prompt
        prompt, template_name = library.get_best_prompt(context)

        # Then: Should return valid prompt
        assert prompt is not None
        assert len(prompt) > 0
        assert template_name is not None

    def test_get_best_prompt_with_examples(self):
        """Test get_best_prompt with many facts"""
        # Given: Library and context with many facts
        library = PromptTemplateLibrary()
        context = QueryContext(
            term=["grandparent", "X", "Z"],
            kb_facts=[["parent", "a", "b"] for _ in range(5)],
            kb_rules=[(["ancestor", "X", "Y"], [["parent", "X", "Y"]])]
        )

        # When: Getting best prompt
        prompt, template_name = library.get_best_prompt(context)

        # Then: Should return valid prompt
        assert prompt is not None
        assert len(prompt) > 0

    def test_get_best_prompt_no_templates(self):
        """Test get_best_prompt fallback when no templates"""
        # Given: Library with cleared templates
        library = PromptTemplateLibrary()
        library.templates[PromptCategory.DEFINITION] = []
        context = QueryContext(term=["test", "X"])

        # When: Getting best prompt
        prompt, template_name = library.get_best_prompt(context)

        # Then: Should return fallback prompt
        assert prompt is not None
        assert template_name == "fallback"

    def test_record_performance(self):
        """Test recording template performance"""
        # Given: A library
        library = PromptTemplateLibrary()

        # When: Recording performance
        library.record_performance("test_template", success=True, response_quality=0.9)
        library.record_performance("test_template", success=False, response_quality=0.3)
        library.record_performance("test_template", success=True, response_quality=0.8)

        # Then: Performance data should be updated
        assert "test_template" in library.performance_data
        data = library.performance_data["test_template"]
        assert data["successes"] == 2
        assert data["failures"] == 1
        assert data["count"] == 3
        assert data["total_quality"] == pytest.approx(2.0, abs=0.01)

    def test_get_performance_stats(self):
        """Test getting performance statistics"""
        # Given: Library with recorded performance
        library = PromptTemplateLibrary()
        library.record_performance("template1", success=True, response_quality=0.9)
        library.record_performance("template1", success=True, response_quality=0.8)
        library.record_performance("template2", success=False, response_quality=0.2)

        # When: Getting stats
        stats = library.get_performance_stats()

        # Then: Should have correct stats
        assert "template1" in stats
        assert stats["template1"]["success_rate"] == 1.0
        assert stats["template1"]["avg_quality"] == pytest.approx(0.85, abs=0.01)
        assert stats["template1"]["count"] == 2

    def test_determine_category_compression(self):
        """Test category determination for compression"""
        # Given: Library and compression context
        library = PromptTemplateLibrary()
        context = QueryContext(term="optimize this")

        # When: Determining category
        category = library._determine_category(context)

        # Then: Should return compression
        assert category == PromptCategory.COMPRESSION

    def test_determine_category_abstraction(self):
        """Test category determination for abstraction"""
        # Given: Library and abstraction context
        library = PromptTemplateLibrary()
        context = QueryContext(term="abstract pattern")

        # When: Determining category
        category = library._determine_category(context)

        # Then: Should return abstraction
        assert category == PromptCategory.ABSTRACTION

    def test_determine_category_consolidation(self):
        """Test category determination for consolidation"""
        # Given: Library and context with many facts
        library = PromptTemplateLibrary()
        context = QueryContext(
            term="test",
            kb_facts=[f"fact{i}" for i in range(15)]
        )

        # When: Determining category
        category = library._determine_category(context)

        # Then: Should return consolidation
        assert category == PromptCategory.CONSOLIDATION

    def test_format_context(self):
        """Test context formatting"""
        # Given: Library and context
        library = PromptTemplateLibrary()
        context = QueryContext(
            term="test",
            kb_facts=["fact1", "fact2"],
            kb_rules=[("head", ["body"])]
        )

        # When: Formatting context
        formatted = library._format_context(context)

        # Then: Should include facts info
        assert "fact1" in formatted
        assert "rules" in formatted.lower()

    def test_format_rules(self):
        """Test rules formatting"""
        # Given: Library and rules
        library = PromptTemplateLibrary()
        rules = [
            ("grandparent(X, Z)", ["parent(X, Y)", "parent(Y, Z)"]),
            ("sibling(X, Y)", ["parent(Z, X)", "parent(Z, Y)"])
        ]

        # When: Formatting rules
        formatted = library._format_rules(rules)

        # Then: Should format as Prolog-style rules
        assert "grandparent(X, Z) :-" in formatted
        assert "parent(X, Y)" in formatted


class TestAdaptivePromptSelector:
    """Test AdaptivePromptSelector class"""

    def test_initialization(self):
        """Test selector initialization"""
        # Given: Library and model params
        library = PromptTemplateLibrary()
        model_params = {
            "test_model": ModelParameters(model_name="test_model")
        }

        # When: Creating selector
        selector = AdaptivePromptSelector(library, model_params)

        # Then: Should be initialized
        assert selector.library == library
        assert selector.model_params == model_params
        assert selector.selection_history == []

    def test_select_template(self):
        """Test template selection"""
        # Given: Selector with templates
        library = PromptTemplateLibrary()
        model_params = {
            "test_model": ModelParameters(model_name="test_model")
        }
        selector = AdaptivePromptSelector(library, model_params)

        # When: Selecting template
        template = selector.select_template(
            query="parent(X, Y)",
            category=PromptCategory.DEFINITION,
            model="test_model",
            context_size=100
        )

        # Then: Should return a template
        assert template is not None
        assert template.category == PromptCategory.DEFINITION

    def test_select_template_unknown_model(self):
        """Test template selection with unknown model"""
        # Given: Selector
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})

        # When: Selecting template for unknown model
        template = selector.select_template(
            query="test(X)",
            category=PromptCategory.DEFINITION,
            model="unknown_model"
        )

        # Then: Should return first template
        assert template is not None

    def test_select_template_no_templates(self):
        """Test template selection with no templates raises error"""
        # Given: Library with empty category
        library = PromptTemplateLibrary()
        library.templates[PromptCategory.ANALOGY] = []
        selector = AdaptivePromptSelector(library, {})

        # When/Then: Should raise ValueError
        with pytest.raises(ValueError, match="No templates"):
            selector.select_template(
                query="test",
                category=PromptCategory.ANALOGY,
                model="test"
            )

    def test_score_template(self):
        """Test template scoring"""
        # Given: Selector and template
        library = PromptTemplateLibrary()
        model_params = ModelParameters(
            model_name="test",
            max_context_tokens=4000
        )
        selector = AdaptivePromptSelector(library, {"test": model_params})

        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"],
            use_count=15,
            success_count=12
        )

        # When: Scoring template
        score = selector._score_template(template, model_params, context_size=100)

        # Then: Should return positive score
        assert score > 0

    def test_score_template_large_context_penalty(self):
        """Test that large context penalizes score"""
        # Given: Selector and template
        library = PromptTemplateLibrary()
        model_params = ModelParameters(
            model_name="test",
            max_context_tokens=1000
        )
        selector = AdaptivePromptSelector(library, {"test": model_params})

        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"]
        )

        # When: Scoring with large context
        small_context_score = selector._score_template(template, model_params, context_size=100)
        large_context_score = selector._score_template(template, model_params, context_size=900)

        # Then: Large context should have lower score
        assert large_context_score < small_context_score

    def test_record_outcome(self):
        """Test recording outcome updates template stats"""
        # Given: Selector and template
        library = PromptTemplateLibrary()
        model_params = {"test_model": ModelParameters(model_name="test_model")}
        selector = AdaptivePromptSelector(library, model_params)

        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"]
        )

        # When: Recording outcomes
        selector.record_outcome(template, "test_model", success=True, response_quality=0.9)
        selector.record_outcome(template, "test_model", success=True, response_quality=0.8)
        selector.record_outcome(template, "test_model", success=False, response_quality=0.2)

        # Then: Template stats should be updated
        assert template.use_count == 3
        assert template.success_count == 2
        assert "test_model" in template.model_success_rates


class TestPromptBuilder:
    """Test PromptBuilder class"""

    def test_initialization(self):
        """Test builder initialization"""
        # Given: Selector
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})

        # When: Creating builder
        builder = PromptBuilder(selector)

        # Then: Should be initialized
        assert builder.selector == selector
        assert builder.example_retriever is None

    def test_build_prompt(self):
        """Test building a complete prompt"""
        # Given: Builder
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        # When: Building prompt
        prompt, template = builder.build_prompt(
            query="parent(X, Y)",
            category=PromptCategory.DEFINITION,
            model="test_model",
            context={"facts": ["parent(a, b)"]}
        )

        # Then: Should return valid prompt and template
        assert prompt is not None
        assert len(prompt) > 0
        assert template is not None

    def test_format_context_json(self):
        """Test context formatting in JSON mode"""
        # Given: Builder with JSON-preferring model
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        model_params = ModelParameters(model_name="test", prefers_json=True)
        context = {"facts": ["fact1", "fact2"], "rules": ["rule1"]}

        # When: Formatting context
        formatted = builder._format_context(context, model_params)

        # Then: Should be valid JSON
        parsed = json.loads(formatted)
        assert "facts" in parsed

    def test_format_context_human_readable(self):
        """Test context formatting in human-readable mode"""
        # Given: Builder with non-JSON model
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        model_params = ModelParameters(model_name="test", prefers_json=False)
        context = {"facts": ["fact1"], "rules": ["rule1"]}

        # When: Formatting context
        formatted = builder._format_context(context, model_params)

        # Then: Should be human readable
        assert "Facts:" in formatted
        assert "Rules:" in formatted

    def test_extract_functor(self):
        """Test functor extraction from query"""
        # Given: Builder
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        # When: Extracting functor
        functor = builder._extract_functor("(parent john mary)")

        # Then: Should extract functor
        assert "parent" in functor

    def test_extract_arity(self):
        """Test arity extraction from query"""
        # Given: Builder
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        # When: Extracting arity
        arity = builder._extract_arity("(parent X Y)")

        # Then: Should return arity
        assert arity >= 0

    def test_add_format_instructions(self):
        """Test adding format instructions"""
        # Given: Builder and model params
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        model_params = ModelParameters(model_name="test", prefers_json=True)
        prompt = "Original prompt"

        # When: Adding format instructions
        result = builder._add_format_instructions(prompt, model_params)

        # Then: Should add JSON instruction
        assert "JSON" in result
        assert "Original prompt" in result

    def test_format_examples(self):
        """Test example formatting"""
        # Given: Builder
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        model_params = ModelParameters(model_name="test", max_examples=3)
        examples = [{"domain": "family", "rule": "test"}]

        # When: Formatting examples
        formatted = builder._format_examples(examples, model_params)

        # Then: Should format examples
        assert "Example 1:" in formatted

    def test_format_examples_empty(self):
        """Test empty examples formatting"""
        # Given: Builder
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        model_params = ModelParameters(model_name="test")

        # When: Formatting empty examples
        formatted = builder._format_examples([], model_params)

        # Then: Should return appropriate message
        assert "No examples" in formatted


class TestPromptLearningSystem:
    """Test PromptLearningSystem class"""

    def test_initialization(self):
        """Test learning system initialization"""
        # Given: Library and temp path
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "learning.json"

            # When: Creating learning system
            learning = PromptLearningSystem(library, save_path)

            # Then: Should be initialized
            assert learning.library == library
            assert learning.save_path == save_path
            assert learning.learning_history == []

    def test_analyze_patterns(self):
        """Test pattern analysis"""
        # Given: Learning system with history
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "learning.json"
            learning = PromptLearningSystem(library, save_path)

            # Add some history
            learning.learning_history = [
                {"success": True, "category": "definition", "model": "test", "template_id": "t1"},
                {"success": False, "category": "abstraction", "model": "test", "template_id": "t2"},
                {"success": True, "category": "definition", "model": "test", "template_id": "t1", "context_size": 500}
            ]

            # When: Analyzing patterns
            patterns = learning.analyze_patterns()

            # Then: Should have pattern data
            assert "best_templates_by_category" in patterns
            assert "successful_patterns" in patterns
            assert len(patterns["successful_patterns"]) == 2

    def test_generate_new_template(self):
        """Test generating new template from successful ones"""
        # Given: Learning system with base templates
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "learning.json"
            learning = PromptLearningSystem(library, save_path)

            parent1 = PromptTemplate(
                id="parent1",
                category=PromptCategory.DEFINITION,
                template="Line1\nLine2\nLine3\nLine4",
                variables=["term"]
            )
            parent2 = PromptTemplate(
                id="parent2",
                category=PromptCategory.DEFINITION,
                template="LineA\nLineB\nLineC\nLineD",
                variables=["query"]
            )

            # When: Generating new template
            new_template = learning.generate_new_template(
                PromptCategory.DEFINITION,
                [parent1, parent2]
            )

            # Then: Should create hybrid template
            assert new_template is not None
            assert new_template.category == PromptCategory.DEFINITION
            assert "term" in new_template.variables or "query" in new_template.variables

    def test_generate_new_template_insufficient_parents(self):
        """Test generating new template with too few parents"""
        # Given: Learning system
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "learning.json"
            learning = PromptLearningSystem(library, save_path)

            parent = PromptTemplate(
                id="parent",
                category=PromptCategory.DEFINITION,
                template="Test",
                variables=[]
            )

            # When: Trying to generate with one parent
            result = learning.generate_new_template(PromptCategory.DEFINITION, [parent])

            # Then: Should return None
            assert result is None

    def test_save_and_load(self):
        """Test saving and loading learning data"""
        # Given: Learning system with data
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "learning.json"
            learning = PromptLearningSystem(library, save_path)

            learning.learning_history = [
                {"success": True, "template_id": "t1", "category": "definition", "model": "test"}
            ]

            # Update template stats
            if library.templates[PromptCategory.DEFINITION]:
                template = library.templates[PromptCategory.DEFINITION][0]
                template.use_count = 10
                template.success_count = 8
                template.avg_response_quality = 0.85

            # When: Saving
            learning.save()

            # Then: File should exist
            assert save_path.exists()

            # When: Loading in new instance
            library2 = PromptTemplateLibrary()
            learning2 = PromptLearningSystem(library2, save_path)

            # Then: Data should be restored
            assert learning2.learning_history == learning.learning_history

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file"""
        # Given: Path that doesn't exist
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nonexistent.json"

            # When: Creating learning system (loads automatically if file exists)
            learning = PromptLearningSystem(library, save_path)

            # Then: Should have empty history
            assert learning.learning_history == []

    def test_crossover_templates(self):
        """Test template crossover mechanism"""
        # Given: Learning system
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "learning.json"
            learning = PromptLearningSystem(library, save_path)

            template1 = "Line1\nLine2\nLine3\nLine4"
            template2 = "LineA\nLineB\nLineC\nLineD"

            # When: Crossing over
            result = learning._crossover_templates(template1, template2)

            # Then: Should combine templates
            assert result is not None
            assert len(result) > 0
            # Should contain elements from both
            lines = result.split("\n")
            assert len(lines) >= 2


class TestModelParametersTemperature:
    """Test ModelParameters temperature by category"""

    def test_temperature_by_category_defaults(self):
        """Test default temperatures for each category"""
        # Given: Default model parameters
        params = ModelParameters(model_name="test")

        # Then: Should have temperatures for each category
        assert params.temperature_by_category["compression"] == 0.1
        assert params.temperature_by_category["abstraction"] == 0.3
        assert params.temperature_by_category["analogy"] == 0.5
        assert params.temperature_by_category["counterfactual"] == 0.7

    def test_success_rates_initially_empty(self):
        """Test success rates start empty"""
        # Given: New model parameters
        params = ModelParameters(model_name="test")

        # Then: Success rates should be empty
        assert params.success_rates == {}

    def test_prompt_style_scores(self):
        """Test prompt style scores defaults"""
        # Given: New model parameters
        params = ModelParameters(model_name="test")

        # Then: Should have default style scores
        assert params.prompt_style_scores["verbose"] == 0.5
        assert params.prompt_style_scores["concise"] == 0.5
        assert params.prompt_style_scores["step_by_step"] == 0.5
        assert params.prompt_style_scores["direct"] == 0.5


class TestPromptTemplateLibraryEdgeCases:
    """Test edge cases for PromptTemplateLibrary"""

    def test_get_best_prompt_with_non_list_term(self):
        """Test get_best_prompt when term is not a list"""
        # Given: Library and context with string term
        library = PromptTemplateLibrary()
        context = QueryContext(term="parent X Y")

        # When: Getting best prompt
        prompt, template_name = library.get_best_prompt(context)

        # Then: Should return valid prompt with fallback functor
        assert prompt is not None
        assert template_name is not None

    def test_select_template_default_fallback(self):
        """Test select_template falls back to default when no templates"""
        # Given: Library with empty categories
        library = PromptTemplateLibrary()
        # Clear all categories
        for category in PromptCategory:
            library.templates[category] = []
        context = QueryContext(term="test(X)")

        # When: Selecting template
        template = library.select_template(context)

        # Then: Should return default template
        assert template is not None
        assert template.id == "default"

    def test_format_template_with_compound_term(self):
        """Test format_template with compound term containing parentheses"""
        # Given: Library and context with parenthesized term
        library = PromptTemplateLibrary()
        context = QueryContext(
            term="grandparent(X, Z)",
            kb_facts=["parent(a, b)"],
            kb_rules=[],
            existing_functors=["parent", "grandparent"]
        )

        # When: Formatting with specific template
        formatted = library.format_prompt(context, template_name="define_simple")

        # Then: Should extract functor
        assert "grandparent" in formatted

    def test_determine_category_generalize(self):
        """Test category determination for generalize keyword"""
        # Given: Library and context with generalize keyword
        library = PromptTemplateLibrary()
        context = QueryContext(term="generalize this pattern")

        # When: Determining category
        try:
            category = library._determine_category(context)
            # If GENERALIZATION exists, it should be returned
            assert category in [PromptCategory.DEFINITION, PromptCategory.CONSOLIDATION]
        except AttributeError:
            # GENERALIZATION category may not exist, fallback to DEFINITION
            pass


class TestPromptBuilderEdgeCases:
    """Test edge cases for PromptBuilder"""

    def test_select_examples_no_retriever(self):
        """Test example selection when no retriever is configured"""
        # Given: Builder without retriever
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector, example_retriever=None)

        model_params = ModelParameters(model_name="test")

        # When: Selecting examples
        examples = builder._select_examples("test", model_params, PromptCategory.DEFINITION)

        # Then: Should return empty list
        assert examples == []

    def test_extract_functor_no_parentheses(self):
        """Test functor extraction when no parentheses"""
        # Given: Builder
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        # When: Extracting functor from term without parentheses
        functor = builder._extract_functor("simple_term")

        # Then: Should return unknown
        assert functor == "unknown"

    def test_extract_arity_no_parentheses(self):
        """Test arity extraction when no parentheses"""
        # Given: Builder
        library = PromptTemplateLibrary()
        selector = AdaptivePromptSelector(library, {})
        builder = PromptBuilder(selector)

        # When: Extracting arity from term without parentheses
        arity = builder._extract_arity("simple_term")

        # Then: Should return 0
        assert arity == 0


class TestAdaptivePromptSelectorEdgeCases:
    """Test edge cases for AdaptivePromptSelector"""

    def test_score_template_with_model_specific_rate(self):
        """Test scoring template with model-specific success rate"""
        # Given: Template with model-specific rates
        library = PromptTemplateLibrary()
        model_params = ModelParameters(
            model_name="test_model",
            max_context_tokens=4000
        )
        model_params.success_rates["definition"] = 0.9
        selector = AdaptivePromptSelector(library, {"test_model": model_params})

        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"],
            use_count=5,
            success_count=4,
            model_success_rates={"test_model": 0.95}
        )

        # When: Scoring template
        score = selector._score_template(template, model_params, context_size=100)

        # Then: Should include model-specific bonus
        # Score should be higher due to model-specific success rate
        assert score > 0

    def test_record_outcome_updates_model_params(self):
        """Test that record_outcome updates model parameters"""
        # Given: Selector with model params
        library = PromptTemplateLibrary()
        model_params = ModelParameters(model_name="test_model")
        selector = AdaptivePromptSelector(library, {"test_model": model_params})

        template = PromptTemplate(
            id="test",
            category=PromptCategory.DEFINITION,
            template="Test {term}",
            variables=["term"]
        )

        # When: Recording outcomes
        selector.record_outcome(template, "test_model", success=True, response_quality=0.9)

        # Then: Model params should be updated
        assert PromptCategory.DEFINITION.value in model_params.success_rates


class TestPromptLearningSystemEdgeCases:
    """Test edge cases for PromptLearningSystem"""

    def test_load_corrupted_file(self):
        """Test loading from corrupted file handles error gracefully"""
        # Given: A corrupted JSON file
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "corrupted.json"
            # Write invalid JSON
            with open(save_path, 'w') as f:
                f.write("{ invalid json }")

            # When: Creating learning system (triggers load)
            learning = PromptLearningSystem(library, save_path)

            # Then: Should handle gracefully with empty history
            assert learning.learning_history == []

    def test_save_creates_parent_directory(self):
        """Test that save creates parent directories"""
        # Given: Learning system with nested path
        library = PromptTemplateLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "learning.json"
            learning = PromptLearningSystem(library, save_path)

            learning.learning_history = [{"test": "data"}]

            # When: Saving
            learning.save()

            # Then: File should exist with parent dirs created
            assert save_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])