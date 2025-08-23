"""
Prompt Template System for DreamLog

A flexible, parameterized prompt template system that can be tuned per LLM model.
Supports categories of prompts and learns what works best for different query types.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from pathlib import Path


class PromptCategory(Enum):
    """Categories of prompt templates for different reasoning tasks"""
    COMPRESSION = "compression"
    ABSTRACTION = "abstraction"
    ANALOGY = "analogy"
    COUNTERFACTUAL = "counterfactual"
    DECOMPOSITION = "decomposition"
    BRIDGE = "bridge"
    DEFINITION = "definition"  # For undefined predicates
    EXAMPLE_GENERATION = "example_generation"


@dataclass
class QueryContext:
    """Context for a query to help select appropriate template"""
    term: str
    kb_facts: List[str] = field(default_factory=list)
    kb_rules: List[Tuple[str, List[str]]] = field(default_factory=list)
    existing_functors: List[str] = field(default_factory=list)
    
    @property
    def is_empty_kb(self) -> bool:
        return len(self.kb_facts) == 0 and len(self.kb_rules) == 0
    
    @property
    def has_examples(self) -> bool:
        return len(self.kb_facts) > 0
    
    @property
    def has_rules(self) -> bool:
        return len(self.kb_rules) > 0


@dataclass
class ModelParameters:
    """Parameters that can be tuned per LLM model"""
    model_name: str
    
    # Many-shot learning parameters
    min_examples: int = 3
    max_examples: int = 8
    optimal_examples: int = 5  # Learned over time
    
    # Context window management
    max_context_tokens: int = 4000
    example_selection_strategy: str = "similarity"  # similarity, diversity, mixed
    
    # Response format preferences
    prefers_json: bool = True
    needs_explicit_format: bool = True
    handles_complex_reasoning: bool = True
    
    # Temperature for different tasks
    temperature_by_category: Dict[str, float] = field(default_factory=lambda: {
        "compression": 0.1,
        "abstraction": 0.3,
        "analogy": 0.5,
        "counterfactual": 0.7,
        "decomposition": 0.2,
        "bridge": 0.4,
        "definition": 0.1,
        "example_generation": 0.3
    })
    
    # Success rates by category (learned)
    success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Prompt style preferences (learned)
    prompt_style_scores: Dict[str, float] = field(default_factory=lambda: {
        "verbose": 0.5,
        "concise": 0.5,
        "step_by_step": 0.5,
        "direct": 0.5
    })


@dataclass
class PromptTemplate:
    """A single prompt template"""
    id: str
    category: PromptCategory
    template: str
    variables: List[str]  # Variables to fill in
    
    # Performance tracking
    use_count: int = 0
    success_count: int = 0
    avg_response_quality: float = 0.0
    
    # Model-specific success rates
    model_success_rates: Dict[str, float] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render the template with the provided variables"""
        result = self.template
        for var in self.variables:
            if var in kwargs:
                # Simple replacement - could be more sophisticated
                result = result.replace(f"{{{var}}}", str(kwargs[var]))
        return result
    
    @property
    def success_rate(self) -> float:
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count


class PromptTemplateLibrary:
    """Library of prompt templates organized by category"""
    
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.model_params = ModelParameters(model_name=model_name)
        self.templates: Dict[PromptCategory, List[PromptTemplate]] = {
            category: [] for category in PromptCategory
        }
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize with starter templates for each category"""
        
        # COMPRESSION templates
        self.templates[PromptCategory.COMPRESSION].extend([
            PromptTemplate(
                id="compress_patterns",
                category=PromptCategory.COMPRESSION,
                template="""Find patterns in these facts that could be expressed as a single rule:
{facts}

Return a JSON array with the compressed rule in S-expression format:
[["rule", [head], [body]]]""",
                variables=["facts"]
            ),
            PromptTemplate(
                id="compress_redundant",
                category=PromptCategory.COMPRESSION,
                template="""These rules seem redundant:
{rules}

Can you merge them into a more general rule?
Output format: [["rule", [head], [body]]]""",
                variables=["rules"]
            )
        ])
        
        # DEFINITION templates (for undefined predicates)
        self.templates[PromptCategory.DEFINITION].extend([
            PromptTemplate(
                id="define_simple",
                category=PromptCategory.DEFINITION,
                template="""Query: {query}
Context: {context}

Define the predicate '{functor}' with {arity} arguments.

Output examples (return ONLY JSON in this exact format):
- Fact: [["fact", ["parent", "alice", "bob"]]]
- Rule: [["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
- Rule with multiple conditions: [["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"], ["different", "X", "Y"]]]]

Your output (JSON array only, no other text):""",
                variables=["query", "context", "functor", "arity"]
            ),
            PromptTemplate(
                id="define_with_examples",
                category=PromptCategory.DEFINITION,
                template="""Query: {query}
Context: {context}

Examples from knowledge base:
{examples}

Define '{functor}' following similar patterns.

Output format examples:
[["rule", ["grandparent", "X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]]]]
[["rule", ["sibling", "X", "Y"], [["parent", "Z", "X"], ["parent", "Z", "Y"]]]]
[["fact", ["parent", "john", "mary"]]]

Return ONLY the JSON array:""",
                variables=["query", "context", "functor", "examples"]
            ),
            PromptTemplate(
                id="define_step_by_step",
                category=PromptCategory.DEFINITION,
                template="""Let's define '{functor}' step by step.

1. The query is: {query}
2. Existing knowledge: {context}
3. This predicate likely means: {hint}

Provide the definition as JSON: [["rule", [head], [body]]]""",
                variables=["query", "context", "functor", "hint"]
            )
        ])
        
        # ABSTRACTION templates
        self.templates[PromptCategory.ABSTRACTION].extend([
            PromptTemplate(
                id="find_abstraction",
                category=PromptCategory.ABSTRACTION,
                template="""What higher-level concept explains these patterns?
{patterns}

Create an abstract rule that captures the essence.
Output: [["rule", [abstract_predicate, vars...], [conditions...]]]""",
                variables=["patterns"]
            )
        ])
        
        # EXAMPLE_GENERATION templates
        self.templates[PromptCategory.EXAMPLE_GENERATION].extend([
            PromptTemplate(
                id="generate_examples",
                category=PromptCategory.EXAMPLE_GENERATION,
                template="""Given this rule: {rule}

Generate {num_examples} example facts that would match this rule.
Output: [["fact", [pred, args...]], ...]""",
                variables=["rule", "num_examples"]
            )
        ])
        
        # Initialize performance tracking (selector needs library reference, so skip for now)
        self.performance_data = {}
    
    def get_best_prompt(self, context: QueryContext) -> Tuple[str, str]:
        """
        Get the best prompt for the given context
        
        Returns:
            Tuple of (prompt_text, template_name)
        """
        # For now, use DEFINITION templates for queries
        # Later this can be more sophisticated based on context
        templates = self.templates[PromptCategory.DEFINITION]
        
        if not templates:
            # Fallback prompt
            prompt = f"""Query: {context.term}
Knowledge base has {len(context.kb_facts)} facts and {len(context.kb_rules)} rules.

Please generate relevant facts and rules for this query.
Output JSON: [["fact", [pred, args...]], ["rule", [head], [body]]]"""
            return prompt, "fallback"
        
        # Choose template based on context
        if context.is_empty_kb:
            template = templates[0]  # Simple definition
        elif context.has_examples and len(context.kb_facts) > 3:
            # Use template with examples if we have them
            template = next((t for t in templates if "examples" in t.id), templates[0])
        else:
            template = templates[0]
        
        # Build prompt from template
        variables = {}
        if "query" in template.variables:
            variables["query"] = context.term
        if "context" in template.variables:
            # Build context string
            ctx_parts = []
            if context.kb_facts:
                ctx_parts.append(f"Facts: {', '.join(context.kb_facts[:5])}")
            if context.kb_rules:
                rule_strs = [f"{h} :- {', '.join(b)}" for h, b in context.kb_rules[:3]]
                ctx_parts.append(f"Rules: {'; '.join(rule_strs)}")
            variables["context"] = "\n".join(ctx_parts) if ctx_parts else "Empty knowledge base"
        if "functor" in template.variables:
            # Extract functor from term
            term_str = context.term
            if term_str.startswith("("):
                # S-expression format: (functor arg1 arg2)
                parts = term_str[1:-1].split()
                functor = parts[0] if parts else "unknown"
            elif "(" in term_str:
                # Prolog format: functor(arg1, arg2)
                functor = term_str.split("(")[0]
            else:
                functor = term_str
            variables["functor"] = functor
        if "arity" in template.variables:
            # Count arguments
            term_str = context.term
            if term_str.startswith("("):
                # S-expression: (functor arg1 arg2)
                parts = term_str[1:-1].split()
                arity = len(parts) - 1 if len(parts) > 1 else 0
            elif "(" in term_str:
                # Prolog format: functor(arg1, arg2)
                args_str = term_str.split("(")[1].rstrip(")")
                if args_str:
                    # Count commas + 1, handling spaces
                    arity = args_str.count(",") + 1
                else:
                    arity = 0
            else:
                arity = 0
            variables["arity"] = arity
        if "examples" in template.variables:
            variables["examples"] = "\n".join(context.kb_facts[:3])
        
        prompt = template.render(**variables)
        return prompt, template.id
    
    def record_performance(self, template_name: str, success: bool, response_quality: float):
        """Record performance data for a template"""
        if template_name not in self.performance_data:
            self.performance_data[template_name] = {
                'successes': 0,
                'failures': 0,
                'total_quality': 0.0,
                'count': 0
            }
        
        data = self.performance_data[template_name]
        if success:
            data['successes'] += 1
        else:
            data['failures'] += 1
        data['total_quality'] += response_quality
        data['count'] += 1
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all templates"""
        stats = {}
        for name, data in self.performance_data.items():
            if data['count'] > 0:
                stats[name] = {
                    'success_rate': data['successes'] / data['count'],
                    'avg_quality': data['total_quality'] / data['count'],
                    'count': data['count']
                }
        return stats


class AdaptivePromptSelector:
    """
    Selects and adapts prompts based on:
    - Query type
    - Model being used
    - Historical performance
    - Current context
    """
    
    def __init__(self, 
                 library: PromptTemplateLibrary,
                 model_params: Dict[str, ModelParameters]):
        self.library = library
        self.model_params = model_params
        self.selection_history = []
    
    def select_template(self,
                        query: str,
                        category: PromptCategory,
                        model: str,
                        context_size: int = 0) -> PromptTemplate:
        """
        Select the best template for the given context.
        
        Args:
            query: The query being processed
            category: Category of reasoning needed
            model: Model name
            context_size: Current context size in tokens (approximate)
        """
        templates = self.library.templates[category]
        if not templates:
            raise ValueError(f"No templates for category {category}")
        
        model_param = self.model_params.get(model)
        if not model_param:
            # Use first template if model unknown
            return templates[0]
        
        # Score each template
        scores = []
        for template in templates:
            score = self._score_template(template, model_param, context_size)
            scores.append((template, score))
        
        # Use weighted random selection (exploration vs exploitation)
        if random.random() < 0.1:  # 10% exploration
            return random.choice(templates)
        else:
            # Select based on scores
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[0][0]
    
    def _score_template(self, 
                       template: PromptTemplate,
                       model_param: ModelParameters,
                       context_size: int) -> float:
        """Score a template for the given context"""
        score = 0.0
        
        # Base success rate
        if model_param.model_name in template.model_success_rates:
            score += template.model_success_rates[model_param.model_name] * 10
        else:
            score += template.success_rate * 5
        
        # Penalize if context too large
        if context_size > model_param.max_context_tokens * 0.8:
            score -= 5
        
        # Bonus for frequently successful templates
        if template.use_count > 10 and template.success_rate > 0.8:
            score += 3
        
        # Category-specific model performance
        category_name = template.category.value
        if category_name in model_param.success_rates:
            score += model_param.success_rates[category_name] * 2
        
        return score
    
    def record_outcome(self,
                       template: PromptTemplate,
                       model: str,
                       success: bool,
                       response_quality: float = 0.5):
        """
        Record the outcome of using a template.
        
        Args:
            template: The template that was used
            model: Model name
            success: Whether the query succeeded
            response_quality: Quality score (0-1)
        """
        template.use_count += 1
        if success:
            template.success_count += 1
        
        # Update running average of quality
        alpha = 0.1  # Learning rate
        template.avg_response_quality = (
            (1 - alpha) * template.avg_response_quality + 
            alpha * response_quality
        )
        
        # Update model-specific success rate
        if model not in template.model_success_rates:
            template.model_success_rates[model] = 0.0
        
        old_rate = template.model_success_rates[model]
        template.model_success_rates[model] = (
            (1 - alpha) * old_rate + alpha * (1.0 if success else 0.0)
        )
        
        # Update model parameters
        if model in self.model_params:
            model_param = self.model_params[model]
            category = template.category.value
            
            if category not in model_param.success_rates:
                model_param.success_rates[category] = 0.5
            
            old_rate = model_param.success_rates[category]
            model_param.success_rates[category] = (
                (1 - alpha) * old_rate + alpha * (1.0 if success else 0.0)
            )


class PromptBuilder:
    """
    Builds complete prompts with examples and context.
    Handles model-specific formatting.
    """
    
    def __init__(self,
                 template_selector: AdaptivePromptSelector,
                 example_retriever=None):  # RAG system
        self.selector = template_selector
        self.example_retriever = example_retriever
    
    def build_prompt(self,
                    query: str,
                    category: PromptCategory,
                    model: str,
                    context: Dict[str, Any],
                    include_examples: bool = True) -> Tuple[str, PromptTemplate]:
        """
        Build a complete prompt for the given query.
        
        Returns:
            Tuple of (prompt_text, template_used)
        """
        # Get model parameters
        model_param = self.selector.model_params.get(
            model, 
            ModelParameters(model_name=model)
        )
        
        # Select template
        template = self.selector.select_template(
            query, category, model, 
            context_size=len(str(context))  # Rough estimate
        )
        
        # Prepare variables
        variables = {}
        
        # Add context variables
        if "query" in template.variables:
            variables["query"] = query
        if "context" in template.variables:
            variables["context"] = self._format_context(context, model_param)
        if "functor" in template.variables:
            variables["functor"] = self._extract_functor(query)
        if "arity" in template.variables:
            variables["arity"] = self._extract_arity(query)
        
        # Add examples if needed
        if include_examples and "examples" in template.variables:
            examples = self._select_examples(
                query, model_param, category
            )
            variables["examples"] = self._format_examples(examples, model_param)
        
        # Fill in any missing variables
        for var in template.variables:
            if var not in variables:
                variables[var] = f"[{var}]"  # Placeholder
        
        # Build the prompt
        prompt_text = template.template.format(**variables)
        
        # Add model-specific formatting
        if model_param.needs_explicit_format:
            prompt_text = self._add_format_instructions(prompt_text, model_param)
        
        return prompt_text, template
    
    def _format_context(self, context: Dict[str, Any], model_param: ModelParameters) -> str:
        """Format context for the prompt"""
        if model_param.prefers_json:
            return json.dumps(context, indent=2)
        else:
            # Human-readable format
            lines = []
            if "facts" in context:
                lines.append(f"Facts: {context['facts']}")
            if "rules" in context:
                lines.append(f"Rules: {context['rules']}")
            return "\n".join(lines)
    
    def _select_examples(self, 
                        query: str,
                        model_param: ModelParameters,
                        category: PromptCategory) -> List[Dict]:
        """Select examples using the configured strategy"""
        if not self.example_retriever:
            return []
        
        num_examples = model_param.optimal_examples
        
        if model_param.example_selection_strategy == "similarity":
            # Get most similar examples
            examples = self.example_retriever.retrieve(
                query, k=num_examples
            )
        elif model_param.example_selection_strategy == "diversity":
            # Get diverse examples (TODO: implement clustering)
            examples = self.example_retriever.retrieve(
                query, k=num_examples * 2
            )
            # Sample for diversity
            examples = random.sample(examples, min(num_examples, len(examples)))
        else:  # mixed
            # Half similar, half random
            similar = self.example_retriever.retrieve(
                query, k=num_examples // 2
            )
            random_examples = self.example_retriever.retrieve(
                query, k=num_examples
            )
            examples = similar + random.sample(
                random_examples, 
                min(num_examples - len(similar), len(random_examples))
            )
        
        return examples
    
    def _format_examples(self, examples: List[Dict], model_param: ModelParameters) -> str:
        """Format examples for inclusion in prompt"""
        if not examples:
            return "No examples available"
        
        formatted = []
        for i, ex in enumerate(examples[:model_param.max_examples], 1):
            formatted.append(f"Example {i}: {json.dumps(ex)}")
        
        return "\n".join(formatted)
    
    def _extract_functor(self, query: str) -> str:
        """Extract the main functor from a query"""
        # Simple extraction - can be improved
        if "(" in query:
            return query.split("(")[1].split()[0]
        return "unknown"
    
    def _extract_arity(self, query: str) -> int:
        """Extract arity (number of arguments) from query"""
        # Simple counting - can be improved
        if "(" in query and ")" in query:
            args = query[query.index("(")+1:query.index(")")].split()
            return len(args) - 1  # Subtract functor
        return 0
    
    def _add_format_instructions(self, prompt: str, model_param: ModelParameters) -> str:
        """Add model-specific format instructions"""
        if model_param.prefers_json:
            prompt += "\n\nIMPORTANT: Return only valid JSON, no other text."
        return prompt


class PromptLearningSystem:
    """
    System that learns which prompts work best over time.
    Implements meta-learning for prompt optimization.
    """
    
    def __init__(self, 
                 library: PromptTemplateLibrary,
                 save_path: Optional[Path] = None):
        self.library = library
        self.save_path = save_path or Path.home() / ".dreamlog" / "prompt_learning.json"
        self.learning_history = []
        
        # Load existing learning if available
        if self.save_path.exists():
            self.load()
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze what makes prompts successful"""
        patterns = {
            "best_templates_by_category": {},
            "optimal_example_counts": {},
            "successful_patterns": [],
            "failure_patterns": []
        }
        
        # Find best templates per category
        for category in PromptCategory:
            templates = self.library.templates[category]
            if templates:
                best = max(templates, key=lambda t: t.success_rate)
                patterns["best_templates_by_category"][category.value] = {
                    "id": best.id,
                    "success_rate": best.success_rate,
                    "use_count": best.use_count
                }
        
        # Analyze successful patterns
        for entry in self.learning_history:
            if entry.get("success"):
                patterns["successful_patterns"].append({
                    "category": entry["category"],
                    "model": entry["model"],
                    "template_id": entry["template_id"],
                    "context_size": entry.get("context_size", 0)
                })
        
        return patterns
    
    def generate_new_template(self, 
                             category: PromptCategory,
                             based_on: List[PromptTemplate]) -> PromptTemplate:
        """
        Generate a new template based on successful ones.
        This is where meta-learning happens.
        """
        # Simple crossover for now - can be made more sophisticated
        if len(based_on) < 2:
            return None
        
        parent1, parent2 = random.sample(based_on, 2)
        
        # Combine elements from both templates
        new_template = PromptTemplate(
            id=f"generated_{category.value}_{len(self.library.templates[category])}",
            category=category,
            template=self._crossover_templates(parent1.template, parent2.template),
            variables=list(set(parent1.variables + parent2.variables))
        )
        
        return new_template
    
    def _crossover_templates(self, template1: str, template2: str) -> str:
        """Combine two templates to create a new one"""
        # Simple approach: take first half of one, second half of other
        lines1 = template1.split("\n")
        lines2 = template2.split("\n")
        
        mid1 = len(lines1) // 2
        mid2 = len(lines2) // 2
        
        new_lines = lines1[:mid1] + lines2[mid2:]
        return "\n".join(new_lines)
    
    def save(self):
        """Save learning history and template performance"""
        data = {
            "learning_history": self.learning_history,
            "template_performance": {}
        }
        
        for category in PromptCategory:
            data["template_performance"][category.value] = [
                {
                    "id": t.id,
                    "use_count": t.use_count,
                    "success_count": t.success_count,
                    "avg_quality": t.avg_response_quality,
                    "model_success_rates": t.model_success_rates
                }
                for t in self.library.templates[category]
            ]
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load learning history and update template performance"""
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            self.learning_history = data.get("learning_history", [])
            
            # Update template performance
            for category_name, templates_data in data.get("template_performance", {}).items():
                category = PromptCategory(category_name)
                for template_data in templates_data:
                    # Find matching template
                    for template in self.library.templates[category]:
                        if template.id == template_data["id"]:
                            template.use_count = template_data["use_count"]
                            template.success_count = template_data["success_count"]
                            template.avg_response_quality = template_data["avg_quality"]
                            template.model_success_rates = template_data["model_success_rates"]
                            break
        except Exception as e:
            print(f"Error loading prompt learning data: {e}")