# LLM Integration Guide

DreamLog seamlessly integrates Large Language Models to automatically generate knowledge when undefined predicates are encountered. This creates an adaptive, self-extending knowledge system.

## How LLM Integration Works

### The Hook System

When DreamLog encounters an undefined predicate during query evaluation:

1. **Detection**: Query evaluator detects undefined predicate
2. **Hook Trigger**: LLM hook is called with context
3. **Generation**: LLM generates relevant facts and rules
4. **Integration**: Generated knowledge is added to KB
5. **Re-evaluation**: Query continues with new knowledge

```python
from dreamlog.pythonic import dreamlog

# Create KB with LLM support
kb = dreamlog(llm_provider="openai")

# Query undefined predicate
# LLM automatically generates definition of "healthy"
for result in kb.query("healthy", "alice"):
    print(result)

# Generated knowledge is now part of KB
kb.get_rules("healthy")  # Shows LLM-generated rules
```

## Configuration

### Provider Setup

```python
# Using environment variables (recommended)
import os
os.environ['OPENAI_API_KEY'] = 'your-key'
kb = dreamlog(llm_provider="openai")

# Using Anthropic
os.environ['ANTHROPIC_API_KEY'] = 'your-key'
kb = dreamlog(llm_provider="anthropic")

# Using local Ollama
kb = dreamlog(llm_provider="ollama", llm_model="llama2")

# Using mock provider for testing
kb = dreamlog(llm_provider="mock")
```

### Configuration File

Create `llm_config.json`:

```json
{
  "provider": "openai",
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 500,
  "api_key_env": "OPENAI_API_KEY",
  "prompt_style": "structured",
  "cache_responses": true,
  "debug": false
}
```

Load configuration:

```python
from dreamlog.llm_config import LLMConfig

config = LLMConfig.from_file("llm_config.json")
kb = dreamlog(llm_config=config)
```

### Custom Configuration

```python
from dreamlog.llm_config import LLMConfig

config = LLMConfig(
    provider="openai",
    model="gpt-4",
    temperature=0.3,  # Lower = more deterministic
    max_tokens=1000,
    system_prompt="You are a logic programming expert.",
    response_format="structured"
)

kb = dreamlog(llm_config=config)
```

## Providers

### OpenAI Provider

```python
from dreamlog.llm_providers import OpenAIProvider

provider = OpenAIProvider(
    api_key=os.environ['OPENAI_API_KEY'],
    model="gpt-4",
    temperature=0.7
)

kb = dreamlog(llm_provider=provider)
```

### Anthropic Provider

```python
from dreamlog.llm_providers import AnthropicProvider

provider = AnthropicProvider(
    api_key=os.environ['ANTHROPIC_API_KEY'],
    model="claude-3-opus-20240229",
    max_tokens=1000
)

kb = dreamlog(llm_provider=provider)
```

### Ollama Provider (Local)

```python
from dreamlog.llm_providers import OllamaProvider

provider = OllamaProvider(
    model="llama2",
    base_url="http://localhost:11434"
)

kb = dreamlog(llm_provider=provider)
```

### Custom HTTP Provider

```python
from dreamlog.llm_http_provider import HTTPLLMAdapter

provider = HTTPLLMAdapter(
    endpoint="https://your-api.com/v1/completions",
    headers={"Authorization": "Bearer your-token"},
    model="custom-model",
    request_template={
        "model": "{model}",
        "prompt": "{prompt}",
        "max_tokens": "{max_tokens}"
    },
    response_path="choices.0.text"
)

kb = dreamlog(llm_provider=provider)
```

### Mock Provider (Testing)

```python
from dreamlog.llm_providers import MockLLMProvider

# Predefined responses for testing
provider = MockLLMProvider({
    "healthy": [
        ["healthy", "X"],  # Simple fact
        ["rule", ["healthy", "X"], 
         [["exercises", "X"], ["eats_well", "X"]]]
    ]
})

kb = dreamlog(llm_provider=provider)
```

## Prompt Templates

### Default Templates

DreamLog's default prompt template automatically includes existing facts and rules to ensure generated knowledge is consistent:

```python
# The default template receives:
# - ${knowledge_base}: Sample of existing facts and rules
# - ${term}: The undefined term being queried
# - ${functor}: The main functor
# - ${domain}: Knowledge domain

# Example of what LLM sees:
"""
EXISTING KNOWLEDGE BASE:
RELATED FACTS:
(parent john mary)
(parent mary alice)
(age john 45)
RELATED RULES:
(grandparent X Z) :- (parent X Y), (parent Y Z)
OTHER PREDICATES IN KB: age, parent, grandparent

The knowledge base above shows existing facts and rules. Your generated knowledge should:
1. Be consistent with existing facts and rules
2. Follow the same naming conventions and patterns
3. Extend the knowledge base coherently
4. Not contradict existing knowledge

TASK: Generate facts and/or rules to define "sibling" that are consistent with the existing knowledge.
"""
```

The system automatically:
- Extracts related facts (up to 20, with sampling if more)
- Includes all related rules (up to 15)
- Shows other predicates for context
- Ensures consistency with existing knowledge

### Domain-Specific Templates

```python
# Medical domain
medical_template = PromptTemplate(
    system="You are a medical knowledge expert.",
    user="""In a medical context, define the predicate: {predicate}
    Consider symptoms, conditions, and treatments.
    
    Current facts: {context}
    
    Generate medically accurate rules in S-expression format."""
)

# Academic domain
academic_template = PromptTemplate(
    system="You are an academic advisor.",
    user="""For academic advising, define: {predicate}
    Consider prerequisites, requirements, and policies.
    
    Context: {context}
    
    Generate rules following university policies."""
)

# Set domain-specific template
kb.set_prompt_template(medical_template)
```

### Custom Response Parsing

```python
def custom_parser(response_text):
    """Parse LLM response into facts and rules"""
    facts = []
    rules = []
    
    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('('):
            if ':-' in line:
                # Parse rule
                rules.append(parse_rule(line))
            else:
                # Parse fact
                facts.append(parse_fact(line))
    
    return facts, rules

template = PromptTemplate(
    system="Generate logic rules",
    user="{predicate} with context: {context}",
    parse_response=custom_parser
)
```

## Context Management

### Providing Context

```python
# Set domain context for better generation
kb.set_llm_context({
    "domain": "healthcare",
    "entities": ["patients", "doctors", "treatments"],
    "constraints": ["HIPAA compliance", "medical ethics"]
})

# Context is included in LLM prompts
for result in kb.query("can_prescribe", "dr_smith", "medication_x"):
    print(result)
```

### Dynamic Context

```python
class ContextualKB:
    def __init__(self):
        self.kb = dreamlog(llm_provider="openai")
    
    def query_with_context(self, *query_args):
        # Gather relevant context
        context = self._gather_context(query_args[0])
        
        # Set context for this query
        self.kb.set_llm_context(context)
        
        # Execute query
        return self.kb.query(*query_args)
    
    def _gather_context(self, predicate):
        # Extract related facts
        related = []
        for fact in self.kb.get_facts():
            if self._is_related(fact, predicate):
                related.append(fact)
        
        return {
            "related_facts": related,
            "timestamp": datetime.now(),
            "query_predicate": predicate
        }
```

### Context Window Management

```python
from dreamlog.llm_hook import LLMHook

class SmartLLMHook(LLMHook):
    def __init__(self, provider, max_context_facts=50):
        super().__init__(provider)
        self.max_context_facts = max_context_facts
    
    def generate_knowledge(self, predicate, args, kb):
        # Limit context to most relevant facts
        context = self._get_relevant_context(predicate, kb)
        
        # Generate with limited context
        return self.provider.generate(
            predicate=predicate,
            context=context[:self.max_context_facts]
        )
    
    def _get_relevant_context(self, predicate, kb):
        # Prioritize related predicates
        scores = {}
        for fact in kb.get_facts():
            score = self._relevance_score(fact, predicate)
            scores[fact] = score
        
        # Return top-scored facts
        sorted_facts = sorted(scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        return [f for f, _ in sorted_facts]
```

## Caching and Performance

### Response Caching

```python
from functools import lru_cache
import hashlib

class CachedLLMProvider:
    def __init__(self, base_provider):
        self.provider = base_provider
        self.cache = {}
    
    def generate(self, predicate, context):
        # Create cache key
        key = self._cache_key(predicate, context)
        
        # Check cache
        if key in self.cache:
            return self.cache[key]
        
        # Generate and cache
        response = self.provider.generate(predicate, context)
        self.cache[key] = response
        return response
    
    def _cache_key(self, predicate, context):
        content = f"{predicate}:{str(context)}"
        return hashlib.md5(content.encode()).hexdigest()

# Use cached provider
base = OpenAIProvider(api_key="...")
cached = CachedLLMProvider(base)
kb = dreamlog(llm_provider=cached)
```

### Batch Generation

```python
class BatchLLMProvider:
    def __init__(self, provider):
        self.provider = provider
        self.pending = []
    
    def queue_generation(self, predicate, context):
        """Queue for batch processing"""
        self.pending.append((predicate, context))
    
    def process_batch(self):
        """Process all queued requests"""
        if not self.pending:
            return []
        
        # Combine into single prompt
        combined_prompt = self._combine_prompts(self.pending)
        
        # Single LLM call
        response = self.provider.generate_batch(combined_prompt)
        
        # Parse and distribute responses
        results = self._parse_batch_response(response)
        self.pending.clear()
        
        return results
```

## Advanced Patterns

### Confidence Scoring

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ScoredKnowledge:
    facts: List[Tuple]
    rules: List[Tuple]
    confidence: float
    reasoning: str

class ConfidenceLLMProvider:
    def __init__(self, provider):
        self.provider = provider
    
    def generate_with_confidence(self, predicate, context):
        # Request confidence in response
        prompt = f"""
        Generate knowledge for: {predicate}
        Context: {context}
        
        Also provide:
        CONFIDENCE: 0.0-1.0
        REASONING: Why this knowledge is appropriate
        """
        
        response = self.provider.generate(prompt)
        return self._parse_scored_response(response)
    
    def _parse_scored_response(self, response):
        # Parse facts, rules, confidence, reasoning
        return ScoredKnowledge(
            facts=parse_facts(response),
            rules=parse_rules(response),
            confidence=parse_confidence(response),
            reasoning=parse_reasoning(response)
        )

# Use only high-confidence knowledge
kb = dreamlog()
provider = ConfidenceLLMProvider(base_provider)

knowledge = provider.generate_with_confidence("diagnosis", context)
if knowledge.confidence > 0.8:
    for fact in knowledge.facts:
        kb.fact(*fact)
```

### Multi-Model Ensemble

```python
class EnsembleLLMProvider:
    def __init__(self, providers):
        self.providers = providers
    
    def generate(self, predicate, context):
        # Get responses from all models
        responses = []
        for provider in self.providers:
            resp = provider.generate(predicate, context)
            responses.append(resp)
        
        # Combine/vote on responses
        return self._combine_responses(responses)
    
    def _combine_responses(self, responses):
        # Majority voting on facts
        fact_votes = {}
        for resp in responses:
            for fact in resp.facts:
                key = str(fact)
                fact_votes[key] = fact_votes.get(key, 0) + 1
        
        # Keep facts with majority agreement
        threshold = len(self.providers) / 2
        agreed_facts = [
            eval(fact) for fact, votes in fact_votes.items()
            if votes > threshold
        ]
        
        return {"facts": agreed_facts, "rules": []}

# Create ensemble
ensemble = EnsembleLLMProvider([
    OpenAIProvider(model="gpt-4"),
    AnthropicProvider(model="claude-3"),
    OllamaProvider(model="llama2")
])

kb = dreamlog(llm_provider=ensemble)
```

### Interactive Refinement

```python
class InteractiveLLMProvider:
    def __init__(self, provider):
        self.provider = provider
        self.refinements = []
    
    def generate_interactive(self, predicate, context):
        # Initial generation
        response = self.provider.generate(predicate, context)
        
        # Show to user
        print(f"Generated knowledge for {predicate}:")
        print(response)
        
        # Get feedback
        feedback = input("Feedback (or 'ok' to accept): ")
        
        if feedback != 'ok':
            # Refine based on feedback
            refined = self.provider.generate(
                f"Refine {predicate} based on: {feedback}",
                context + [("feedback", feedback)]
            )
            response = refined
        
        self.refinements.append((predicate, feedback))
        return response
```

### Learning from Usage

```python
class LearningLLMProvider:
    def __init__(self, provider):
        self.provider = provider
        self.usage_history = []
        self.successful_patterns = []
    
    def generate(self, predicate, context):
        # Check if we've seen similar before
        similar = self._find_similar_cases(predicate, context)
        
        if similar:
            # Use successful patterns
            enhanced_context = context + similar
        else:
            enhanced_context = context
        
        response = self.provider.generate(predicate, enhanced_context)
        
        # Track usage
        self.usage_history.append({
            'predicate': predicate,
            'context': context,
            'response': response,
            'timestamp': datetime.now()
        })
        
        return response
    
    def mark_successful(self, predicate):
        """Mark a generation as successful"""
        for entry in self.usage_history:
            if entry['predicate'] == predicate:
                self.successful_patterns.append(entry)
    
    def _find_similar_cases(self, predicate, context):
        """Find similar successful cases"""
        similar = []
        for pattern in self.successful_patterns:
            if self._similarity(pattern['predicate'], predicate) > 0.7:
                similar.append(pattern)
        return similar[:3]  # Top 3 similar cases
```

## Error Handling

### Graceful Degradation

```python
class RobustLLMHook:
    def __init__(self, provider, fallback_provider=None):
        self.provider = provider
        self.fallback = fallback_provider
    
    def generate_knowledge(self, predicate, args, kb):
        try:
            # Try primary provider
            return self.provider.generate(predicate, kb.get_context())
        except Exception as e:
            print(f"Primary LLM failed: {e}")
            
            if self.fallback:
                try:
                    # Try fallback
                    return self.fallback.generate(predicate, kb.get_context())
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
            
            # Return empty knowledge
            return {"facts": [], "rules": []}

# Setup with fallback
primary = OpenAIProvider()
fallback = OllamaProvider()  # Local fallback
hook = RobustLLMHook(primary, fallback)
```

### Validation

```python
class ValidatingLLMProvider:
    def __init__(self, provider):
        self.provider = provider
    
    def generate(self, predicate, context):
        response = self.provider.generate(predicate, context)
        
        # Validate generated knowledge
        valid_facts = []
        valid_rules = []
        
        for fact in response.facts:
            if self._validate_fact(fact):
                valid_facts.append(fact)
            else:
                print(f"Invalid fact rejected: {fact}")
        
        for rule in response.rules:
            if self._validate_rule(rule):
                valid_rules.append(rule)
            else:
                print(f"Invalid rule rejected: {rule}")
        
        return {"facts": valid_facts, "rules": valid_rules}
    
    def _validate_fact(self, fact):
        # Check fact structure
        if not isinstance(fact, (list, tuple)):
            return False
        if len(fact) < 1:
            return False
        # Additional validation...
        return True
    
    def _validate_rule(self, rule):
        # Check rule structure
        if not isinstance(rule, (list, tuple)):
            return False
        if len(rule) != 3 or rule[0] != "rule":
            return False
        # Additional validation...
        return True
```

## Best Practices

### 1. Choose the Right Provider

- **OpenAI GPT-4**: Best for complex reasoning
- **Anthropic Claude**: Good for structured generation
- **Ollama**: Best for privacy/local deployment
- **Mock**: Essential for testing

### 2. Optimize Prompts

```python
# Be specific about format
template = PromptTemplate(
    system="Generate Prolog facts and rules in S-expression format.",
    user="""Generate knowledge for: {predicate}

Requirements:
- Use S-expression syntax: (functor arg1 arg2)
- Keep facts simple and atomic
- Rules should use :- for implication
- Variables start with uppercase

Context: {context}

Response format:
FACTS:
(fact1 ...)
RULES:
(rule1 ...) :- (condition1 ...), (condition2 ...)
"""
)
```

### 3. Manage Costs

```python
class CostAwareLLMProvider:
    def __init__(self, provider, max_cost_per_query=0.10):
        self.provider = provider
        self.max_cost = max_cost_per_query
        self.total_cost = 0.0
    
    def generate(self, predicate, context):
        # Estimate cost
        estimated_cost = self._estimate_cost(context)
        
        if estimated_cost > self.max_cost:
            # Reduce context size
            context = self._reduce_context(context)
        
        response = self.provider.generate(predicate, context)
        
        # Track actual cost
        actual_cost = self._calculate_cost(response)
        self.total_cost += actual_cost
        
        return response
```

### 4. Monitor Quality

```python
class QualityMonitor:
    def __init__(self, kb):
        self.kb = kb
        self.metrics = {
            'generated_facts': 0,
            'generated_rules': 0,
            'successful_queries': 0,
            'failed_queries': 0
        }
    
    def track_generation(self, predicate, response):
        self.metrics['generated_facts'] += len(response.facts)
        self.metrics['generated_rules'] += len(response.rules)
    
    def track_query(self, query, success):
        if success:
            self.metrics['successful_queries'] += 1
        else:
            self.metrics['failed_queries'] += 1
    
    def report(self):
        total_queries = (self.metrics['successful_queries'] + 
                        self.metrics['failed_queries'])
        success_rate = (self.metrics['successful_queries'] / 
                       total_queries if total_queries > 0 else 0)
        
        print(f"Quality Report:")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Generated Facts: {self.metrics['generated_facts']}")
        print(f"  Generated Rules: {self.metrics['generated_rules']}")
```

## Next Steps

- [Examples](../examples/llm-examples.md) - LLM integration examples
- [API Reference](../api/llm.md) - Complete LLM API
- [Providers](../api/providers.md) - Provider implementations
- [Templates](../api/templates.md) - Prompt template guide