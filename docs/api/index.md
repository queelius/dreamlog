# API Reference

The DreamLog API provides a comprehensive set of classes and functions for logic programming with wake-sleep optimization cycles.

## Core Modules

### [Terms](terms.md)
- `Term` - Base class for all terms
- `Atom` - Atomic constants
- `Variable` - Logic variables
- `Compound` - Compound terms with functors and arguments
- Factory functions: `atom()`, `var()`, `compound()`

### [Parser](parser.md)
- `parse_s_expression()` - Parse S-expression syntax
- `parse_prefix_notation()` - Parse JSON prefix arrays
- `term_to_sexp()` - Convert terms to S-expressions
- `term_to_prefix_json()` - Convert terms to JSON

### [Knowledge Base](knowledge_base.md)
- `KnowledgeBase` - Main storage for facts and rules
- `Fact` - Ground terms
- `Rule` - Head-body implications
- Indexing and retrieval methods

### [Unification](unification.md)
- `unify()` - Standard unification
- `match()` - One-way matching
- `subsumes()` - Subsumption checking
- `Unifier` - Stateful unification operations

### [Evaluator](evaluator.md)
- `PrologEvaluator` - SLD resolution with backtracking
- `Solution` - Query results with variable bindings
- Query evaluation strategies

### [Engine](engine.md)
- `JLogEngine` - High-level API combining all components
- `add_fact()`, `add_rule()`, `query()` methods
- Persistence with `save_to_prefix()`, `load_from_prefix()`

## LLM Integration

### [LLM Providers](llm_providers.md)
- `LLMProvider` - Protocol for all providers
- `MockLLMProvider` - Testing provider
- `OpenAIProvider` - OpenAI API integration
- `AnthropicProvider` - Anthropic Claude integration
- `OllamaProvider` - Local LLM support

### [LLM Hook](llm_hook.md)
- `LLMHook` - Automatic knowledge generation
- Context extraction and management
- Caching and rate limiting

### [Prompt Templates](prompt_templates.md)
- `PromptTemplateManager` - Manage prompt templates
- Template variables and substitution
- Custom template creation

## Wake-Sleep System

### [KB Dreamer](kb_dreamer.md)
- `KnowledgeBaseDreamer` - Wake-sleep optimization
- `DreamSession` - Dream cycle results
- `DreamInsight` - Individual optimizations
- Compression, abstraction, and generalization

### [Configuration](config.md)
- `DreamLogConfig` - Main configuration
- `LLMSamplingConfig` - Sampling strategies
- YAML configuration support

## Pythonic Interface

### [Pythonic API](pythonic.md)
- `dreamlog()` - Fluent API factory
- Method chaining for facts and rules
- Query execution and results

## Integration Modules

### [REPL](repl.md)
- Interactive command-line interface
- Commands for queries, facts, rules
- Syntax highlighting and completion

### [MCP Server](mcp.md)
- Model Context Protocol integration
- Tool definitions for DreamLog operations
- WebSocket communication

### [REST API](rest_api.md)
- HTTP endpoints for DreamLog
- WebSocket REPL support
- JSON request/response format

## Utilities

### [Types](types.md)
- Type hints and protocols
- Common type definitions

### [Exceptions](exceptions.md)
- Custom exception classes
- Error handling patterns