"""
Fluent Pythonic API for DreamLog

Provides a natural Python interface for mixing DreamLog logic programming with Python code.
Supports method chaining, context managers, and seamless integration.

Example usage:
    from dreamlog.pythonic import DreamLog

    # Fluent API
    jl = DreamLog()
    jl.fact("parent", "john", "mary") \
      .fact("parent", "mary", "alice") \
      .rule("grandparent", ["X", "Z"]).when("parent", ["X", "Y"]).and_("parent", ["Y", "Z"])

    # Query with Python iteration
    for solution in jl.query("grandparent", "john", "Z"):
        print(f"John is grandparent of {solution['Z']}")

    # Mix with Python
    people = ["john", "mary", "alice"]
    for person in people:
        jl.fact("person", person)
        if person != "alice":
            jl.fact("adult", person)
"""

from typing import Any, Dict, List, Iterator, Optional
from contextlib import contextmanager
from dataclasses import dataclass

from .engine import DreamLogEngine
from .terms import Term
from .factories import atom, var, compound
from .knowledge import Fact, Rule
from .prefix_parser import parse_s_expression
from .llm_client import LLMClient
from .tfidf_embedding_provider import TfIdfEmbeddingProvider
from .prompt_template_system import RULE_EXAMPLES


def _to_term(arg: Any) -> Term:
    """Convert a Python value to a DreamLog term (variables from uppercase strings)."""
    if isinstance(arg, Term):
        return arg
    if isinstance(arg, str):
        return var(arg) if arg[0].isupper() else atom(arg)
    return atom(str(arg))


def _to_fact_term(arg: Any) -> Term:
    """Convert a Python value to a ground term (no variables)."""
    if isinstance(arg, Term):
        return arg
    return atom(str(arg))


def _build_compound(functor: str, term_args: List[Term]) -> Term:
    """Build an atom or compound term from a functor and argument list."""
    if not term_args:
        return atom(functor)
    return compound(functor, *term_args)


@dataclass
class QueryResult:
    """Result of a DreamLog query with Pythonic access"""
    bindings: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to bindings"""
        value = self.bindings.get(key)
        if value:
            # Convert term to Python value
            if hasattr(value, 'value'):
                return value.value
            return str(value)
        return None

    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to bindings"""
        return self[name]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain Python dict"""
        return {k: self[k] for k in self.bindings}


class RuleBuilder:
    """Fluent builder for rules"""

    def __init__(self, dreamlog: 'DreamLog', head_functor: str, head_args: List[Any]):
        self.dreamlog = dreamlog
        self.head_functor = head_functor
        self.head_args = head_args
        self.body_conditions = []

    def when(self, functor: str, args: List[Any]) -> 'RuleBuilder':
        """Add first condition to rule body"""
        self.body_conditions.append((functor, args))
        return self

    def and_(self, functor: str, args: List[Any]) -> 'RuleBuilder':
        """Add additional condition to rule body (and_ to avoid Python keyword)"""
        self.body_conditions.append((functor, args))
        return self

    def build(self) -> 'DreamLog':
        """Build and add the rule to the knowledge base"""
        head = _build_compound(self.head_functor, [_to_term(a) for a in self.head_args])
        body = [_build_compound(f, [_to_term(a) for a in args])
                for f, args in self.body_conditions]
        self.dreamlog.engine.add_rule(Rule(head, body))
        return self.dreamlog


class DreamLog:
    """
    Fluent Pythonic interface to DreamLog

    Provides method chaining, Python iteration, and seamless integration.
    """

    def __init__(self, llm_provider: Optional[str] = None, use_retry: bool = True, **llm_config):
        """
        Initialize DreamLog with optional LLM support

        Args:
            llm_provider: Provider name (openai, anthropic, ollama, etc.)
            use_retry: Whether to use retry wrapper for better JSON parsing
            **llm_config: Provider-specific configuration
        """
        if llm_provider:
            from .llm_hook import LLMHook

            # Extract retry-specific config
            max_retries = llm_config.pop('max_retries', 3)
            verbose_retry = llm_config.pop('verbose_retry', False)

            # Create base provider - support both string identifiers and direct injection
            if isinstance(llm_provider, str):
                provider = LLMClient(provider=llm_provider, **llm_config)
            else:
                # Already a provider object - use directly (for testing)
                provider = llm_provider

            # Wrap with retry logic if requested (skip for mock providers)
            is_mock = getattr(provider, 'provider', '') == 'mock'

            if use_retry and not is_mock:
                from .llm_retry_wrapper import create_retry_provider
                provider = create_retry_provider(
                    provider,
                    max_retries=max_retries,
                    verbose=verbose_retry
                )

            # Create embedding provider for RAG (use TF-IDF with RULE_EXAMPLES corpus)
            embedding_provider = TfIdfEmbeddingProvider(corpus=RULE_EXAMPLES)

            llm_hook = LLMHook(provider, embedding_provider)
            self.engine = DreamLogEngine(llm_hook)
        else:
            self.engine = DreamLogEngine()

    # Fluent fact/rule methods

    def fact(self, functor: str, *args: Any) -> 'DreamLog':
        """
        Add a fact to the knowledge base

        Examples:
            jl.fact("parent", "john", "mary")
            jl.fact("age", "john", 42)
        """
        fact_term = _build_compound(functor, [_to_fact_term(a) for a in args])
        self.engine.add_fact(Fact(fact_term))
        return self

    def rule(self, functor: str, args: List[Any]) -> RuleBuilder:
        """
        Start building a rule

        Example:
            jl.rule("grandparent", ["X", "Z"]) \
              .when("parent", ["X", "Y"]) \
              .and_("parent", ["Y", "Z"]) \
              .build()
        """
        return RuleBuilder(self, functor, args)

    def query(self, functor: str, *args: Any) -> Iterator[QueryResult]:
        """
        Query the knowledge base

        Examples:
            # Find all children of john
            for result in jl.query("parent", "john", "X"):
                print(result["X"])

            # Check if fact exists
            if jl.query("parent", "john", "mary"):
                print("John is Mary's parent")

        Returns:
            Iterator of QueryResult objects
        """
        query_term = _build_compound(functor, [_to_term(a) for a in args])
        for solution in self.engine.query([query_term]):
            yield QueryResult(solution.get_ground_bindings())

    def ask(self, functor: str, *args: Any) -> bool:
        """
        Check if a query has at least one solution

        Example:
            if jl.ask("parent", "john", "mary"):
                print("Yes, John is Mary's parent")
        """
        for _ in self.query(functor, *args):
            return True
        return False

    def find_all(self, functor: str, *args: Any) -> List[QueryResult]:
        """
        Find all solutions to a query

        Returns:
            List of all QueryResult objects
        """
        return list(self.query(functor, *args))

    def find_one(self, functor: str, *args: Any) -> Optional[QueryResult]:
        """
        Find first solution to a query

        Returns:
            First QueryResult or None
        """
        for result in self.query(functor, *args):
            return result
        return None

    # Batch operations

    def facts(self, *facts_data) -> 'DreamLog':
        """
        Add multiple facts at once

        Example:
            jl.facts(
                ("parent", "john", "mary"),
                ("parent", "mary", "alice"),
                ("age", "john", 42)
            )
        """
        for fact_tuple in facts_data:
            self.fact(*fact_tuple)
        return self

    # S-expression support

    def parse(self, sexp: str) -> 'DreamLog':
        """
        Parse and add S-expression fact or rule

        Examples:
            jl.parse("(parent john mary)")
            jl.parse("(grandparent X Z) :- (parent X Y), (parent Y Z)")
        """
        if ":-" in sexp:
            # It's a rule
            parts = sexp.split(":-")
            head = parse_s_expression(parts[0].strip())
            body_parts = parts[1].split(",")
            body = [parse_s_expression(b.strip()) for b in body_parts]
            self.engine.add_rule(Rule(head, body))
        else:
            # It's a fact
            term = parse_s_expression(sexp)
            self.engine.add_fact(Fact(term))
        return self

    # File operations

    def load(self, filename: str) -> 'DreamLog':
        """Load knowledge base from file"""
        with open(filename, 'r') as f:
            self.engine.load_from_prefix(f.read())
        return self

    def save(self, filename: str) -> 'DreamLog':
        """Save knowledge base to file"""
        with open(filename, 'w') as f:
            f.write(self.engine.save_to_prefix())
        return self

    # Context manager support

    @contextmanager
    def transaction(self):
        """
        Context manager for transactional operations

        Example:
            with jl.transaction():
                jl.fact("parent", "john", "mary")
                jl.fact("parent", "mary", "alice")
                # All facts added atomically
        """
        snapshot = self.engine.kb.copy()
        try:
            yield self
        except Exception:
            self.engine.kb.restore_from(snapshot)
            raise

    # Statistics and introspection

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        from .terms import Compound
        functors = set()
        for fact in self.engine.kb.facts:
            if isinstance(fact.term, Compound):
                functors.add(fact.term.functor)
        for rule in self.engine.kb.rules:
            if isinstance(rule.head, Compound):
                functors.add(rule.head.functor)
        return {
            "num_facts": len(self.engine.kb.facts),
            "num_rules": len(self.engine.kb.rules),
            "functors": sorted(functors),
            "total_items": len(self.engine.kb),
        }

    def clear(self) -> 'DreamLog':
        """Clear all facts and rules"""
        self.engine.clear_knowledge()
        return self

    def __repr__(self) -> str:
        """String representation"""
        return f"DreamLog({self.stats['num_facts']} facts, {self.stats['num_rules']} rules)"

    def __len__(self) -> int:
        """Number of facts and rules."""
        return len(self.engine.kb)


# Convenience functions for quick usage

def dreamlog(*args, **kwargs) -> DreamLog:
    """
    Create a new DreamLog instance

    Examples:
        jl = dreamlog()  # Basic instance
        jl = dreamlog(llm_provider="openai")  # With LLM support
    """
    return DreamLog(*args, **kwargs)
