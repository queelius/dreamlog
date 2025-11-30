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

from typing import Any, Dict, List, Iterator, Optional, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import json

from .engine import DreamLogEngine
from .terms import Term
from .factories import atom, var, compound
from .knowledge import Fact, Rule
from .prefix_parser import parse_s_expression, parse_prefix_notation
from .config import DreamLogConfig
from .llm_providers import create_provider
from .tfidf_embedding_provider import TfIdfEmbeddingProvider
from .prompt_template_system import RULE_EXAMPLES


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
    
    def then(self, functor: str, args: List[Any]) -> 'RuleBuilder':
        """Alias for and_ - add condition"""
        return self.and_(functor, args)
    
    def build(self) -> 'DreamLog':
        """Build and add the rule to the knowledge base"""
        # Convert to terms
        head = self._make_term(self.head_functor, self.head_args)
        body = [self._make_term(f, a) for f, a in self.body_conditions]
        
        # Add rule
        self.dreamlog.engine.add_rule(Rule(head, body))
        return self.dreamlog
    
    def _make_term(self, functor: str, args: List[Any]) -> Term:
        """Convert Python values to DreamLog terms"""
        term_args = []
        for arg in args:
            if isinstance(arg, str):
                if arg[0].isupper():
                    term_args.append(var(arg))
                else:
                    term_args.append(atom(arg))
            elif isinstance(arg, Term):
                term_args.append(arg)
            else:
                term_args.append(atom(str(arg)))
        
        if len(term_args) == 0:
            return atom(functor)
        return compound(functor, *term_args)


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
            from .llm_providers import create_provider
            from .llm_hook import LLMHook
            
            # Extract retry-specific config
            max_retries = llm_config.pop('max_retries', 3)
            verbose_retry = llm_config.pop('verbose_retry', False)
            
            # Create base provider - support both string identifiers and direct injection
            if isinstance(llm_provider, str):
                # String identifier - use factory
                provider = create_provider(llm_provider, **llm_config)
            else:
                # Already a provider object - use directly (for testing)
                provider = llm_provider
            
            # Wrap with retry logic if requested (skip for mock providers)  
            provider_metadata = provider.get_metadata()
            is_mock = provider_metadata.get("provider_type") == "mock"
            
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
        term_args = []
        for arg in args:
            if isinstance(arg, str):
                term_args.append(atom(arg))
            elif isinstance(arg, (int, float)):
                term_args.append(atom(str(arg)))
            elif isinstance(arg, Term):
                term_args.append(arg)
            else:
                term_args.append(atom(str(arg)))
        
        if len(term_args) == 0:
            fact_term = atom(functor)
        else:
            fact_term = compound(functor, *term_args)
        
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
        # Build query term
        term_args = []
        for arg in args:
            if isinstance(arg, str):
                if arg[0].isupper():
                    term_args.append(var(arg))
                else:
                    term_args.append(atom(arg))
            elif isinstance(arg, (int, float)):
                term_args.append(atom(str(arg)))
            elif isinstance(arg, Term):
                term_args.append(arg)
            else:
                term_args.append(atom(str(arg)))
        
        if len(term_args) == 0:
            query_term = atom(functor)
        else:
            query_term = compound(functor, *term_args)
        
        # Execute query
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
        # Save current state (deep copy of internal lists)
        saved_facts = list(self.engine.kb._facts)
        saved_rules = list(self.engine.kb._rules)

        try:
            yield self
        except Exception:
            # Rollback on error - restore internal state and rebuild indices
            self.engine.kb._facts.clear()
            self.engine.kb._facts.extend(saved_facts)
            self.engine.kb._rules.clear()
            self.engine.kb._rules.extend(saved_rules)
            self.engine.kb._rebuild_indices()
            raise
    
    # Python integration
    
    def to_dataframe(self, functor: str, *args: Any):
        """
        Convert query results to pandas DataFrame
        
        Requires pandas to be installed.
        """
        try:
            import pandas as pd
            results = self.find_all(functor, *args)
            if results:
                return pd.DataFrame([r.to_dict() for r in results])
            return pd.DataFrame()
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
    
    def visualize(self, output_file: Optional[str] = None):
        """
        Visualize the knowledge base as a graph
        
        Requires networkx and matplotlib.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            
            # Add facts as edges
            for fact in self.engine.kb.facts:
                if hasattr(fact.term, 'functor') and hasattr(fact.term, 'args'):
                    if len(fact.term.args) == 2:
                        # Binary relation
                        G.add_edge(
                            str(fact.term.args[0]),
                            str(fact.term.args[1]),
                            label=fact.term.functor
                        )
            
            # Draw
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue',
                   node_size=500, font_size=10, arrows=True)
            
            # Add edge labels
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            
            if output_file:
                plt.savefig(output_file)
            else:
                plt.show()
            
        except ImportError:
            raise ImportError("networkx and matplotlib are required for visualize()")
    
    # Functional programming support
    
    def map_query(self, functor: str, *args: Any, 
                  mapper: Callable[[QueryResult], Any]) -> List[Any]:
        """
        Map a function over query results
        
        Example:
            names = jl.map_query("parent", "john", "X", 
                                mapper=lambda r: r["X"].upper())
        """
        return [mapper(result) for result in self.query(functor, *args)]
    
    def filter_query(self, functor: str, *args: Any,
                    predicate: Callable[[QueryResult], bool]) -> List[QueryResult]:
        """
        Filter query results
        
        Example:
            adults = jl.filter_query("person", "X",
                                   predicate=lambda r: jl.ask("adult", r["X"]))
        """
        return [r for r in self.query(functor, *args) if predicate(r)]
    
    # Statistics and introspection
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        functors = set()
        for fact in self.engine.kb.facts:
            if hasattr(fact.term, 'functor'):
                functors.add(fact.term.functor)
        for rule in self.engine.kb.rules:
            if hasattr(rule.head, 'functor'):
                functors.add(rule.head.functor)
        
        return {
            "num_facts": len(self.engine.kb.facts),
            "num_rules": len(self.engine.kb.rules),
            "functors": sorted(list(functors)),
            "total_items": len(self.engine.kb.facts) + len(self.engine.kb.rules)
        }
    
    def clear(self) -> 'DreamLog':
        """Clear all facts and rules"""
        self.engine.clear_knowledge()
        return self
    
    def __repr__(self) -> str:
        """String representation"""
        return f"DreamLog({self.stats['num_facts']} facts, {self.stats['num_rules']} rules)"
    
    def __len__(self) -> int:
        """Number of facts and rules"""
        return self.stats['total_items']


# Convenience functions for quick usage

def dreamlog(*args, **kwargs) -> DreamLog:
    """
    Create a new DreamLog instance
    
    Examples:
        jl = dreamlog()  # Basic instance
        jl = dreamlog(llm_provider="openai")  # With LLM support
    """
    return DreamLog(*args, **kwargs)


def demo():
    """Run a demonstration of the Pythonic API"""
    print("DreamLog Pythonic API Demo")
    print("=" * 50)
    
    # Create instance
    jl = dreamlog()
    
    # Add facts using method chaining
    jl.fact("parent", "john", "mary") \
      .fact("parent", "mary", "alice") \
      .fact("parent", "tom", "bob") \
      .fact("age", "john", 45) \
      .fact("age", "mary", 25)
    
    # Add a rule
    jl.rule("grandparent", ["X", "Z"]) \
      .when("parent", ["X", "Y"]) \
      .and_("parent", ["Y", "Z"]) \
      .build()
    
    print(f"\nKnowledge base: {jl}")
    print(f"Stats: {jl.stats}")
    
    # Query examples
    print("\nChildren of John:")
    for result in jl.query("parent", "john", "X"):
        print(f"  - {result['X']}")
    
    print("\nGrandparents:")
    for result in jl.query("grandparent", "X", "Z"):
        print(f"  {result['X']} is grandparent of {result['Z']}")
    
    # Boolean query
    if jl.ask("parent", "john", "mary"):
        print("\n✓ John is Mary's parent")
    
    # Using Python data
    print("\nAdding people from Python list:")
    people = ["alice", "bob", "charlie"]
    for person in people:
        jl.fact("person", person)
        print(f"  Added {person}")
    
    # Functional programming
    print("\nAll people (uppercase):")
    names = jl.map_query("person", "X", mapper=lambda r: r["X"].upper())
    print(f"  {', '.join(names)}")
    
    # S-expression parsing
    print("\nAdding via S-expressions:")
    jl.parse("(likes alice bob)") \
      .parse("(likes bob charlie)")
    
    for result in jl.query("likes", "X", "Y"):
        print(f"  {result['X']} likes {result['Y']}")


if __name__ == "__main__":
    demo()