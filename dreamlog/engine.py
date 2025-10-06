"""
Main engine for DreamLog

Provides a high-level interface that combines the knowledge base, evaluator, and LLM hook.
"""

from typing import List, Dict, Any, Optional
from .terms import Term, Atom
from .factories import atom, var, compound, term_from_prefix
from .knowledge import KnowledgeBase, Fact, Rule
from .evaluator import PrologEvaluator, Solution
from .llm_hook import LLMHook
from .llm_providers import LLMProvider
import json


class DreamLogEngine:
    """
    Main interface for DreamLog - combines knowledge base, evaluator, and LLM integration
    """
    
    def __init__(self, llm_hook: Optional[LLMHook] = None):
        """
        Initialize the DreamLog engine
        
        Args:
            llm_hook: Optional LLM hook for automatic knowledge generation
        """
        self.kb = KnowledgeBase()
        self.llm_hook = llm_hook
        
        # Create evaluator with LLM hook if provided
        unknown_hook = llm_hook if llm_hook else None
        self.evaluator = PrologEvaluator(self.kb, unknown_hook)
        
        self._trace = False
    
    def add_fact(self, fact: Fact) -> None:
        """Add a fact to the knowledge base"""
        self.kb.add_fact(fact)
        if self._trace:
            print(f"Added fact: {fact}")
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the knowledge base"""
        self.kb.add_rule(rule)
        if self._trace:
            print(f"Added rule: {rule}")
    
    def add_fact_from_term(self, term: Term) -> None:
        """Add a fact from a term"""
        fact = Fact(term)
        self.add_fact(fact)
    
    def add_rule_from_terms(self, head: Term, body: List[Term]) -> None:
        """Add a rule from head and body terms"""
        rule = Rule(head, body)
        self.add_rule(rule)
    
    def query(self, goals: List[Term]) -> List[Solution]:
        """
        Execute a query and return all solutions
        
        Args:
            goals: List of goal terms to solve
            
        Returns:
            List of solutions with variable bindings
        """
        if self._trace:
            goals_str = ", ".join(str(goal) for goal in goals)
            print(f"Query: ?- {goals_str}.")
        
        solutions = list(self.evaluator.query(goals))
        
        if self._trace:
            print(f"Found {len(solutions)} solution(s)")
            for i, solution in enumerate(solutions, 1):
                bindings_str = self._format_bindings(solution.get_ground_bindings())
                print(f"  {i}. {bindings_str}")
        
        return solutions
    
    def ask(self, *args) -> bool:
        """
        Ask a yes/no question
        
        Args:
            Either a single Term object, or functor and arguments as strings
            
        Returns:
            True if at least one solution exists
        """
        if len(args) == 1 and isinstance(args[0], Term):
            goal = args[0]
        else:
            # Build compound term from strings
            functor = args[0]
            term_args = []
            for arg in args[1:]:
                if isinstance(arg, str) and arg[0].isupper():
                    term_args.append(var(arg))
                else:
                    term_args.append(atom(arg))
            goal = compound(functor, *term_args)
        
        return self.evaluator.ask_yes_no(goal)
    
    def find_all_terms(self, goal: Term, var_name: str) -> List[Term]:
        """
        Find all term values for a variable in a goal.
        
        Args:
            goal: The goal term to query
            var_name: Name of the variable to collect values for
            
        Returns:
            List of Terms that bind to the variable
            
        Examples:
            >>> engine.find_all_terms(compound("parent", var("X"), atom("mary")), "X")
            [Atom("john"), Atom("jane")]
        """
        solutions = self.query([goal])
        values = []
        
        for solution in solutions:
            binding = solution.get_binding(var_name)
            if binding is not None:
                # Apply all bindings to get the final value
                final_value = binding.substitute(solution.bindings)
                if final_value not in values:
                    values.append(final_value)
        
        return values
    
    def find_all(self, functor: str, *args: str) -> List[Dict[str, Any]]:
        """
        Convenience method to find all solutions with string-based queries.
        
        Variables are identified by starting with uppercase letters.
        Returns solutions as dictionaries with string values.
        
        Args:
            functor: The predicate functor
            *args: Arguments where uppercase strings become variables
            
        Returns:
            List of dictionaries mapping variable names to their string values
            
        Examples:
            >>> engine.find_all("parent", "john", "X")
            [{"X": "mary"}, {"X": "bob"}]
            >>> engine.find_all("parent", "X", "Y")
            [{"X": "john", "Y": "mary"}, {"X": "jane", "Y": "bob"}]
        """
        term_args = []
        var_names = []
        
        for arg in args:
            if isinstance(arg, str) and arg and arg[0].isupper():
                term_args.append(var(arg))
                var_names.append(arg)
            else:
                term_args.append(atom(arg))
        
        goal = compound(functor, *term_args)
        solutions = self.query([goal])
        
        # Return list of binding dicts
        results = []
        for solution in solutions:
            result = {}
            for var_name in var_names:
                binding = solution.get_binding(var_name)
                if binding is not None:
                    # Convert to string for convenience
                    if isinstance(binding, Atom):
                        result[var_name] = binding.value
                    else:
                        result[var_name] = str(binding)
            if result:
                results.append(result)
        
        return results
    
    def _format_bindings(self, bindings: Dict[str, Term]) -> str:
        """Format variable bindings for display"""
        if not bindings:
            return "Yes"
        
        binding_strs = []
        for var_name, term in bindings.items():
            binding_strs.append(f"{var_name} = {term}")
        
        return ", ".join(binding_strs)
    
    def set_trace(self, trace: bool) -> None:
        """Enable or disable tracing"""
        self._trace = trace
    
    def clear_knowledge(self) -> None:
        """Clear all knowledge from the knowledge base"""
        self.kb.clear()
        if self.llm_hook:
            self.llm_hook.clear_cache()
    
    def load_from_prefix(self, json_str: str) -> None:
        """Load knowledge base from prefix notation JSON string"""
        self.kb.from_prefix(json_str)
    
    def save_to_prefix(self) -> str:
        """Save knowledge base to prefix notation JSON string"""
        return self.kb.to_prefix()
    
    def add_from_prefix(self, prefix_json: str) -> None:
        """Add facts and rules from prefix notation JSON string"""
        data = json.loads(prefix_json)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list) and len(item) > 0:
                    if item[0] == "fact":
                        self.add_fact(Fact.from_prefix(item))
                    elif item[0] == "rule":
                        self.add_rule(Rule.from_prefix(item))
    
    @property
    def facts(self) -> List[Fact]:
        """Get all facts in the knowledge base"""
        return self.kb.facts
    
    @property
    def rules(self) -> List[Rule]:
        """Get all rules in the knowledge base"""
        return self.kb.rules
    
    def __str__(self) -> str:
        """String representation of the engine"""
        return f"DreamLogEngine: {len(self.kb.facts)} facts, {len(self.kb.rules)} rules"


# Convenience functions for creating common patterns
def create_family_kb() -> DreamLogEngine:
    """Create a DreamLog engine with basic family relationships"""
    engine = DreamLogEngine()
    
    # Add some basic facts
    engine.add_fact_from_term(compound("parent", atom("john"), atom("mary")))
    engine.add_fact_from_term(compound("parent", atom("john"), atom("tom")))
    engine.add_fact_from_term(compound("parent", atom("mary"), atom("alice")))
    
    # Add rules
    engine.add_rule_from_terms(
        compound("grandparent", var("X"), var("Z")),
        [
            compound("parent", var("X"), var("Y")),
            compound("parent", var("Y"), var("Z"))
        ]
    )
    
    return engine


def create_engine_with_llm(llm_provider: LLMProvider) -> DreamLogEngine:
    """Create a DreamLog engine with LLM integration"""
    llm_hook = LLMHook(llm_provider)
    return DreamLogEngine(llm_hook)
