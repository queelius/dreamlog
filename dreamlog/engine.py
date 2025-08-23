"""
Main engine for DreamLog

Provides a high-level interface that combines the knowledge base, evaluator, and LLM hook.
"""

from typing import List, Dict, Any, Optional
from .terms import Term, term_from_prefix, atom, var, compound
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
    
    def ask(self, goal: Term) -> bool:
        """
        Ask a yes/no question
        
        Args:
            goal: The goal to check
            
        Returns:
            True if at least one solution exists
        """
        return self.evaluator.ask_yes_no(goal)
    
    def find_all(self, goal: Term, var_name: str) -> List[Term]:
        """
        Find all values for a variable in a goal
        
        Args:
            goal: The goal term
            var_name: Name of the variable to collect values for
            
        Returns:
            List of terms that bind to the variable
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
