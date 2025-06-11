"""
Prolog-style evaluator for JLOG

Implements SLD resolution for query evaluation with proper backtracking.
"""

from typing import List, Dict, Any, Iterator, Optional, Callable, Set
from dataclasses import dataclass
from .terms import Term, Variable, atom, var, compound
from .knowledge import KnowledgeBase, Fact, Rule
from .unification import Unifier
import copy


@dataclass
class Goal:
    """Represents a goal to be solved"""
    term: Term
    bindings: Dict[str, Term]
    
    def substitute(self, new_bindings: Dict[str, Term]) -> 'Goal':
        """Apply substitutions to create a new goal"""
        combined_bindings = Unifier.compose_bindings(self.bindings, new_bindings)
        new_term = self.term.substitute(new_bindings)
        return Goal(new_term, combined_bindings)


@dataclass 
class Solution:
    """Represents a solution to a query"""
    bindings: Dict[str, Term]
    
    def get_binding(self, var_name: str) -> Optional[Term]:
        """Get the binding for a variable"""
        return self.bindings.get(var_name)
    
    def get_ground_bindings(self) -> Dict[str, Term]:
        """Get only the ground (variable-free) bindings"""
        ground_bindings = {}
        for var_name, term in self.bindings.items():
            final_term = term.substitute(self.bindings)
            if Unifier.ground_term(final_term, {}):
                ground_bindings[var_name] = final_term
        return ground_bindings


class PrologEvaluator:
    """
    Evaluates Prolog queries using SLD resolution
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, 
                 unknown_hook: Optional[Callable[[Term, 'PrologEvaluator'], None]] = None):
        """
        Initialize the evaluator
        
        Args:
            knowledge_base: The knowledge base to query against
            unknown_hook: Called when an unknown term is encountered
        """
        self.kb = knowledge_base
        self.unknown_hook = unknown_hook
        self._call_count = 0
        self._max_depth = 100  # Prevent infinite recursion
    
    def query(self, goals: List[Term]) -> Iterator[Solution]:
        """
        Evaluate a query (list of goals) and yield all solutions
        
        Args:
            goals: List of goal terms to solve
            
        Yields:
            Solution objects containing variable bindings
        """
        if not goals:
            return
        
        self._call_count = 0
        initial_goals = [Goal(goal, {}) for goal in goals]
        
        # Generate solutions using SLD resolution
        yield from self._solve_goals(initial_goals, {})
    
    def _solve_goals(self, goals: List[Goal], global_bindings: Dict[str, Term]) -> Iterator[Solution]:
        """
        Solve a list of goals using SLD resolution
        
        Args:
            goals: Current goals to solve
            global_bindings: Global variable bindings
            
        Yields:
            Solutions when all goals are satisfied
        """
        self._call_count += 1
        if self._call_count > self._max_depth:
            return  # Prevent stack overflow
        
        if not goals:
            # All goals solved - yield solution
            yield Solution(global_bindings)
            return
        
        # Take the first goal
        current_goal = goals[0]
        remaining_goals = goals[1:]
        
        # Apply global bindings to the current goal
        current_goal = current_goal.substitute(global_bindings)
        
        # Try to solve the current goal
        solutions_found = False
        
        # Try facts first
        for fact in self.kb.get_matching_facts(current_goal.term):
            renamed_fact = self._rename_variables_in_fact(fact)
            
            bindings = Unifier.unify(current_goal.term, renamed_fact.term, current_goal.bindings)
            if bindings is not None:
                solutions_found = True
                new_global_bindings = Unifier.compose_bindings(global_bindings, bindings)
                
                # Apply bindings to remaining goals
                new_remaining_goals = []
                for goal in remaining_goals:
                    new_goal = goal.substitute(bindings)
                    new_remaining_goals.append(new_goal)
                
                # Recursively solve remaining goals
                yield from self._solve_goals(new_remaining_goals, new_global_bindings)
        
        # Try rules (with cycle detection)
        for rule in self.kb.get_matching_rules(current_goal.term):
            renamed_rule = self._rename_variables_in_rule(rule)
            
            bindings = Unifier.unify(current_goal.term, renamed_rule.head, current_goal.bindings)
            if bindings is not None:
                solutions_found = True
                
                # Create new goals from the rule body
                new_goals = []
                for body_term in renamed_rule.body:
                    new_goal = Goal(body_term, bindings)
                    new_goals.append(new_goal)
                
                # Add remaining goals
                for goal in remaining_goals:
                    new_goal = goal.substitute(bindings)
                    new_goals.append(new_goal)
                
                new_global_bindings = Unifier.compose_bindings(global_bindings, bindings)
                
                # Check for simple infinite recursion (same goal appearing again)
                if self._detect_simple_cycle(new_goals, current_goal):
                    continue
                
                # Recursively solve new goals
                yield from self._solve_goals(new_goals, new_global_bindings)
        
        # If no solutions found and we have an unknown hook, try it ONCE
        if not solutions_found and self.unknown_hook and self._call_count < 50:
            original_kb_size = len(self.kb)
            self.unknown_hook(current_goal.term, self)
            
            # Only try again if new knowledge was actually added
            if len(self.kb) > original_kb_size:
                yield from self._solve_goals(goals, global_bindings)
    
    def _rename_variables_in_fact(self, fact: Fact) -> Fact:
        """Rename variables in a fact to avoid conflicts"""
        suffix = str(self._call_count)
        variables = fact.get_variables()
        
        if not variables:
            return fact
        
        bindings = {}
        for var_name in variables:
            new_name = f"{var_name}_{suffix}"
            bindings[var_name] = var(new_name)
        
        return fact.substitute(bindings)
    
    def _rename_variables_in_rule(self, rule: Rule) -> Rule:
        """Rename variables in a rule to avoid conflicts"""
        suffix = str(self._call_count)
        return rule.rename_variables(suffix)
    
    def ask_yes_no(self, goal: Term) -> bool:
        """
        Ask a yes/no question
        
        Args:
            goal: The goal to check
            
        Returns:
            True if at least one solution exists, False otherwise
        """
        solutions = list(self.query([goal]))
        return len(solutions) > 0
    
    def find_all_solutions(self, goals: List[Term]) -> List[Solution]:
        """Find all solutions to a query"""
        return list(self.query(goals))
    
    def find_first_solution(self, goals: List[Term]) -> Optional[Solution]:
        """Find the first solution to a query"""
        for solution in self.query(goals):
            return solution
        return None
    
    def _detect_simple_cycle(self, new_goals: List[Goal], current_goal: Goal) -> bool:
        """
        Detect simple cycles in goal expansion to prevent infinite recursion
        
        Args:
            new_goals: The new goals that would be added
            current_goal: The current goal being expanded
            
        Returns:
            True if a cycle is detected
        """
        # Very simple cycle detection: check if the same goal appears in new_goals
        current_term_str = str(current_goal.term)
        for goal in new_goals:
            if str(goal.term) == current_term_str:
                return True
        return False
