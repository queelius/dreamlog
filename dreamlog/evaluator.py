"""
Prolog-style evaluator for DreamLog

Implements SLD resolution for query evaluation with proper backtracking.
"""

from typing import List, Dict, Iterator, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from .terms import Term, Atom, Variable, Compound
from .factories import var
from .knowledge import KnowledgeBase, Fact, Rule
from .unification import unify, compose_substitutions, is_ground


class FlounderingError(Exception):
    """Raised when not/1 is applied to a non-ground goal."""


class InstantiationError(Exception):
    """Raised when call/N receives a non-ground or invalid functor."""


@dataclass
class Goal:
    """Represents a goal to be solved"""
    term: Term
    bindings: Dict[str, Term]
    
    def substitute(self, new_bindings: Dict[str, Term]) -> 'Goal':
        """Apply substitutions to create a new goal"""
        combined_bindings = compose_substitutions(self.bindings, new_bindings)
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
            if is_ground(final_term):
                ground_bindings[var_name] = final_term
        return ground_bindings


class PrologEvaluator:
    """
    Evaluates Prolog queries using SLD resolution
    """
    
    def __init__(self, knowledge_base: KnowledgeBase,
                 unknown_hook: Optional[Callable[[Term, 'PrologEvaluator'], None]] = None,
                 max_total_calls: int = 0):
        self.kb = knowledge_base
        self.unknown_hook = unknown_hook
        self._recursion_depth = 0
        self._max_recursion_depth = 100
        self._total_calls = 0
        self._max_total_calls = max_total_calls  # 0 = unlimited
    
    @contextmanager
    def _track_recursion(self):
        """Track recursion depth; raises RecursionError if limit exceeded."""
        self._recursion_depth += 1
        self._total_calls += 1
        if self._recursion_depth > self._max_recursion_depth:
            self._recursion_depth -= 1
            raise RecursionError(f"Maximum recursion depth ({self._max_recursion_depth}) exceeded")
        if self._max_total_calls and self._total_calls > self._max_total_calls:
            self._recursion_depth -= 1
            raise RecursionError(f"Maximum total calls ({self._max_total_calls}) exceeded")
        try:
            yield
        finally:
            self._recursion_depth -= 1
    
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

        self._recursion_depth = 0
        # _total_calls persists across query() invocations on the same
        # evaluator instance. Callers can enforce a per-evaluator budget
        # by reusing one evaluator across queries, or a per-query budget
        # by creating a fresh evaluator each time. VerificationSuite.verify
        # uses the per-query pattern so one slow query cannot starve others.
        initial_goals = [Goal(goal, {}) for goal in goals]
        
        # Generate solutions using SLD resolution
        for solution in self._solve_goals(initial_goals, {}):
            # Record derived terms for lemma caching
            for goal in goals:
                resolved = goal.substitute(solution.bindings)
                if is_ground(resolved):
                    self.kb.record_derivation(resolved)
            yield solution
    
    def query_with_proof(self, goals: List[Term]):
        """Evaluate a query and yield (Solution, ProofNode) pairs.

        Like query(), but also constructs proof trees showing how each
        solution was derived. More expensive than query() due to tree
        allocation.
        """
        from .proof_tree import ProofNode

        if not goals:
            return

        self._recursion_depth = 0
        self._total_calls = 0
        initial_goals = [Goal(goal, {}) for goal in goals]

        for solution, proof_nodes in self._solve_goals_with_proof(
                initial_goals, {}, 0):
            # Record derived terms
            for goal in goals:
                resolved = goal.substitute(solution.bindings)
                if is_ground(resolved):
                    self.kb.record_derivation(resolved)
            # Build top-level proof: one node per goal
            if len(proof_nodes) == 1:
                yield solution, proof_nodes[0]
            else:
                root = ProofNode(
                    goal=goals[0] if goals else Atom("query"),
                    children=proof_nodes, depth=0)
                yield solution, root

    def _solve_goals_with_proof(self, goals, global_bindings, depth):
        """Like _solve_goals but yields (Solution, [ProofNode]) pairs."""
        from .proof_tree import ProofNode

        try:
            with self._track_recursion():
                if not goals:
                    yield Solution(global_bindings), []
                    return

                current_goal = goals[0]
                remaining_goals = goals[1:]
                current_goal = current_goal.substitute(global_bindings)

                # NAF and call/N handling (simplified: delegate to normal solve)
                if isinstance(current_goal.term, Compound):
                    if current_goal.term.functor == "not" and current_goal.term.arity == 1:
                        # For NAF, just use the non-proof path
                        for sol in self._solve_goals([current_goal] + remaining_goals, global_bindings):
                            yield sol, [ProofNode(goal=current_goal.term, depth=depth)]
                        return
                    if current_goal.term.functor == "call" and current_goal.term.arity >= 2:
                        for sol in self._solve_goals([current_goal] + remaining_goals, global_bindings):
                            yield sol, [ProofNode(goal=current_goal.term, depth=depth)]
                        return

                # Try facts
                for fact in self.kb.get_matching_facts(current_goal.term):
                    renamed_fact = self._rename_variables_in_fact(fact)
                    bindings = unify(current_goal.term, renamed_fact.term, current_goal.bindings)
                    if bindings is not None:
                        self.kb.record_usage(fact)
                        new_global = compose_substitutions(global_bindings, bindings)
                        new_remaining = [g.substitute(bindings) for g in remaining_goals]
                        for sol, rest_proof in self._solve_goals_with_proof(
                                new_remaining, new_global, depth + 1):
                            node = ProofNode(goal=current_goal.term.substitute(bindings),
                                             clause=fact, depth=depth)
                            yield sol, [node] + rest_proof

                # Try rules
                for rule in self.kb.get_matching_rules(current_goal.term):
                    renamed_rule = self._rename_variables_in_rule(rule)
                    bindings = unify(current_goal.term, renamed_rule.head, current_goal.bindings)
                    if bindings is not None:
                        self.kb.record_usage(rule)
                        new_goals = [Goal(bt, bindings) for bt in renamed_rule.body]
                        for g in remaining_goals:
                            new_goals.append(g.substitute(bindings))
                        new_global = compose_substitutions(global_bindings, bindings)

                        if self._detect_simple_cycle(new_goals, current_goal):
                            continue

                        body_len = len(renamed_rule.body)
                        for sol, child_proofs in self._solve_goals_with_proof(
                                new_goals, new_global, depth + 1):
                            body_proofs = child_proofs[:body_len]
                            rest_proofs = child_proofs[body_len:]
                            node = ProofNode(
                                goal=current_goal.term.substitute(bindings),
                                clause=rule, children=body_proofs, depth=depth)
                            yield sol, [node] + rest_proofs

        except RecursionError:
            return

    def _solve_goals(self, goals: List[Goal], global_bindings: Dict[str, Term]) -> Iterator[Solution]:
        """Solve a list of goals using SLD resolution."""
        try:
            with self._track_recursion():
                if not goals:
                    yield Solution(global_bindings)
                    return

                current_goal = goals[0]
                remaining_goals = goals[1:]
                current_goal = current_goal.substitute(global_bindings)

                # Handle negation as failure: not/1
                if (isinstance(current_goal.term, Compound)
                        and current_goal.term.functor == "not"
                        and current_goal.term.arity == 1):
                    inner_resolved = current_goal.term.args[0].substitute(global_bindings)
                    if inner_resolved.get_variables():
                        raise FlounderingError(
                            f"not/1 applied to non-ground goal: {inner_resolved}")
                    naf_evaluator = PrologEvaluator(self.kb, unknown_hook=None)
                    naf_evaluator._max_recursion_depth = self._max_recursion_depth
                    if not naf_evaluator.has_solution(inner_resolved):
                        yield from self._solve_goals(
                            [g.substitute(global_bindings) for g in remaining_goals],
                            global_bindings)
                    return

                # Handle call/N meta-predicate
                if (isinstance(current_goal.term, Compound)
                        and current_goal.term.functor == "call"):
                    if current_goal.term.arity < 2:
                        raise InstantiationError(
                            "call/N requires at least 2 arguments: call(Functor, Arg1, ...)")
                    functor_arg = current_goal.term.args[0].substitute(global_bindings)
                    if isinstance(functor_arg, Variable):
                        raise InstantiationError(
                            f"call/N: first argument is unbound variable: {functor_arg}")
                    if not isinstance(functor_arg, Atom):
                        raise InstantiationError(
                            f"call/N: first argument must be an atom, got: {functor_arg}")
                    constructed = Compound(functor_arg.value, list(current_goal.term.args[1:]))
                    new_goals = [Goal(constructed, current_goal.bindings)] + list(remaining_goals)
                    yield from self._solve_goals(new_goals, global_bindings)
                    return

                solutions_found = False

                # Try facts
                for fact in self.kb.get_matching_facts(current_goal.term):
                    renamed_fact = self._rename_variables_in_fact(fact)
                    bindings = unify(current_goal.term, renamed_fact.term, current_goal.bindings)
                    if bindings is not None:
                        solutions_found = True
                        self.kb.record_usage(fact)
                        new_global = compose_substitutions(global_bindings, bindings)
                        new_remaining = [g.substitute(bindings) for g in remaining_goals]
                        yield from self._solve_goals(new_remaining, new_global)

                # Try rules (with cycle detection)
                for rule in self.kb.get_matching_rules(current_goal.term):
                    renamed_rule = self._rename_variables_in_rule(rule)
                    bindings = unify(current_goal.term, renamed_rule.head, current_goal.bindings)
                    if bindings is not None:
                        solutions_found = True
                        self.kb.record_usage(rule)
                        new_goals = [Goal(bt, bindings) for bt in renamed_rule.body]
                        new_goals += [g.substitute(bindings) for g in remaining_goals]
                        new_global = compose_substitutions(global_bindings, bindings)
                        if self._detect_simple_cycle(new_goals, current_goal):
                            continue
                        yield from self._solve_goals(new_goals, new_global)

                # If no solutions found and we have an unknown hook, try it
                if not solutions_found and self.unknown_hook and self._recursion_depth < self._max_recursion_depth:
                    original_kb_size = len(self.kb)
                    self.unknown_hook(current_goal.term, self)
                    if len(self.kb) > original_kb_size:
                        yield from self._solve_goals(goals, global_bindings)
        except RecursionError:
            return
    
    def _rename_variables_in_fact(self, fact: Fact) -> Fact:
        """Rename variables in a fact to avoid conflicts"""
        suffix = str(self._total_calls)
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
        suffix = str(self._total_calls)
        return rule.rename_variables(suffix)
    
    def ask_yes_no(self, goal: Term) -> bool:
        """Check if a goal has at least one solution."""
        return self.has_solution(goal)
    
    def find_all_solutions(self, goals: List[Term]) -> List[Solution]:
        """Find all solutions to a query"""
        return list(self.query(goals))
    
    def find_first_solution(self, goals: List[Term]) -> Optional[Solution]:
        """Find the first solution to a query"""
        for solution in self.query(goals):
            return solution
        return None

    def has_solution(self, term: Term) -> bool:
        """Check if a term is derivable. Stops at first solution."""
        for _ in self.query([term]):
            return True
        return False

    def _detect_simple_cycle(self, new_goals: List[Goal], current_goal: Goal) -> bool:
        """Detect if current_goal reappears structurally in new_goals."""
        current_prefix = current_goal.term.to_prefix()
        return any(g.term.to_prefix() == current_prefix for g in new_goals)
