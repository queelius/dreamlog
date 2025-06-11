"""
Unification algorithm for JLOG

Implements the unification algorithm with occurs check to prevent infinite structures.
"""

from typing import Dict, Optional, Set
from .terms import Term, Atom, Variable, Compound


class Unifier:
    """Implements the unification algorithm"""
    
    @staticmethod
    def unify(term1: Term, term2: Term, bindings: Optional[Dict[str, Term]] = None) -> Optional[Dict[str, Term]]:
        """
        Unify two terms, returning the most general unifier (MGU) or None if unification fails.
        
        Args:
            term1: First term to unify
            term2: Second term to unify 
            bindings: Existing variable bindings (optional)
            
        Returns:
            Dictionary of variable bindings if successful, None if unification fails
        """
        if bindings is None:
            bindings = {}
        
        # Apply existing bindings
        term1 = term1.substitute(bindings)
        term2 = term2.substitute(bindings)
        
        # Same term
        if term1 == term2:
            return bindings
        
        # Variable unification
        if isinstance(term1, Variable):
            return Unifier._unify_variable(term1, term2, bindings)
        elif isinstance(term2, Variable):
            return Unifier._unify_variable(term2, term1, bindings)
        
        # Compound term unification
        if isinstance(term1, Compound) and isinstance(term2, Compound):
            return Unifier._unify_compound(term1, term2, bindings)
        
        # Atom unification (already handled by equality check above)
        # Different atoms or incompatible types cannot unify
        return None
    
    @staticmethod
    def _unify_variable(var: Variable, term: Term, bindings: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Unify a variable with a term"""
        
        # Check if variable is already bound
        if var.name in bindings:
            return Unifier.unify(bindings[var.name], term, bindings)
        
        # Check if term is a variable that's already bound
        if isinstance(term, Variable) and term.name in bindings:
            return Unifier.unify(var, bindings[term.name], bindings)
        
        # Occurs check to prevent infinite structures
        if Unifier._occurs_check(var.name, term, bindings):
            return None
        
        # Create new binding
        new_bindings = bindings.copy()
        new_bindings[var.name] = term
        return new_bindings
    
    @staticmethod
    def _unify_compound(comp1: Compound, comp2: Compound, bindings: Dict[str, Term]) -> Optional[Dict[str, Term]]:
        """Unify two compound terms"""
        
        # Must have same functor and arity
        if comp1.functor != comp2.functor or comp1.arity != comp2.arity:
            return None
        
        # Unify all arguments pairwise
        current_bindings = bindings
        for arg1, arg2 in zip(comp1.args, comp2.args):
            current_bindings = Unifier.unify(arg1, arg2, current_bindings)
            if current_bindings is None:
                return None
        
        return current_bindings
    
    @staticmethod
    def _occurs_check(var_name: str, term: Term, bindings: Dict[str, Term]) -> bool:
        """
        Check if variable occurs in term (prevents infinite structures).
        
        Returns True if variable occurs in term, False otherwise.
        """
        # Apply bindings to get the actual term
        term = term.substitute(bindings)
        
        if isinstance(term, Variable):
            return var_name == term.name
        elif isinstance(term, Compound):
            return any(Unifier._occurs_check(var_name, arg, bindings) for arg in term.args)
        else:
            return False
    
    @staticmethod
    def apply_bindings(term: Term, bindings: Dict[str, Term]) -> Term:
        """Apply variable bindings to a term"""
        return term.substitute(bindings)
    
    @staticmethod
    def compose_bindings(bindings1: Dict[str, Term], bindings2: Dict[str, Term]) -> Dict[str, Term]:
        """
        Compose two sets of bindings.
        
        The result applies bindings2 first, then bindings1.
        """
        result = {}
        
        # Apply bindings2 to all terms in bindings1
        for var, term in bindings1.items():
            result[var] = term.substitute(bindings2)
        
        # Add bindings from bindings2 that aren't overridden
        for var, term in bindings2.items():
            if var not in result:
                result[var] = term
        
        return result
    
    @staticmethod
    def ground_term(term: Term, bindings: Dict[str, Term]) -> bool:
        """Check if a term is ground (contains no unbound variables) under the given bindings"""
        term = term.substitute(bindings)
        return len(term.get_variables()) == 0
