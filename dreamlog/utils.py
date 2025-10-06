"""
Utility functions for DreamLog

Common functionality used across multiple modules.
"""

from typing import Dict, Set
from .terms import Term, Variable


def dereference_variable(var: Variable, bindings: Dict[str, Term]) -> Term:
    """
    Follow variable bindings to find the final value.
    
    This function follows chains of variable-to-variable bindings
    until it finds a non-variable term or an unbound variable.
    Handles circular references by tracking visited variables.
    
    Args:
        var: The variable to dereference
        bindings: Dictionary mapping variable names to terms
    
    Returns:
        The final term after following all bindings, or the 
        original variable if unbound
    
    Examples:
        >>> bindings = {"X": Variable("Y"), "Y": Atom("a")}
        >>> dereference_variable(Variable("X"), bindings)
        Atom("a")
        
        >>> bindings = {"X": Variable("Y"), "Y": Variable("X")}  # Circular
        >>> dereference_variable(Variable("X"), bindings)
        Variable("Y")  # Stops at cycle
    """
    if var.name not in bindings:
        return var
    
    result = bindings[var.name]
    visited = {var.name}
    
    # Follow the chain of variable substitutions
    while isinstance(result, Variable) and result.name in bindings:
        if result.name in visited:
            break  # Circular reference detected
        visited.add(result.name)
        result = bindings[result.name]
    
    return result


def get_all_variables(terms: list[Term]) -> Set[str]:
    """
    Collect all variable names from a list of terms.
    
    Args:
        terms: List of terms to extract variables from
    
    Returns:
        Set of all variable names found
    
    Examples:
        >>> terms = [compound("p", var("X")), compound("q", var("Y"), var("X"))]
        >>> get_all_variables(terms)
        {"X", "Y"}
    """
    variables = set()
    for term in terms:
        variables.update(term.get_variables())
    return variables


def is_ground_term(term: Term) -> bool:
    """
    Check if a term contains no variables.
    
    Args:
        term: The term to check
    
    Returns:
        True if the term contains no variables, False otherwise
    
    Examples:
        >>> is_ground_term(atom("a"))
        True
        >>> is_ground_term(compound("p", atom("a"), atom("b")))
        True
        >>> is_ground_term(compound("p", var("X"), atom("b")))
        False
    """
    return len(term.get_variables()) == 0