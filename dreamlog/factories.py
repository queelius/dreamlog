"""
Factory functions for creating DreamLog terms and knowledge structures.

This module provides convenience functions for creating terms without
circular import issues.
"""

from typing import Any, List
from .terms import Term, Atom, Variable, Compound


def atom(value: Any) -> Atom:
    """
    Create an atomic term.
    
    Args:
        value: The value for the atom (string, number, etc.)
    
    Returns:
        An Atom instance
    
    Examples:
        >>> atom("john")
        Atom("john")
        >>> atom(42)
        Atom(42)
    """
    return Atom(value)


def var(name: str) -> Variable:
    """
    Create a logical variable.
    
    Variables should start with uppercase letters by convention.
    
    Args:
        name: The variable name
    
    Returns:
        A Variable instance
    
    Examples:
        >>> var("X")
        Variable("X")
        >>> var("Person")
        Variable("Person")
    """
    return Variable(name)


def compound(functor: str, *args: Term) -> Compound:
    """
    Create a compound term.
    
    Args:
        functor: The functor/predicate name
        *args: The arguments (Terms)
    
    Returns:
        A Compound instance
    
    Examples:
        >>> compound("parent", atom("john"), atom("mary"))
        Compound("parent", [Atom("john"), Atom("mary")])
        >>> compound("likes", var("X"), atom("pizza"))
        Compound("likes", [Variable("X"), Atom("pizza")])
    """
    return Compound(functor, list(args))


def term_from_prefix(data: Any) -> Term:
    """
    Create a term from prefix notation.
    
    This function avoids circular imports by importing the parser
    only when needed.
    
    Args:
        data: Prefix notation data (list or primitive)
    
    Returns:
        The parsed Term
    
    Examples:
        >>> term_from_prefix(["parent", "john", "mary"])
        Compound("parent", [Atom("john"), Atom("mary")])
        >>> term_from_prefix("X")
        Variable("X")
    """
    from .prefix_parser import parse_prefix_notation
    return parse_prefix_notation(data)