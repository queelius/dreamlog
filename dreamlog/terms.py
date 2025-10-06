"""
Term representations for DreamLog

This module defines the core term types used in DreamLog:
- Atom: Constants like 'john', 'mary', etc.
- Variable: Variables like 'X', 'Y', etc. 
- Compound: Complex terms like 'parent(john, mary)'
"""

from typing import List, Dict, Any, Set, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


class Term(ABC):
    """Abstract base class for all terms"""
    
    @abstractmethod
    def substitute(self, bindings: Dict[str, 'Term']) -> 'Term':
        """
        Apply variable substitutions to this term.
        
        Args:
            bindings: Dictionary mapping variable names to terms.
                     Chains of variables are followed transitively.
        
        Returns:
            New term with all variables replaced according to bindings.
            Returns self if no substitutions apply.
        
        Examples:
            >>> term = compound("p", var("X"), var("Y"))
            >>> term.substitute({"X": atom("a"), "Y": atom("b")})
            Compound("p", [Atom("a"), Atom("b")])
        """
        pass
    
    @abstractmethod
    def get_variables(self) -> Set[str]:
        """Get all variable names in this term"""
        pass
    
    @abstractmethod
    def to_prefix(self) -> Any:
        """Convert to prefix notation (S-expression style)"""
        pass
    
    def __eq__(self, other) -> bool:
        """Terms are equal if their prefix representations are equal"""
        if not isinstance(other, Term):
            return False
        return self.to_prefix() == other.to_prefix()
    
    def __hash__(self) -> int:
        """Hash based on prefix representation"""
        # Use repr for safer hashing that won't fail on non-JSON-serializable data
        return hash(repr(self.to_prefix()))


@dataclass(frozen=True)
class Atom(Term):
    """Represents an atomic constant"""
    value: str
    
    def substitute(self, bindings: Dict[str, Term]) -> Term:
        """Atoms are not affected by substitution"""
        return self
    
    def get_variables(self) -> Set[str]:
        """Atoms contain no variables"""
        return set()
    
    def to_prefix(self) -> Any:
        """Atom is just its value in prefix notation"""
        return self.value
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Variable(Term):
    """Represents a logical variable"""
    name: str
    
    
    def substitute(self, subst: Dict[str, 'Term']) -> 'Term':
        """Apply substitution to this variable, following chains"""
        from .utils import dereference_variable
        return dereference_variable(self, subst)
    
    def get_variables(self) -> Set[str]:
        """Variables contain themselves"""
        return {self.name}
    
    def to_prefix(self) -> Any:
        """Variable is just its name in prefix notation"""
        return self.name
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Compound(Term):
    """Represents a compound term with functor and arguments"""
    functor: str
    args: tuple[Term, ...]  # Using tuple for immutability
    
    def __init__(self, functor: str, args: List[Term]):
        if not functor:
            raise ValueError("Compound term functor cannot be empty")
        if not isinstance(args, (list, tuple)):
            raise TypeError(f"Compound args must be list or tuple, got {type(args)}")
        object.__setattr__(self, 'functor', functor)
        object.__setattr__(self, 'args', tuple(args))
    
    @property
    def arity(self) -> int:
        """Number of arguments"""
        return len(self.args)
    
    def substitute(self, bindings: Dict[str, Term]) -> Term:
        """Apply substitutions to all arguments"""
        new_args = [arg.substitute(bindings) for arg in self.args]
        return Compound(self.functor, new_args)
    
    def get_variables(self) -> Set[str]:
        """Get variables from all arguments"""
        variables = set()
        for arg in self.args:
            variables.update(arg.get_variables())
        return variables
    
    def to_prefix(self) -> List[Any]:
        """Compound is [functor, arg1, arg2, ...] in prefix notation"""
        result = [self.functor]
        for arg in self.args:
            result.append(arg.to_prefix())
        return result
    
    def __str__(self) -> str:
        # Use S-expression format as the default string representation
        if not self.args:
            return f"({self.functor})"
        args_str = " ".join(str(arg) for arg in self.args)
        return f"({self.functor} {args_str})"


# Note: Factory functions moved to factories.py to avoid circular imports
# Import them from dreamlog.factories or dreamlog.__init__
