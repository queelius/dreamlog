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
        """Apply variable substitutions to this term"""
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
        if self.name in subst:
            result = subst[self.name]
            visited = {self.name}  # Track visited variables to prevent infinite loops
            
            # Follow the chain of variable substitutions
            while isinstance(result, Variable) and result.name in subst and result.name not in visited:
                visited.add(result.name)
                result = subst[result.name]
            
            return result
        return self
    
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
        if not self.args:
            return self.functor
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.functor}({args_str})"


def term_from_prefix(data: Any) -> Term:
    """Factory function to create term from prefix notation"""
    from dreamlog.prefix_parser import parse_prefix_notation
    return parse_prefix_notation(data)


# Convenience functions for creating terms
def atom(value: str) -> Atom:
    """Create an atom"""
    return Atom(value)

def var(name: str) -> Variable:
    """Create a variable"""
    return Variable(name)

def compound(functor: str, *args: Term) -> Compound:
    """Create a compound term"""
    return Compound(functor, list(args))
