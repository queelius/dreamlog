"""
Term representations for JLOG

This module defines the core term types used in JLOG:
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
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON representation"""
        pass
    
    @classmethod
    @abstractmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Term':
        """Create term from JSON representation"""
        pass
    
    def __eq__(self, other) -> bool:
        """Terms are equal if their JSON representations are equal"""
        if not isinstance(other, Term):
            return False
        return self.to_json() == other.to_json()
    
    def __hash__(self) -> int:
        """Hash based on JSON representation"""
        return hash(json.dumps(self.to_json(), sort_keys=True))


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
    
    def to_json(self) -> Dict[str, Any]:
        return {"type": "atom", "value": self.value}
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Atom':
        if data["type"] != "atom":
            raise ValueError(f"Expected atom, got {data['type']}")
        return cls(data["value"])
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Variable(Term):
    """Represents a logical variable"""
    name: str
    
    # def substitute(self, bindings: Dict[str, Term]) -> Term:
    #     """Return the binding if it exists, otherwise return self"""
    #     return bindings.get(self.name, self)
    
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
    
    def to_json(self) -> Dict[str, Any]:
        return {"type": "variable", "name": self.name}
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Variable':
        if data["type"] != "variable":
            raise ValueError(f"Expected variable, got {data['type']}")
        return cls(data["name"])
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Compound(Term):
    """Represents a compound term with functor and arguments"""
    functor: str
    args: tuple[Term, ...]  # Using tuple for immutability
    
    def __init__(self, functor: str, args: List[Term]):
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
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "compound",
            "functor": self.functor,
            "args": [arg.to_json() for arg in self.args]
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Compound':
        if data["type"] != "compound":
            raise ValueError(f"Expected compound, got {data['type']}")
        
        args = []
        for arg_data in data["args"]:
            args.append(term_from_json(arg_data))
        
        return cls(data["functor"], args)
    
    def __str__(self) -> str:
        if not self.args:
            return self.functor
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.functor}({args_str})"


def term_from_json(data: Dict[str, Any]) -> Term:
    """Factory function to create appropriate term from JSON"""
    if "type" not in data:
        raise KeyError("Missing 'type' field in JSON data")
    
    term_type = data["type"]
    
    if term_type == "atom":
        return Atom.from_json(data)
    elif term_type == "variable":
        return Variable.from_json(data)
    elif term_type == "compound":
        return Compound.from_json(data)
    else:
        raise ValueError(f"Unknown term type: {term_type}")


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
