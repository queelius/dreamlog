"""
Knowledge representation for DreamLog

This module defines facts, rules, and the knowledge base structure.
"""

from typing import List, Dict, Any, Set, Optional, Iterator
from dataclasses import dataclass
from .terms import Term, term_from_prefix
import json


@dataclass(frozen=True)
class Fact:
    """Represents a fact - a ground term that is assumed true"""
    term: Term
    
    def get_variables(self) -> Set[str]:
        """Get all variables in this fact"""
        return self.term.get_variables()
    
    def substitute(self, bindings: Dict[str, Term]) -> 'Fact':
        """Apply substitutions to create a new fact"""
        return Fact(self.term.substitute(bindings))
    
    def to_prefix(self) -> List[Any]:
        """Convert to prefix notation: ["fact", term]"""
        return ["fact", self.term.to_prefix()]
    
    @classmethod
    def from_prefix(cls, data: List[Any]) -> 'Fact':
        """Create from prefix notation"""
        if not isinstance(data, list) or len(data) != 2 or data[0] != "fact":
            raise ValueError(f"Expected ['fact', term], got {data}")
        return cls(term_from_prefix(data[1]))
    
    def __str__(self) -> str:
        return f"{self.term}."


@dataclass(frozen=True)
class Rule:
    """Represents a rule - a logical implication"""
    head: Term
    body: tuple[Term, ...]  # Immutable tuple
    
    def __init__(self, head: Term, body: List[Term]):
        object.__setattr__(self, 'head', head)
        object.__setattr__(self, 'body', tuple(body))
    
    @property
    def is_fact(self) -> bool:
        """True if this rule has no body (is a fact)"""
        return len(self.body) == 0
    
    def get_variables(self) -> Set[str]:
        """Get all variables in this rule"""
        variables = self.head.get_variables()
        for term in self.body:
            variables.update(term.get_variables())
        return variables
    
    def substitute(self, bindings: Dict[str, Term]) -> 'Rule':
        """Apply substitutions to create a new rule"""
        new_head = self.head.substitute(bindings)
        new_body = [term.substitute(bindings) for term in self.body]
        return Rule(new_head, new_body)
    
    def rename_variables(self, suffix: str = "") -> 'Rule':
        """Rename all variables in this rule to avoid conflicts"""
        variables = self.get_variables()
        bindings = {}
        
        from .terms import Variable
        for var_name in variables:
            new_name = f"{var_name}_{suffix}" if suffix else f"{var_name}_renamed"
            bindings[var_name] = Variable(new_name)
        
        return self.substitute(bindings)
    
    def to_prefix(self) -> List[Any]:
        """Convert to prefix notation: ["rule", head, [body1, body2, ...]]"""
        return ["rule", self.head.to_prefix(), [t.to_prefix() for t in self.body]]
    
    @classmethod
    def from_prefix(cls, data: List[Any]) -> 'Rule':
        """Create from prefix notation"""
        if not isinstance(data, list) or len(data) != 3 or data[0] != "rule":
            raise ValueError(f"Expected ['rule', head, body], got {data}")
        
        head = term_from_prefix(data[1])
        body = [term_from_prefix(term_data) for term_data in data[2]]
        return cls(head, body)
    
    def __str__(self) -> str:
        if self.is_fact:
            return f"{self.head}."
        
        body_str = ", ".join(str(term) for term in self.body)
        return f"{self.head} :- {body_str}."


class KnowledgeBase:
    """Container for facts and rules with efficient lookup"""
    
    def __init__(self):
        self._facts: List[Fact] = []
        self._rules: List[Rule] = []
        
        # Index for efficient lookup by functor/arity
        self._fact_index: Dict[tuple[str, int], List[Fact]] = {}
        self._rule_index: Dict[tuple[str, int], List[Rule]] = {}
    
    def add_fact(self, fact: Fact) -> None:
        """Add a fact to the knowledge base"""
        if fact not in self._facts:
            self._facts.append(fact)
            self._index_fact(fact)
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the knowledge base"""
        if rule not in self._rules:
            self._rules.append(rule)
            self._index_rule(rule)
    
    def _index_fact(self, fact: Fact) -> None:
        """Add fact to the index"""
        key = self._get_term_key(fact.term)
        if key:
            if key not in self._fact_index:
                self._fact_index[key] = []
            self._fact_index[key].append(fact)
    
    def _index_rule(self, rule: Rule) -> None:
        """Add rule to the index"""
        key = self._get_term_key(rule.head)
        if key:
            if key not in self._rule_index:
                self._rule_index[key] = []
            self._rule_index[key].append(rule)
    
    def _get_term_key(self, term: Term) -> Optional[tuple[str, int]]:
        """Get indexing key for a term"""
        from .terms import Compound, Atom
        
        if isinstance(term, Compound):
            return (term.functor, term.arity)
        elif isinstance(term, Atom):
            return (term.value, 0)
        return None
    
    def get_matching_facts(self, term: Term) -> Iterator[Fact]:
        """Get facts that might unify with the given term"""
        key = self._get_term_key(term)
        if key and key in self._fact_index:
            yield from self._fact_index[key]
    
    def get_matching_rules(self, term: Term) -> Iterator[Rule]:
        """Get rules whose head might unify with the given term"""
        key = self._get_term_key(term)
        if key and key in self._rule_index:
            yield from self._rule_index[key]
    
    @property
    def facts(self) -> List[Fact]:
        """Get all facts"""
        return self._facts.copy()
    
    @property
    def rules(self) -> List[Rule]:
        """Get all rules"""
        return self._rules.copy()
    
    def clear(self) -> None:
        """Clear all facts and rules"""
        self._facts.clear()
        self._rules.clear()
        self._fact_index.clear()
        self._rule_index.clear()
    
    def to_prefix(self) -> str:
        """Export knowledge base to prefix notation JSON"""
        data = []
        for fact in self._facts:
            data.append(fact.to_prefix())
        for rule in self._rules:
            data.append(rule.to_prefix())
        return json.dumps(data, indent=2)
    
    def from_prefix(self, json_str: str) -> None:
        """Import knowledge base from prefix notation JSON"""
        data = json.loads(json_str)
        
        # Clear existing content
        self.clear()
        
        # Load facts and rules
        for item in data:
            if isinstance(item, list) and len(item) > 0:
                if item[0] == "fact":
                    self.add_fact(Fact.from_prefix(item))
                elif item[0] == "rule":
                    self.add_rule(Rule.from_prefix(item))
                else:
                    raise ValueError(f"Unknown item type: {item[0]}")
    
    def __len__(self) -> int:
        """Total number of facts and rules"""
        return len(self._facts) + len(self._rules)
    
    def __str__(self) -> str:
        """String representation of the knowledge base"""
        lines = []
        
        if self._facts:
            lines.append("Facts:")
            for fact in self._facts:
                lines.append(f"  {fact}")
        
        if self._rules:
            if lines:
                lines.append("")
            lines.append("Rules:")
            for rule in self._rules:
                lines.append(f"  {rule}")
        
        return "\n".join(lines) if lines else "Empty knowledge base"
