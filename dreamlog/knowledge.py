"""
Knowledge representation for DreamLog

This module defines facts, rules, and the knowledge base structure.
"""

from typing import List, Dict, Any, Set, Optional, Iterator, Union
from dataclasses import dataclass
from .terms import Term
from .factories import term_from_prefix
from .unification import is_ground
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

        # Usage frequency tracking (clause hash -> count)
        self._usage_counts: Dict[int, int] = {}

        # Derivation tracking (term hash -> (count, term))
        self._derivation_counts: Dict[int, int] = {}
        self._derivation_terms: Dict[int, Term] = {}
        self._derivation_tracking: bool = False
    
    def add_fact(self, fact: Union[Fact, Term]) -> None:
        """
        Add a fact to the knowledge base.

        Raises:
            ValueError: If the fact contains variables (not ground)
        """
        # Convert Term to Fact if needed
        if isinstance(fact, Term):
            term = fact
            fact = Fact(fact)
        else:
            term = fact.term

        # Validate: facts must be ground (no variables)
        if not is_ground(term):
            variables = term.get_variables()
            var_list = ", ".join(sorted(variables))
            raise ValueError(f"Facts cannot contain variables. Found: {var_list}")

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
        """
        Get indexing key for a term.
        
        Returns None for Variables as they cannot be indexed directly
        (they match everything). Raises TypeError for unknown term types.
        
        Args:
            term: Term to get the key for
            
        Returns:
            Tuple of (functor/value, arity) or None for Variables
            
        Raises:
            TypeError: If term is not a recognized Term subclass
        """
        from .terms import Compound, Atom, Variable
        
        if isinstance(term, Variable):
            # Variables are wildcards that match anything, cannot index
            return None
        elif isinstance(term, Compound):
            return (term.functor, term.arity)
        elif isinstance(term, Atom):
            return (term.value, 0)
        else:
            raise TypeError(f"Unknown term type: {type(term).__name__}. "
                          f"Expected Variable, Compound, or Atom.")
    
    def get_matching_facts(self, term: Term) -> Iterator[Fact]:
        """
        Get facts that might unify with the given term.
        
        Uses functor/arity indexing for efficient lookup.
        Variables in the query term will match any fact.
        
        Args:
            term: The term to match against
        
        Yields:
            Facts that have the same functor and arity as the term
        
        Examples:
            >>> kb.add_fact(Fact(compound("parent", atom("john"), atom("mary"))))
            >>> list(kb.get_matching_facts(compound("parent", var("X"), var("Y"))))
            [Fact(Compound("parent", [Atom("john"), Atom("mary")]))]
        """
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
    
    def remove_fact(self, index: int) -> Fact:
        """
        Remove a fact by index.

        Args:
            index: Index of the fact to remove (0-based)

        Returns:
            The removed Fact

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self._facts):
            raise IndexError(f"Fact index {index} out of range (0-{len(self._facts)-1})")

        fact = self._facts[index]

        # Remove from list
        self._facts.pop(index)

        # Rebuild fact index (simpler than trying to update it)
        self._fact_index.clear()
        for f in self._facts:
            self._index_fact(f)

        return fact

    def remove_rule(self, index: int) -> Rule:
        """
        Remove a rule by index.

        Args:
            index: Index of the rule to remove (0-based)

        Returns:
            The removed Rule

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self._rules):
            raise IndexError(f"Rule index {index} out of range (0-{len(self._rules)-1})")

        rule = self._rules[index]

        # Remove from list
        self._rules.pop(index)

        # Rebuild rule index
        self._rule_index.clear()
        for r in self._rules:
            self._index_rule(r)

        return rule

    def clear(self) -> None:
        """Clear all facts and rules"""
        self._facts.clear()
        self._rules.clear()
        self._fact_index.clear()
        self._rule_index.clear()

    def _rebuild_indices(self) -> None:
        """Rebuild fact and rule indices from current lists"""
        self._fact_index.clear()
        self._rule_index.clear()
        for fact in self._facts:
            self._index_fact(fact)
        for rule in self._rules:
            self._index_rule(rule)
    
    def record_usage(self, clause: Union[Fact, Rule]) -> None:
        """Increment usage counter for a clause."""
        key = hash(clause)
        self._usage_counts[key] = self._usage_counts.get(key, 0) + 1

    def get_usage(self, clause: Union[Fact, Rule]) -> int:
        """Get usage count for a clause (0 if never used)."""
        return self._usage_counts.get(hash(clause), 0)

    def reset_usage(self) -> None:
        """Clear all usage counters."""
        self._usage_counts.clear()

    def total_queries_tracked(self) -> int:
        """Total usage events recorded."""
        return sum(self._usage_counts.values())

    # Derivation tracking methods

    def enable_derivation_tracking(self) -> None:
        self._derivation_tracking = True

    def disable_derivation_tracking(self) -> None:
        self._derivation_tracking = False

    def record_derivation(self, term: Term) -> None:
        """Record that a ground term was successfully derived."""
        if not self._derivation_tracking:
            return
        key = hash(term)
        self._derivation_counts[key] = self._derivation_counts.get(key, 0) + 1
        if key not in self._derivation_terms:
            self._derivation_terms[key] = term

    def get_derivation_count(self, term: Term) -> int:
        return self._derivation_counts.get(hash(term), 0)

    def get_frequent_derivations(self, min_count: int = 5) -> list:
        """Return (term, count) pairs for terms derived >= min_count times
        that are NOT already stored as facts."""
        fact_hashes = {hash(f.term) for f in self._facts}
        results = []
        for key, count in self._derivation_counts.items():
            if count >= min_count and key not in fact_hashes:
                term = self._derivation_terms.get(key)
                if term is not None:
                    results.append((term, count))
        results.sort(key=lambda x: -x[1])
        return results

    def reset_derivations(self) -> None:
        self._derivation_counts.clear()
        self._derivation_terms.clear()

    def copy(self) -> 'KnowledgeBase':
        """Deep copy for rollback."""
        new_kb = KnowledgeBase()
        for fact in self._facts:
            new_kb.add_fact(fact)
        for rule in self._rules:
            new_kb.add_rule(rule)
        new_kb._usage_counts = dict(self._usage_counts)
        new_kb._derivation_counts = dict(self._derivation_counts)
        new_kb._derivation_terms = dict(self._derivation_terms)
        new_kb._derivation_tracking = self._derivation_tracking
        return new_kb

    def restore_from(self, other: 'KnowledgeBase') -> None:
        """Replace contents with another KB's contents."""
        self.clear()
        for fact in other._facts:
            self.add_fact(fact)
        for rule in other._rules:
            self.add_rule(rule)
        self._usage_counts = dict(other._usage_counts)

    def remove_fact_by_value(self, fact: Fact) -> None:
        """Remove a fact by equality."""
        try:
            idx = self._facts.index(fact)
        except ValueError:
            raise ValueError(f"Fact not found: {fact}")
        self.remove_fact(idx)

    def remove_rule_by_value(self, rule: Rule) -> None:
        """Remove a rule by equality."""
        try:
            idx = self._rules.index(rule)
        except ValueError:
            raise ValueError(f"Rule not found: {rule}")
        self.remove_rule(idx)

    def replace_facts(self, old: List[Fact],
                      new: List[Union[Fact, Rule]]) -> None:
        """Atomic replacement: remove old facts, add new facts/rules."""
        for fact in old:
            self.remove_fact_by_value(fact)
        for item in new:
            if isinstance(item, Fact):
                self.add_fact(item)
            elif isinstance(item, Rule):
                self.add_rule(item)
            elif isinstance(item, Term):
                self.add_fact(item)
            else:
                raise TypeError(f"Expected Fact or Rule, got {type(item)}")

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
