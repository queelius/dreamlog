"""
Proof tree recording for DreamLog.

Captures the structure of how queries are resolved: which clause matched
each goal, what sub-goals were generated, and how they were resolved.
This is the dynamic counterpart to static KB analysis.
"""

from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass, field
from .terms import Term, Atom, Compound
from .knowledge import Fact, Rule


@dataclass
class ProofNode:
    """A node in a proof tree.

    Represents the resolution of a single goal.
    """
    goal: Term                                    # the goal that was resolved
    clause: Optional[Union[Fact, Rule]] = None    # the clause used
    children: List['ProofNode'] = field(default_factory=list)
    depth: int = 0

    def subtree_size(self) -> int:
        """Total nodes in this subtree."""
        return 1 + sum(c.subtree_size() for c in self.children)

    def structural_key(self) -> tuple:
        """Hashable key based on structure (clause functors + tree shape),
        ignoring specific atom values. Used for common subtree detection."""
        if self.clause is None:
            return ("leaf", _functor_key(self.goal))

        clause_key = _clause_structure_key(self.clause)
        child_keys = tuple(c.structural_key() for c in self.children)
        return ("node", _functor_key(self.goal), clause_key, child_keys)

    def clause_sequence(self) -> List[Optional[Union[Fact, Rule]]]:
        """Flat list of clauses used in this proof (pre-order traversal)."""
        result = [self.clause]
        for child in self.children:
            result.extend(child.clause_sequence())
        return result

    def __repr__(self):
        if not self.children:
            return f"ProofNode({self.goal})"
        children_str = ", ".join(repr(c) for c in self.children)
        return f"ProofNode({self.goal} <- [{children_str}])"


def _functor_key(term: Term) -> tuple:
    """Extract (functor, arity) from a term, or ('atom',) for atoms."""
    if isinstance(term, Compound):
        return (term.functor, term.arity)
    if isinstance(term, Atom):
        return ("atom",)
    return ("var",)


def _clause_structure_key(clause: Union[Fact, Rule]) -> tuple:
    """Structural key for a clause (functor/arity of head + body goals)."""
    if isinstance(clause, Fact):
        return ("fact", _functor_key(clause.term))
    if isinstance(clause, Rule):
        head_key = _functor_key(clause.head)
        body_keys = tuple(_functor_key(g) for g in clause.body)
        return ("rule", head_key, body_keys)
    return ("unknown",)


class ProofLog:
    """Collects proof trees from query resolution.

    Stores proof trees keyed by query structure. Provides methods
    for finding common subtrees across proofs.
    """

    def __init__(self):
        self._proofs: List[Tuple[Term, ProofNode]] = []
        self._subtree_counts: Dict[tuple, int] = {}

    def add_proof(self, query: Term, tree: ProofNode) -> None:
        """Record a proof tree for a query."""
        self._proofs.append((query, tree))
        self._count_subtrees(tree)

    def _count_subtrees(self, node: ProofNode) -> None:
        """Count occurrences of each structural subtree pattern."""
        key = node.structural_key()
        self._subtree_counts[key] = self._subtree_counts.get(key, 0) + 1
        for child in node.children:
            self._count_subtrees(child)

    def get_common_subtrees(self, min_count: int = 3,
                            min_depth: int = 2) -> List[Tuple[tuple, int]]:
        """Find subtree patterns that appear frequently.

        Returns (structural_key, count) pairs sorted by count descending.
        Only includes subtrees with depth >= min_depth (to filter trivial
        single-fact matches).
        """
        results = []
        for key, count in self._subtree_counts.items():
            if count >= min_count and _key_depth(key) >= min_depth:
                results.append((key, count))
        results.sort(key=lambda x: -x[1])
        return results

    def find_proof_nodes_matching(self, structural_key: tuple
                                  ) -> List[ProofNode]:
        """Find all proof nodes matching a structural key."""
        matches = []
        for _, tree in self._proofs:
            self._collect_matching(tree, structural_key, matches)
        return matches

    def _collect_matching(self, node: ProofNode, key: tuple,
                          matches: List[ProofNode]) -> None:
        if node.structural_key() == key:
            matches.append(node)
        for child in node.children:
            self._collect_matching(child, key, matches)

    @property
    def proof_count(self) -> int:
        return len(self._proofs)

    def clear(self) -> None:
        self._proofs.clear()
        self._subtree_counts.clear()


def _key_depth(key: tuple) -> int:
    """Compute depth of a structural key."""
    if not key or key[0] == "leaf":
        return 1
    if key[0] == "node" and len(key) >= 4:
        children = key[3]
        if children:
            return 1 + max(_key_depth(c) for c in children)
    return 1
