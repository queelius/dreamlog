# dreamlog/recursive_discovery.py
"""Recursive rule discovery (Operation I): detect transitive-closure
relationships between binary predicates and synthesize right-recursive rules.

The pure closure math lives here so it can be tested without a knowledge base.
"""
from collections import defaultdict, deque
from typing import Set, Tuple, Hashable

Pair = Tuple[Hashable, Hashable]


def transitive_closure(edges: Set[Pair]) -> Set[Pair]:
    """Irreflexive transitive closure of a binary relation given as edge pairs.

    edges: set of (a, b) value pairs. Returns every (a, b) such that there is
    a non-empty directed path a -> ... -> b. On a DAG the result is
    irreflexive; on a cyclic relation a node reachable from itself yields (x, x).
    """
    adj = defaultdict(set)
    nodes = set()
    for a, b in edges:
        adj[a].add(b)
        nodes.add(a)
        nodes.add(b)

    closure: Set[Pair] = set()
    for start in nodes:
        seen = set()
        queue = deque(adj[start])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            closure.add((start, node))
            queue.extend(adj[node])
    return closure
