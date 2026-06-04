# experiments/ex27_recursion.py
"""EX27: recursive rule discovery on a canonical (ancestor) and an
invented-vocabulary (flux_reaches) transitive-closure domain."""
import random
from typing import List, Tuple

# Invented node names the LLM has not seen as graph nodes.
_INVENTED_NODES = [
    "qux", "vor", "zane", "plix", "drub", "yent", "kosh", "wimple",
    "fren", "glorb", "snee", "thock", "vung", "blee", "morx", "quail",
]


def flux_domain(n_nodes: int = 10, n_extra_edges: int = 3, seed: int = 42
                ) -> Tuple[List[str], List[str]]:
    """Return (base_facts, derived_facts) for the invented closure domain.

    base_facts: flux_links edges forming a random DAG over invented nodes.
    derived_facts: flux_reaches = transitive closure of flux_links.

    Facts use S-expression format: "(flux_links a b)" so they are compatible
    with build_kb / parse_s_expression from ex25_generalization.py.
    """
    from dreamlog.recursive_discovery import transitive_closure
    rng = random.Random(seed)
    nodes = _INVENTED_NODES[:n_nodes]

    # Random DAG: only edges from earlier to later in a shuffled order.
    order = nodes[:]
    rng.shuffle(order)
    edges = set()
    # a spanning chain guarantees a non-trivial closure
    for i in range(len(order) - 1):
        edges.add((order[i], order[i + 1]))
    # a few extra forward edges
    for _ in range(n_extra_edges):
        i = rng.randrange(0, len(order) - 1)
        j = rng.randrange(i + 1, len(order))
        edges.add((order[i], order[j]))

    closure = transitive_closure(edges)
    base = [f"(flux_links {a} {b})" for a, b in sorted(edges)]
    derived = [f"(flux_reaches {a} {b})" for a, b in sorted(closure)]
    return base, derived
