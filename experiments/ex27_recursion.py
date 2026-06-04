# experiments/ex27_recursion.py
"""EX27: recursive rule discovery on a canonical (ancestor) and an
invented-vocabulary (flux_reaches) transitive-closure domain."""
import random
from typing import List, Tuple, Dict

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


# ── New-entity protocol domains ─────────────────────────────────────────────
#
# EX27 uses the new-entity protocol (like EX25/EX25b), NOT holdout: the training
# derived facts are the EXACT transitive closure of the training base, so
# Operation I's exact-match gate fires and discovers the recursive rule. New
# entities then get base edges only, and the checks ask whether the discovered
# rule derives their closure pairs. (Holdout would leave the training closure
# incomplete, so the exact-match gate would never fire -- see the design note in
# the 2026-06-03 spec; this protocol decision was made 2026-06-04.)

_REAL_NAMES = [
    "alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi",
    "ivan", "judy", "mallory", "niaj", "olivia", "peggy", "trent", "victor",
]


def _closure_new_entity_domain(node_pool: List[str], base_pred: str,
                               derived_pred: str, n_train: int = 8,
                               n_new: int = 4, n_extra: int = 3, seed: int = 42):
    """Build a transitive-closure domain with a new-entity test split.

    Returns (base, derived, negatives, new_base, new_checks) in the shape
    run_domain_test expects:
      base      : list of "(base_pred a b)" over training nodes
      derived   : {derived_pred: ["(derived_pred a b)", ...]}  EXACT train closure
      negatives : [] (new-entity negatives live in new_checks)
      new_base  : list of "(base_pred a b)" involving the new nodes
      new_checks: [(query, expected_bool, desc)] over derived_pred for new pairs
    All edges go forward in a shuffled order, so the full graph is acyclic.
    """
    from dreamlog.recursive_discovery import transitive_closure
    if n_train < 2 or len(node_pool) < n_train + n_new:
        raise ValueError("need n_train >= 2 and node_pool >= n_train + n_new")
    rng = random.Random(seed)
    train_nodes = node_pool[:n_train]
    new_nodes = node_pool[n_train:n_train + n_new]

    # Training DAG: a spanning chain plus a few forward extras.
    order = train_nodes[:]
    rng.shuffle(order)
    train_edges = set()
    for i in range(len(order) - 1):
        train_edges.add((order[i], order[i + 1]))
    for _ in range(n_extra):
        i = rng.randrange(0, len(order) - 1)
        j = rng.randrange(i + 1, len(order))
        train_edges.add((order[i], order[j]))
    train_closure = transitive_closure(train_edges)

    # New entities extend the forward order, so new edges stay acyclic.
    full_order = order + new_nodes
    new_edges = set()
    for i in range(len(order) - 1, len(full_order) - 1):
        new_edges.add((full_order[i], full_order[i + 1]))  # chain into new nodes
    for _ in range(n_extra):
        i = rng.randrange(0, len(order))
        j = rng.randrange(len(order), len(full_order))
        new_edges.add((full_order[i], full_order[j]))

    full_closure = transitive_closure(train_edges | new_edges)
    new_set = set(new_nodes)
    positives = sorted(p for p in full_closure
                       if p[0] in new_set or p[1] in new_set)
    negatives_pairs = sorted(
        (a, b) for a in full_order for b in full_order
        if a != b and (a in new_set or b in new_set)
        and (a, b) not in full_closure)
    neg_sample = negatives_pairs[:max(1, len(positives))]  # balance for accuracy

    base = [f"({base_pred} {a} {b})" for a, b in sorted(train_edges)]
    derived: Dict[str, List[str]] = {
        derived_pred: [f"({derived_pred} {a} {b})"
                       for a, b in sorted(train_closure)]}
    new_base = [f"({base_pred} {a} {b})" for a, b in sorted(new_edges)]
    new_checks = (
        [(f"({derived_pred} {a} {b})", True, f"{a}->{b} reachable")
         for a, b in positives]
        + [(f"({derived_pred} {a} {b})", False, f"{a}->{b} not reachable")
           for a, b in neg_sample]
    )
    return base, derived, [], new_base, new_checks


def flux_new_entity_domain(seed: int = 42):
    """Invented-vocabulary closure domain (flux_links / flux_reaches)."""
    return _closure_new_entity_domain(
        _INVENTED_NODES, "flux_links", "flux_reaches", seed=seed)


def family_new_entity_domain(seed: int = 42):
    """Canonical closure domain (parent / ancestor) over real names."""
    return _closure_new_entity_domain(
        _REAL_NAMES, "parent", "ancestor", seed=seed)


def run_experiment():
    import argparse
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from ex25b_novel_generalization import run_domain_test
    from ex25_generalization import get_llm_client

    ap = argparse.ArgumentParser(
        description="EX27: recursive (transitive-closure) rule discovery")
    ap.add_argument("--runs", type=int, default=1,
                    help="runs per LLM condition (for variance)")
    ap.add_argument("--api-key-env", default="MY_ANTHROPIC_API_KEY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true",
                    help="symbolic-only: skip ALL LLM conditions "
                         "(no API calls, no cost; reproduces the symbolic rows)")
    args = ap.parse_args()

    # Cost guard: with a key set, the raw_llm baseline fires one live API call
    # per check (~150 calls across both domains). --no-llm forces symbolic-only.
    llm_client = None if args.no_llm else get_llm_client(args)

    domains = [
        ("family (ancestor, canonical)", family_new_entity_domain),
        ("flux (flux_reaches, invented)", flux_new_entity_domain),
    ]

    print("=" * 72)
    print("  EX27: Recursive Rule Discovery (new-entity protocol)")
    print("=" * 72)
    if llm_client is None:
        print("  Symbolic-only run (--no-llm or key unset): no API calls.")
    else:
        est = sum((len(builder(seed=args.seed)[4]) + 2) * args.runs
                  for _, builder in domains)
        print(f"  LLM ENABLED: ~{est} live API calls (raw_llm baseline + "
              f"Operation G). Re-run with --no-llm for a free symbolic-only pass.")

    results = {}
    for name, builder in domains:
        base, derived, negatives, new_base, new_checks = builder(seed=args.seed)
        results[name] = run_domain_test(
            name, base, derived, negatives, new_base, new_checks,
            llm_client, n_runs=args.runs, discover_recursion=True)

    if llm_client is not None:
        try:
            print(f"\n  Total LLM cost: ${llm_client.usage.estimated_cost():.4f} "
                  f"({llm_client.usage.calls} calls)")
        except Exception:
            pass
    return results


if __name__ == "__main__":
    run_experiment()
