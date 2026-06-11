"""Cache-policy pair, outside the MDL objective.

F (evict_dead_clauses) forgets unused non-seed clauses based on wake-phase
usage; H (cache_lemmas) materializes frequently derived terms as facts for
query speed. F largely evicts what H added in earlier cycles. Neither routes
through the compression gate: F is lossy by design and H increases clause
count by design.
"""
from typing import List, Optional, Set, Union

from ..knowledge import KnowledgeBase, Fact, Rule
from ..terms import Compound
from .util import _collect_user_functors, _is_system_predicate


def evict_dead_clauses(kb: KnowledgeBase,
                       min_query_threshold: int = 10,
                       seed_terms: Optional[Set] = None,
                       seed_rules: Optional[Set] = None,
                       ) -> List:
    """Operation F: Remove clauses with 0 usage after sufficient queries.

    Requires both a minimum total query count AND that at least 50% of
    distinct predicates have been queried. This prevents pruning in KBs
    where the wake phase only exercised a narrow subset of predicates.

    User-provided seed facts/rules (present when dream() was called) are
    never pruned - only derived clauses (lemmas, LLM rules from prior
    cycles) are eligible for dead-clause removal.
    """
    from ..kb_dreamer import CompressionCandidate

    if kb.total_queries_tracked() < min_query_threshold:
        return []

    # Check predicate coverage: at least 50% must have usage
    all_functors = _collect_user_functors(kb)
    if not all_functors:
        return []

    used_functors = set()
    for fact in kb.facts:
        if isinstance(fact.term, Compound) and not _is_system_predicate(fact.term.functor):
            if kb.get_usage(fact) > 0:
                used_functors.add(fact.term.functor)
    for rule in kb.rules:
        if isinstance(rule.head, Compound) and not _is_system_predicate(rule.head.functor):
            if kb.get_usage(rule) > 0:
                used_functors.add(rule.head.functor)

    if len(used_functors) / len(all_functors) < 0.5:
        return []

    # Collect dead clauses (unused, non-system, non-seed)
    ops = []
    for fact in kb.facts:
        if isinstance(fact.term, Compound) and _is_system_predicate(fact.term.functor):
            continue
        if seed_terms and fact.term in seed_terms:
            continue
        if kb.get_usage(fact) == 0:
            kb.remove_fact_by_value(fact)
            ops.append(CompressionCandidate(
                operation="dead_clause", original_clauses=[fact]))

    for rule in kb.rules:
        if isinstance(rule.head, Compound) and _is_system_predicate(rule.head.functor):
            continue
        if seed_rules and (rule.head, tuple(rule.body)) in seed_rules:
            continue
        if kb.get_usage(rule) == 0:
            kb.remove_rule_by_value(rule)
            ops.append(CompressionCandidate(
                operation="dead_clause", original_clauses=[rule]))

    return ops


def prune_suite_for_dead(suite, dead_ops) -> None:
    """Remove dead facts' terms from the suite's positive queries (they are
    dead by definition). Moved verbatim from dream()'s inline block."""
    if not (suite and dead_ops):
        return
    dead_terms = set()
    for op in dead_ops:
        for clause in op.original_clauses:
            if isinstance(clause, Fact):
                dead_terms.add(clause.term)
    suite.positive_queries = [
        q for q in suite.positive_queries if q not in dead_terms]


def cache_lemmas(kb: KnowledgeBase,
                 min_derivation_count: int = 5
                 ) -> List:
    """Operation H: Cache frequently-derived terms as facts (lemmas).

    Adds ground terms that are derived >= min_derivation_count times
    but not already stored as facts. This speeds up future queries
    by providing direct fact lookup instead of re-derivation.
    """
    from ..kb_dreamer import CompressionCandidate

    ops = []
    frequent = kb.get_frequent_derivations(min_count=min_derivation_count)

    for term, count in frequent:
        if isinstance(term, Compound) and _is_system_predicate(term.functor):
            continue

        try:
            new_fact = Fact(term)
            kb.add_fact(new_fact)
            ops.append(CompressionCandidate(
                operation="lemma_cache",
                original_clauses=[],
                new_clauses=[new_fact]))
        except (ValueError, TypeError):
            continue  # Not a valid fact (e.g., contains variables)

    return ops
