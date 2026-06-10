"""Reduce: delete clauses implied by the remainder. Detectors lifted from
Operations A (_eliminate_subsumed) and B (_prune_redundant_facts).
Detector order is part of the behavioral contract: A1 (rule-vs-rule
subsumption), then A2 (bodyless-rule-vs-fact), then B (derivability).
A2 must run against the KB AFTER A1's removals are committed.

Unlike the other generators (run-form), reduce exposes pure propose_* forms:
its detection is stateless given a KB snapshot, and B's probe runs on a
scratch copy so detection is safe to call before any commit (spec 5.4,
"Two generator forms")."""
from typing import List

from ...evaluator import PrologEvaluator
from ...knowledge import KnowledgeBase
from ...unification import clause_subsumes, subsumes
from ..proposal import Proposal


def propose_subsumed_rules(kb: KnowledgeBase) -> List[Proposal]:
    # A1, lifted verbatim: same index-set logic, same sorted(reverse=True)
    # emission order.
    rules_to_remove = set()
    rules = kb.rules
    for i, rule_a in enumerate(rules):
        for j, rule_b in enumerate(rules):
            if i == j or j in rules_to_remove:
                continue
            if clause_subsumes(rule_a, rule_b) and rule_a != rule_b:
                rules_to_remove.add(j)
    return [Proposal(kind="subsumption", remove=(rules[idx],))
            for idx in sorted(rules_to_remove, reverse=True)]


def propose_subsumed_facts(kb: KnowledgeBase) -> List[Proposal]:
    # A2, lifted verbatim. Call ONLY after A1's removals are committed.
    out = []
    bodyless_rules = [r for r in kb.rules if len(r.body) == 0]
    for fact in kb.facts:
        for rule in bodyless_rules:
            if subsumes(rule.head, fact.term):
                out.append(Proposal(kind="subsumption", remove=(fact,),
                                    notes={"detector": "bodyless_subsumed"}))
                break
    return out


def propose_redundant_facts(kb: KnowledgeBase, max_calls: int = 0) -> List[Proposal]:
    # B Phase 1, lifted, but probing on a SCRATCH COPY so detection never
    # mutates the real KB. Identical results: the evaluator sees the same
    # clause sets, and dream() discards op-time usage counters anyway.
    scratch = kb.copy()
    out = []
    for fact in list(scratch.facts):
        scratch.remove_fact_by_value(fact)
        ev = PrologEvaluator(scratch, max_total_calls=max_calls)
        try:
            is_derivable = ev.has_solution(fact.term)
        except RecursionError:
            is_derivable = False
        if is_derivable:
            out.append(Proposal(kind="pruning", remove=(fact,),
                                notes={"detector": "derivable"}))
        scratch.add_fact(fact)
    return out
