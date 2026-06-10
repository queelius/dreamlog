"""Single source of truth for description length. P1: clause count.

P3 will replace these internals with a bits-based symbol encoding (and a
functor-signature charge); the signatures are stable so call sites never
change again.
"""
from ..knowledge import KnowledgeBase
from .proposal import Proposal, Clause


def clause_cost(clause: Clause) -> int:
    return 1


def description_length(kb: KnowledgeBase) -> int:
    return len(kb)


def proposal_delta(p: Proposal) -> int:
    return sum(clause_cost(c) for c in p.add) - sum(clause_cost(c) for c in p.remove)
