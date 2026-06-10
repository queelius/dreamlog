"""The single accept/verify/rollback mechanism for compression proposals.

Trial-apply on a copy, policy verification, commit to the real KB on success.
RecursionError during verification maps to Rejected("budget"), matching the
catch-and-skip semantics the ops use today. Rejection reasons:
delta | verify_failed | fp_check | budget | policy.
"""
from dataclasses import dataclass
from typing import List

from ..knowledge import KnowledgeBase, Rule
from .dl import proposal_delta
from .proposal import Proposal


@dataclass(frozen=True)
class Accepted:
    candidate: object   # CompressionCandidate; typed loosely to avoid the import cycle


@dataclass(frozen=True)
class Rejected:
    kind: str
    reason: str
    detail: str = ""


def _apply(kb: KnowledgeBase, p: Proposal) -> None:
    for c in p.remove:
        if isinstance(c, Rule):
            kb.remove_rule_by_value(c)
        else:
            kb.remove_fact_by_value(c)
    for c in p.add:
        if isinstance(c, Rule):
            kb.add_rule(c)
        else:
            kb.add_fact(c)


def _candidate(p: Proposal):
    from ..kb_dreamer import CompressionCandidate
    return CompressionCandidate(operation=p.kind,
                                original_clauses=list(p.remove),
                                new_clauses=list(p.add))


def apply_proposal(kb, p, policy):
    if getattr(policy, "require_negative_delta", False) and proposal_delta(p) >= 0:
        return Rejected(p.kind, "delta")
    reason = policy.pre_check(kb, p)
    if reason:
        return Rejected(p.kind, reason)
    trial = kb.copy()
    try:
        _apply(trial, p)
    except ValueError:
        # A clause in p.remove is absent from the KB. Cannot happen with the
        # P1/P2 generators (one proposal per distinct clause), but overlapping
        # proposals from a future generator must reject, not crash.
        return Rejected(p.kind, "policy",
                        detail="remove-clause absent (overlapping proposals?)")
    try:
        reason = policy.verify(trial, p)
    except RecursionError:
        return Rejected(p.kind, "budget")
    if reason:
        detail = "remove=[" + "; ".join(str(c) for c in p.remove) + "]"
        return Rejected(p.kind, reason, detail=detail)
    _apply(kb, p)
    return Accepted(_candidate(p))


def apply_batch_with_fallback(kb, proposals, policy):
    """Operation B's semantics: try the whole batch; if the batch-level verify
    fails, fall back to applying items one at a time SEQUENTIALLY against the
    live KB (each commit affects the next item's verification)."""
    accepted: List[Accepted] = []
    rejected: List[Rejected] = []
    if not proposals:
        return accepted, rejected
    trial = kb.copy()
    try:
        for p in proposals:
            _apply(trial, p)
    except ValueError:
        # Overlapping removals inside one batch: fall back to per-item, where
        # apply_proposal handles the absence gracefully.
        batch_reason = "policy"
    else:
        try:
            batch_reason = policy.verify_batch(trial, proposals)
        except RecursionError:
            batch_reason = "budget"
    if batch_reason is None:
        for p in proposals:
            _apply(kb, p)
            accepted.append(Accepted(_candidate(p)))
        return accepted, rejected
    for p in proposals:
        res = apply_proposal(kb, p, policy)
        (accepted if isinstance(res, Accepted) else rejected).append(res)
    return accepted, rejected
