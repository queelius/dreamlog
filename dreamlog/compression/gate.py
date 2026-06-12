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


def _record(policy, kb, p, decision, reason=""):
    recorder = getattr(policy, "recorder", None)
    if recorder is None:
        return
    from .dl import proposal_delta as _pd
    rec = {
        "kind": p.kind,
        "removed": [str(c) for c in p.remove][:8],
        "added": [str(c) for c in p.add][:8],
        "delta_clauses": _pd(p),
        "delta_bits": _pd(p, kb=kb, mode="bits"),
        "decision": decision,
        "reason": reason,
    }
    recorder(rec)


def apply_proposal(kb, p, policy):
    mode = getattr(policy, "dl_mode", "clauses")
    if getattr(policy, "require_negative_delta", False) \
            and proposal_delta(p, kb=kb, mode=mode) >= 0:
        _record(policy, kb, p, "rejected", "delta")
        return Rejected(p.kind, "delta")
    reason = policy.pre_check(kb, p)
    if reason:
        _record(policy, kb, p, "rejected", reason)
        return Rejected(p.kind, reason)
    trial = kb.copy()
    try:
        _apply(trial, p)
    except ValueError:
        # A clause in p.remove is absent from the KB. Cannot happen with the
        # P1/P2 generators (one proposal per distinct clause), but overlapping
        # proposals from a future generator must reject, not crash.
        _record(policy, kb, p, "rejected", "policy")
        return Rejected(p.kind, "policy",
                        detail="remove-clause absent (overlapping proposals?)")
    try:
        reason = policy.verify(trial, p)
    except RecursionError:
        _record(policy, kb, p, "rejected", "budget")
        return Rejected(p.kind, "budget")
    if reason:
        _record(policy, kb, p, "rejected", reason)
        detail = "remove=[" + "; ".join(str(c) for c in p.remove) + "]"
        return Rejected(p.kind, reason, detail=detail)
    _record(policy, kb, p, "accepted")
    _apply(kb, p)
    return Accepted(_candidate(p))


def apply_batch_staged_combined(kb, context_proposals, item_proposals, policy):
    """Operation G Phase 4+5 semantics. Context proposals (helper rules) are
    present in every item trial and committed only if the combined check
    passes; items are verified INDEPENDENTLY against kb + context + item
    (not cumulatively); then the combined set is verified; on combined
    failure NOTHING commits (context included)."""
    staged, rejected = [], []
    for p in item_proposals:
        reason = policy.pre_check(kb, p)
        if reason:
            _record(policy, kb, p, "rejected", reason)
            rejected.append(Rejected(p.kind, reason))
            continue
        trial = kb.copy()
        for cp in context_proposals:
            _apply(trial, cp)
        _apply(trial, p)
        try:
            reason = policy.verify(trial, p)
        except RecursionError:
            reason = "budget"
        if reason:
            _record(policy, kb, p, "rejected", reason)
            rejected.append(Rejected(p.kind, reason))
            continue
        _record(policy, kb, p, "staged")
        staged.append(p)
    to_commit = list(context_proposals) + staged
    if to_commit:
        trial = kb.copy()
        for p in to_commit:
            _apply(trial, p)
        try:
            combined_reason = policy.verify_combined(trial)
        except RecursionError:
            combined_reason = "budget"
        if combined_reason:
            for p in to_commit:
                _record(policy, kb, p, "rejected", combined_reason)
            rejected.extend(Rejected(p.kind, combined_reason) for p in to_commit)
            return [], rejected
    accepted = []
    for p in to_commit:
        _apply(kb, p)
        _record(policy, kb, p, "accepted", "combined")
        accepted.append(Accepted(_candidate(p)))
    return accepted, rejected


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
            _record(policy, kb, p, "accepted", "batch")
            _apply(kb, p)
            accepted.append(Accepted(_candidate(p)))
        return accepted, rejected
    for p in proposals:
        res = apply_proposal(kb, p, policy)
        (accepted if isinstance(res, Accepted) else rejected).append(res)
    return accepted, rejected
