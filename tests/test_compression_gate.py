from dreamlog.compression.gate import (Accepted, Rejected, apply_proposal,
                                       apply_batch_with_fallback)
from dreamlog.compression.proposal import Proposal
from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, KnowledgeBase, Rule


def _fact(name, *args):
    return Fact(compound(name, *[atom(a) for a in args]))


def _kb(*facts):
    kb = KnowledgeBase()
    for f in facts:
        kb.add_fact(f)
    return kb


class AcceptAll:
    operation = "pruning"
    require_negative_delta = False
    def pre_check(self, kb, p): return None
    def verify(self, trial, p): return None


class RejectVerify(AcceptAll):
    def verify(self, trial, p): return "verify_failed"


def _state(kb):
    return (sorted(str(f) for f in kb.facts), sorted(str(r) for r in kb.rules))


def test_accept_commits_and_returns_candidate():
    f1, f2 = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f1, f2)
    p = Proposal(kind="pruning", remove=(f1,))
    res = apply_proposal(kb, p, AcceptAll())
    assert isinstance(res, Accepted)
    assert res.candidate.operation == "pruning"
    assert res.candidate.original_clauses == [f1]
    assert _state(kb)[0] == [str(f2)]


def test_reject_leaves_kb_structurally_identical():
    f1, f2 = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f1, f2)
    before = _state(kb)
    res = apply_proposal(kb, Proposal(kind="pruning", remove=(f1,)), RejectVerify())
    assert isinstance(res, Rejected) and res.reason == "verify_failed"
    assert _state(kb) == before


def test_negative_delta_enforced_when_required():
    class Strict(AcceptAll):
        require_negative_delta = True
    kb = _kb(_fact("p", "a"))
    add_only = Proposal(kind="pruning",
                        add=(Rule(compound("q", atom("x")),
                                  [compound("p", atom("x"))]),))
    res = apply_proposal(kb, add_only, Strict())
    assert isinstance(res, Rejected) and res.reason == "delta"


def test_mixed_apply_matches_replace_facts_end_state():
    f1, f2, f3 = _fact("p", "a"), _fact("p", "b"), _fact("q", "c")
    rule = Rule(compound("p", atom("X")), [compound("q", atom("X"))])
    kb1 = _kb(f1, f2, f3)
    kb2 = kb1.copy()
    res = apply_proposal(kb1, Proposal(kind="generalization",
                                       remove=(f1, f2), add=(rule,)), AcceptAll())
    assert isinstance(res, Accepted)
    kb2.replace_facts([f1, f2], [rule])
    assert _state(kb1) == _state(kb2)


def test_batch_fallback_keeps_independent_items():
    # f_a removable alone, f_b not; batch fails, fallback keeps f_a removed.
    class ItemPolicy(AcceptAll):
        def __init__(self, bad): self.bad = bad
        def verify(self, trial, p):
            return "verify_failed" if p.remove and p.remove[0] == self.bad else None
        def verify_batch(self, trial, props):
            return "verify_failed"
    f_a, f_b = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f_a, f_b)
    props = [Proposal(kind="pruning", remove=(f_a,)),
             Proposal(kind="pruning", remove=(f_b,))]
    accepted, rejected = apply_batch_with_fallback(kb, props, ItemPolicy(bad=f_b))
    assert len(accepted) == 1 and len(rejected) == 1
    assert _state(kb)[0] == [str(f_b)]


def test_recursionerror_maps_to_budget():
    class Boom(AcceptAll):
        def verify(self, trial, p): raise RecursionError
    kb = _kb(_fact("p", "a"))
    res = apply_proposal(kb, Proposal(kind="pruning", remove=(_fact("p", "a"),)), Boom())
    assert isinstance(res, Rejected) and res.reason == "budget"


# -- hardening coverage (quality-review gaps) --

class AcceptAllBatch(AcceptAll):
    def verify_batch(self, trial, props): return None


def test_batch_fast_path_commits_all():
    f_a, f_b = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f_a, f_b)
    props = [Proposal(kind="pruning", remove=(f_a,)),
             Proposal(kind="pruning", remove=(f_b,))]
    accepted, rejected = apply_batch_with_fallback(kb, props, AcceptAllBatch())
    assert len(accepted) == 2 and not rejected
    assert _state(kb)[0] == []


def test_batch_empty_list_is_noop():
    kb = _kb(_fact("p", "a"))
    before = _state(kb)
    accepted, rejected = apply_batch_with_fallback(kb, [], AcceptAll())
    assert accepted == [] and rejected == [] and _state(kb) == before


def test_pure_add_proposal_commits():
    kb = _kb(_fact("q", "c"))
    rule = Rule(compound("p", atom("X")), [compound("q", atom("X"))])
    res = apply_proposal(kb, Proposal(kind="llm_compression", add=(rule,)), AcceptAll())
    assert isinstance(res, Accepted)
    assert _state(kb)[1] == [str(rule)]


def test_pre_check_rejection_path():
    class PreReject(AcceptAll):
        def pre_check(self, kb, p): return "policy"
    kb = _kb(_fact("p", "a"))
    before = _state(kb)
    res = apply_proposal(kb, Proposal(kind="pruning", remove=(_fact("p", "a"),)),
                         PreReject())
    assert isinstance(res, Rejected) and res.reason == "policy"
    assert _state(kb) == before


def test_recursionerror_in_verify_batch_falls_back():
    class BoomBatch(AcceptAll):
        def verify_batch(self, trial, props): raise RecursionError
    f_a, f_b = _fact("p", "a"), _fact("p", "b")
    kb = _kb(f_a, f_b)
    props = [Proposal(kind="pruning", remove=(f_a,)),
             Proposal(kind="pruning", remove=(f_b,))]
    accepted, rejected = apply_batch_with_fallback(kb, props, BoomBatch())
    # batch budget failure -> per-item fallback, where AcceptAll verify passes
    assert len(accepted) == 2 and not rejected


def test_overlapping_removals_reject_not_crash():
    f_a = _fact("p", "a")
    kb = _kb(f_a, _fact("p", "b"))
    dup = [Proposal(kind="pruning", remove=(f_a,)),
           Proposal(kind="pruning", remove=(f_a,))]   # same clause twice
    class BatchFails(AcceptAll):
        def verify_batch(self, trial, props): return "verify_failed"
    accepted, rejected = apply_batch_with_fallback(kb, dup, BatchFails())
    assert len(accepted) == 1
    assert len(rejected) == 1 and rejected[0].reason == "policy"
    assert "absent" in rejected[0].detail


def test_dream_session_records_rejections(monkeypatch):
    """DreamSession.rejections carries (kind, reason) pairs end to end."""
    from dreamlog.compression.policies import SuiteVerifyPolicy
    from dreamlog.kb_dreamer import KnowledgeBaseDreamer
    calls = {"n": 0}
    real_verify = SuiteVerifyPolicy.verify
    def failing_verify(self, trial_kb, p):
        calls["n"] += 1
        return "verify_failed"
    monkeypatch.setattr(SuiteVerifyPolicy, "verify", failing_verify)
    kb = KnowledgeBase()
    arts = ["a", "b", "c", "d", "e"]
    for x in arts:
        kb.add_fact(Fact(compound("artisan", atom(x))))
    for x in arts[:4]:
        kb.add_fact(Fact(compound("master", atom(x))))
    session = KnowledgeBaseDreamer().dream(kb)
    assert calls["n"] >= 1, "monkeypatched verify never invoked"
    assert any(k == "generalization" and r == "verify_failed"
               for (k, r) in session.rejections)
