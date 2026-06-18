"""AC2: hand-computed examples for the bits-based description length.

Spec 2026-06-10-p3-bits-dl-design.md Section 3 (P3b rename-invariant code):
  decl(symbol) = elias_gamma_len(arity + 1)   [arity 0 -> 1b, 1,2 -> 3b, 3,4 -> 5b]
  L(D)        = sum over dictionary entries of decl(symbol)   [F keyed by (name, arity)]
  L(clause|D) = 4 (clause tag + terminator)
              + per occurrence: 2 (type tag) + payload
    functor payload  = log2(max(1, |F|))   F keyed by (name, arity)
    constant payload = log2(max(1, |C|))
    variable payload = log2(max(1, |V_clause|))
  DL(kb)      = L(D) + sum of clause costs
  delta       = DL(after) - DL(before), exact (tables re-priced globally)

The dictionary cost depends ONLY on each symbol's arity, never its spelling:
DL is invariant under any consistent renaming (test_rename_invariance).
"""
import math

import pytest

from dreamlog.compression import dl
from dreamlog.compression.proposal import Proposal
from dreamlog.factories import atom, compound, var
from dreamlog.knowledge import Fact, KnowledgeBase, Rule


def _fact(name, *args):
    return Fact(compound(name, *[atom(a) for a in args]))


def _kb(*clauses):
    kb = KnowledgeBase()
    for c in clauses:
        if isinstance(c, Rule):
            kb.add_rule(c)
        else:
            kb.add_fact(c)
    return kb


def test_clauses_mode_unchanged():
    kb = _kb(_fact("p", "a"))
    assert dl.description_length(kb) == 1
    assert dl.clause_cost(_fact("p", "a")) == 1
    p = Proposal(kind="pruning", remove=(_fact("p", "a"),))
    assert dl.proposal_delta(p) == -1


def test_bits_requires_kb():
    with pytest.raises(ValueError):
        dl.description_length(None, mode="bits")
    with pytest.raises(ValueError):
        dl.proposal_delta(Proposal(kind="x"), kb=None, mode="bits")


def test_single_fact_kb_is_12_bits():
    # F={(p,1)} C={a}: L(D) = decl(arity 1) + decl(arity 0) = 3 + 1 = 4
    # fact (p a):     4 + (2+log2 1) + (2+log2 1) = 8
    kb = _kb(_fact("p", "a"))
    assert dl.description_length(kb, mode="bits") == pytest.approx(12.0)
    assert dl.clause_cost(_fact("p", "a"), kb=kb, mode="bits") == pytest.approx(8.0)


def test_three_constant_fact_cost():
    # F={(parent,2)} C={john,mary,sue} -> each fact:
    #   4 + (2+log2 1) + 2*(2 + log2 3)
    kb = _kb(_fact("parent", "john", "mary"), _fact("parent", "mary", "sue"))
    want = 4 + 2 + 2 * (2 + math.log2(3))
    assert dl.clause_cost(_fact("parent", "john", "mary"),
                          kb=kb, mode="bits") == pytest.approx(want)
    # L(D) = decl(parent,arity 2)=3 + 3*decl(constant,arity 0)=3 = 6
    assert dl.description_length(kb, mode="bits") == pytest.approx(6 + 2 * want)


def test_rule_with_variable_kb_is_30_bits():
    # kb: fact (p a) + rule (q X) :- (p X)
    # F={(p,1),(q,1)} |F|=2; C={a}; rule vars {X} |V|=1
    # L(D) = decl(p,1)=3 + decl(q,1)=3 + decl(a,arity 0)=1 = 7
    # fact: 4 + (2+1) + (2+0) = 9
    # rule: 4 + (2+1) + (2+0) + (2+1) + (2+0) = 14
    X = var("X")
    rule = Rule(compound("q", X), [compound("p", X)])
    kb = _kb(_fact("p", "a"), rule)
    assert dl.clause_cost(rule, kb=kb, mode="bits") == pytest.approx(14.0)
    assert dl.description_length(kb, mode="bits") == pytest.approx(30.0)


def test_new_functor_proposal_pays_dictionary_and_repricing():
    # kb: (p a),(p b),(p c). F={(p,1)} C={a,b,c}
    # Proposal adds rule (q X) :- (p X): new functor q (arity 1).
    # delta = dict growth decl(q,1) = 3
    #       + repricing 3 facts' functor occurrence (log2 2 - log2 1 = 1 each) = 3
    #       + rule cost under AFTER tables: 4 + (2+1) + (2+0) + (2+1) + (2+0) = 14
    # total = +20.0 exactly
    X = var("X")
    kb = _kb(_fact("p", "a"), _fact("p", "b"), _fact("p", "c"))
    p = Proposal(kind="llm_compression",
                 add=(Rule(compound("q", X), [compound("p", X)]),))
    assert dl.proposal_delta(p, kb=kb, mode="bits") == pytest.approx(20.0)


def test_last_occurrence_removal_credits_dictionary():
    # kb: (p a),(q a). Remove (q a): q leaves F (credit decl(q,1)=3), |F| 2->1
    # re-prices (p a)'s functor occurrence by -1; the delta is exact:
    # DL_before = (3+3+1) + 9 + 9 = 25 ; DL_after = (3+1) + 8 = 12
    # delta = -13.0
    kb = _kb(_fact("p", "a"), _fact("q", "a"))
    p = Proposal(kind="pruning", remove=(_fact("q", "a"),))
    assert dl.proposal_delta(p, kb=kb, mode="bits") == pytest.approx(-13.0)


def test_extraction_sign_flips_with_sharing():
    # The P3 motivation: extraction is +1 clause ALWAYS, but in bits it must
    # earn its keep on OCCURRENCES (the rename-invariant dictionary charge for
    # the extracted functor is only decl(arity 1)=3 bits). FOUR rules sharing a
    # FOUR-goal body removes ~14 symbol occurrences and pays (delta < 0); a
    # single rule only ADDS occurrences (the definition plus one call site) and
    # cannot pay (delta > 0). The sign, not a pinned margin, is the contract.
    X = var("X")
    body = [compound("s", X), compound("t", X), compound("u", X),
            compound("v", X)]
    heads = [compound(h, X) for h in ("h1", "h2", "h3", "h4")]
    rules = [Rule(h, list(body)) for h in heads]
    e_def = Rule(compound("e1", X), list(body))
    rewritten = [Rule(h, [compound("e1", X)]) for h in heads]

    kb4 = _kb(*rules)
    shared = Proposal(kind="extraction", remove=tuple(rules),
                      add=(e_def,) + tuple(rewritten))
    d_clauses = dl.proposal_delta(shared)
    d_bits = dl.proposal_delta(shared, kb=kb4, mode="bits")
    assert d_clauses == 1
    assert d_bits < 0, f"4x4 shared extraction should pay in bits, got {d_bits}"

    kb1 = _kb(rules[0])
    solo = Proposal(kind="extraction", remove=(rules[0],),
                    add=(e_def, rewritten[0]))
    assert dl.proposal_delta(solo, kb=kb1, mode="bits") > 0, \
        "single-occurrence extraction is pure overhead in bits"


def test_rename_invariance():
    # P3b principle: description length counts how many UNIQUE SYMBOLS and
    # clauses exist, not how their labels are spelled. Two KBs identical up to
    # a consistent renaming (preserving arity and structure) must have the
    # SAME bits-DL. Long names (the old 8*len charge) would break this.
    X, Y, Z = var("X"), var("Y"), var("Z")
    kb_verbose = _kb(
        _fact("parent", "alexander", "magdalena"),
        Rule(compound("grandparent", X, Z),
             [compound("parent", X, Y), compound("parent", Y, Z)]))
    kb_terse = _kb(
        _fact("p", "a", "b"),
        Rule(compound("g", X, Z),
             [compound("p", X, Y), compound("p", Y, Z)]))
    assert dl.description_length(kb_verbose, mode="bits") == \
        pytest.approx(dl.description_length(kb_terse, mode="bits"))


def test_symbol_decl_bits_arity_scale():
    # The arity code: gamma(arity+1). Constants (arity 0) are cheapest.
    assert dl._symbol_decl_bits(0) == 1     # gamma(1)
    assert dl._symbol_decl_bits(1) == 3     # gamma(2)
    assert dl._symbol_decl_bits(2) == 3     # gamma(3)
    assert dl._symbol_decl_bits(3) == 5     # gamma(4)
    assert dl._symbol_decl_bits(4) == 5     # gamma(5)


def test_gate_uses_policy_dl_mode():
    from dreamlog.compression.gate import Accepted, Rejected, apply_proposal

    class BitsStrict:
        operation = "extraction"
        require_negative_delta = True
        dl_mode = "bits"
        recorder = None
        def pre_check(self, kb, p): return None
        def verify(self, trial, p): return None

    X = var("X")
    body = [compound("s", X), compound("t", X), compound("u", X)]
    r1 = Rule(compound("h1", X), list(body))
    e_def = Rule(compound("e1", X), list(body))
    r1x = Rule(compound("h1", X), [compound("e1", X)])
    kb = _kb(_fact("s", "a"), _fact("t", "a"), _fact("u", "a"), r1)
    solo = Proposal(kind="extraction", remove=(r1,), add=(e_def, r1x))
    res = apply_proposal(kb, solo, BitsStrict())
    assert isinstance(res, Rejected) and res.reason == "delta"

    class ClausesLenient(BitsStrict):
        dl_mode = "clauses"
        require_negative_delta = False
    kb2 = _kb(_fact("s", "a"), _fact("t", "a"), _fact("u", "a"), r1)
    res2 = apply_proposal(kb2, solo, ClausesLenient())
    assert isinstance(res2, Accepted)


def test_recorder_receives_both_deltas():
    from dreamlog.compression.gate import apply_proposal

    records = []

    class Recording:
        operation = "pruning"
        require_negative_delta = False
        dl_mode = "clauses"
        recorder = staticmethod(records.append)
        def pre_check(self, kb, p): return None
        def verify(self, trial, p): return None

    kb = _kb(_fact("p", "a"), _fact("p", "b"))
    p = Proposal(kind="pruning", remove=(_fact("p", "a"),))
    apply_proposal(kb, p, Recording())
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "pruning"
    assert rec["decision"] == "accepted"
    assert rec["delta_clauses"] == -1
    assert rec["delta_bits"] < 0          # bits computed because recorder set


def test_no_recorder_no_bits_cost_in_clauses_mode():
    # With recorder None and clauses mode, the gate must not require bits
    # context at all (guard: passing a policy without dl_mode attr works).
    from dreamlog.compression.gate import Accepted, apply_proposal

    class Bare:
        operation = "pruning"
        require_negative_delta = True
        def pre_check(self, kb, p): return None
        def verify(self, trial, p): return None

    kb = _kb(_fact("p", "a"), _fact("p", "b"))
    res = apply_proposal(kb, Proposal(kind="pruning",
                                      remove=(_fact("p", "a"),)), Bare())
    assert isinstance(res, Accepted)


def test_extraction_policy_exemption_is_mode_conditional():
    from dreamlog.compression.policies import ExtractionPolicy
    pol = ExtractionPolicy(None, "extraction")
    assert pol.require_negative_delta is False          # clauses default
    pol.dl_mode = "bits"
    assert pol.require_negative_delta is True


def test_dreamer_threads_mode_and_recorder_to_policies():
    from dreamlog.kb_dreamer import KnowledgeBaseDreamer
    records = []
    d = KnowledgeBaseDreamer(dl_mode="bits", decision_recorder=records.append)
    kb = _kb(_fact("artisan", "a"), _fact("artisan", "b"),
             _fact("artisan", "c"), _fact("artisan", "d"),
             _fact("artisan", "e"),
             _fact("master", "a"), _fact("master", "b"),
             _fact("master", "c"), _fact("master", "d"))
    d.dream(kb)
    assert records, "recorder never invoked during a dream"
    assert all("delta_bits" in r for r in records)


def test_default_dreamer_has_no_recorder_overhead():
    from dreamlog.kb_dreamer import KnowledgeBaseDreamer
    d = KnowledgeBaseDreamer()
    assert d.dl_mode == "clauses" and d.decision_recorder is None


# ---------------------------------------------------------------------------
# Task 4: correction_cost helper + G's priced acceptance criterion (bits mode)
# ---------------------------------------------------------------------------

def test_correction_cost_hand_arithmetic():
    # kb: (p a),(p b). Before: F={(p,1)} C={a,b}.
    # correction_cost("p", 1, kb):
    #   name_charge = decl(exception functor, arity 1) = gamma(2) = 3
    #   AFTER tables add (exception_p,1) and (not,1) to F:
    #     |F| = 3 (p, exception_p, not); |C| = 2 (a, b)
    #   body_literal = (2 + log2 3)*2 + (2 + log2 1) = (2 + log2 3)*2 + 2
    #   per_fact     = 4 + (2 + log2 3) + (2 + log2 2) = 9 + log2 3
    #   total = 3 + body_literal + 1 * per_fact
    kb = _kb(_fact("p", "a"), _fact("p", "b"))
    name_charge = 3                                  # decl(arity 1) = gamma(2)
    body_literal = (2 + math.log2(3)) * 2 + (2 + math.log2(1))
    per_fact = 4 + (2 + math.log2(3)) + (2 + math.log2(2))
    want = name_charge + body_literal + 1 * per_fact
    # numeric pin: 3 + 9.169925 + 10.5849625 = 22.7548875
    assert want == pytest.approx(22.7548875)
    assert dl.correction_cost("p", 1, kb) == pytest.approx(want)
    # zero corrections -> free
    assert dl.correction_cost("p", 0, kb) == 0.0
    assert dl.correction_cost("p", -3, kb) == 0.0


def _g_dream(kb, response_rules, dl_mode, open_world):
    import json
    from dreamlog.kb_dreamer import KnowledgeBaseDreamer
    from tests.mock_provider import MockLLMProvider
    mock = MockLLMProvider(responses=[json.dumps(response_rules)] * 3)
    d = KnowledgeBaseDreamer(llm_client=mock, dl_mode=dl_mode,
                             open_world=open_world)
    return d.dream(kb), kb


def test_g_bits_accepts_rule_that_pays():
    # father facts fully explained by parent+male: savings large, no
    # over-derivations -> accepted in bits mode (closed world).
    resp = [["rule", ["father", "X", "Y"],
             [["parent", "X", "Y"], ["male", "X"]]]]
    kb = _kb(_fact("parent", "tom", "ann"), _fact("parent", "tom", "ben"),
             _fact("parent", "jim", "cat"), _fact("parent", "jim", "dan"),
             _fact("male", "tom"), _fact("male", "jim"),
             _fact("father", "tom", "ann"), _fact("father", "tom", "ben"),
             _fact("father", "jim", "cat"), _fact("father", "jim", "dan"))
    session, kb_out = _g_dream(kb, resp, dl_mode="bits", open_world=False)
    assert any(r.head.functor == "father" and len(r.body) == 2
               for r in kb_out.rules)


def test_g_bits_rejects_rule_that_cannot_pay():
    # Only ONE father fact derivable: savings is a single fact's bits,
    # far below the rule's cost -> rejected with reason "delta".
    resp = [["rule", ["father", "X", "Y"],
             [["parent", "X", "Y"], ["male", "X"]]]]
    kb = _kb(_fact("parent", "tom", "ann"), _fact("male", "tom"),
             _fact("father", "tom", "ann"))
    session, kb_out = _g_dream(kb, resp, dl_mode="bits", open_world=False)
    assert not any(r.head.functor == "father" for r in kb_out.rules)
    assert any(k == "llm_compression" and reason == "delta"
               for (k, reason) in session.rejections)


def test_g_clauses_mode_unchanged_by_task4():
    # Same one-fact KB under clauses mode: today's >= 2 rule also rejects,
    # but with reason "policy" (the proxy), proving the modes diverge only
    # where intended.
    resp = [["rule", ["father", "X", "Y"],
             [["parent", "X", "Y"], ["male", "X"]]]]
    kb = _kb(_fact("parent", "tom", "ann"), _fact("male", "tom"),
             _fact("father", "tom", "ann"))
    session, kb_out = _g_dream(kb, resp, dl_mode="clauses", open_world=False)
    assert not any(r.head.functor == "father" for r in kb_out.rules)
    assert any(k == "llm_compression" and reason == "policy"
               for (k, reason) in session.rejections)


def test_g_open_world_bits_prices_over_derivations():
    # Open world: the rule derives father(tom,ann) [in KB] AND over-derives
    # father(tom,ben) [absent]. The FP reject is disabled in open world;
    # instead the single over-derivation is priced as a correction. The
    # rule is accepted iff rule_delta + corrections < savings. With only one
    # derivable father fact, savings is tiny and the hand inequality fails,
    # so the rule is rejected with reason "delta".
    resp = [["rule", ["father", "X", "Y"],
             [["parent", "X", "Y"], ["male", "X"]]]]
    kb = _kb(_fact("parent", "tom", "ann"), _fact("parent", "tom", "ben"),
             _fact("male", "tom"),
             _fact("father", "tom", "ann"))

    # Hand inequality, computed against the construction-time snapshot kb.
    from dreamlog.compression.proposal import Proposal as _P
    rule = Rule(compound("father", var("X"), var("Y")),
                [compound("parent", var("X"), var("Y")),
                 compound("male", var("X"))])
    derivable = [_fact("father", "tom", "ann")]
    savings = sum(dl.clause_cost(f, kb=kb, mode="bits") for f in derivable)
    rule_delta = dl.proposal_delta(
        _P(kind="llm_compression", add=(rule,)), kb=kb, mode="bits")
    corrections = dl.correction_cost("father", 1, kb)   # one over-derivation
    assert rule_delta + corrections >= savings          # cannot pay

    session, kb_out = _g_dream(kb, resp, dl_mode="bits", open_world=True)
    assert not any(r.head.functor == "father" for r in kb_out.rules)
    assert any(k == "llm_compression" and reason == "delta"
               for (k, reason) in session.rejections)
