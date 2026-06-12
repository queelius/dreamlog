"""AC2: hand-computed examples for the bits-based description length.

Spec 2026-06-10-p3-bits-dl-design.md Section 3:
  L(D)        = sum over dictionary entries of (8*len(name) + 8)
  L(clause|D) = 4 (clause tag + terminator)
              + per occurrence: 2 (type tag) + payload
    functor payload  = log2(max(1, |F|))   F keyed by (name, arity)
    constant payload = log2(max(1, |C|))
    variable payload = log2(max(1, |V_clause|))
  DL(kb)      = L(D) + sum of clause costs
  delta       = DL(after) - DL(before), exact (tables re-priced globally)
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


def test_single_fact_kb_is_40_bits():
    # F={(p,1)} C={a}: L(D) = (8*1+8) + (8*1+8) = 32
    # fact (p a):     4 + (2+log2 1) + (2+log2 1) = 8
    kb = _kb(_fact("p", "a"))
    assert dl.description_length(kb, mode="bits") == pytest.approx(40.0)
    assert dl.clause_cost(_fact("p", "a"), kb=kb, mode="bits") == pytest.approx(8.0)


def test_three_constant_fact_cost():
    # F={(parent,2)} C={john,mary,sue} -> each fact:
    #   4 + (2+log2 1) + 2*(2 + log2 3)
    kb = _kb(_fact("parent", "john", "mary"), _fact("parent", "mary", "sue"))
    want = 4 + 2 + 2 * (2 + math.log2(3))
    assert dl.clause_cost(_fact("parent", "john", "mary"),
                          kb=kb, mode="bits") == pytest.approx(want)
    # L(D) = (8*6+8) + (8*4+8) + (8*4+8) + (8*3+8) = 56 + 40 + 40 + 32 = 168
    assert dl.description_length(kb, mode="bits") == pytest.approx(168 + 2 * want)


def test_rule_with_variable_kb_is_71_bits():
    # kb: fact (p a) + rule (q X) :- (p X)
    # F={(p,1),(q,1)} |F|=2; C={a}; rule vars {X} |V|=1
    # L(D) = (16 + 16) + 16 = 48
    # fact: 4 + (2+1) + (2+0) = 9
    # rule: 4 + (2+1) + (2+0) + (2+1) + (2+0) = 14
    X = var("X")
    rule = Rule(compound("q", X), [compound("p", X)])
    kb = _kb(_fact("p", "a"), rule)
    assert dl.clause_cost(rule, kb=kb, mode="bits") == pytest.approx(14.0)
    assert dl.description_length(kb, mode="bits") == pytest.approx(71.0)


def test_new_functor_proposal_pays_dictionary_and_repricing():
    # kb: (p a),(p b),(p c). F={(p,1)} C={a,b,c}
    # Proposal adds rule (q X) :- (p X): new functor q.
    # delta = dict growth 16
    #       + repricing 3 facts' functor occurrence (log2 2 - log2 1 = 1 each) = 3
    #       + rule cost under AFTER tables: 4 + (2+1) + (2+0) + (2+1) + (2+0) = 14
    # total = +33.0 exactly
    X = var("X")
    kb = _kb(_fact("p", "a"), _fact("p", "b"), _fact("p", "c"))
    p = Proposal(kind="llm_compression",
                 add=(Rule(compound("q", X), [compound("p", X)]),))
    assert dl.proposal_delta(p, kb=kb, mode="bits") == pytest.approx(33.0)


def test_last_occurrence_removal_credits_dictionary():
    # kb: (p a),(q a). Remove (q a): q leaves F (credit 16), |F| 2->1
    # re-prices (p a)'s functor occurrence by -1; the removed clause itself
    # cost 4+(2+1)+(2+0) = 9 under BEFORE tables but delta is exact:
    # DL_before = (16+16+16) + 9 + 9 = 66 ; DL_after = (16+16) + 8 = 40
    # delta = -26.0
    kb = _kb(_fact("p", "a"), _fact("q", "a"))
    p = Proposal(kind="pruning", remove=(_fact("q", "a"),))
    assert dl.proposal_delta(p, kb=kb, mode="bits") == pytest.approx(-26.0)


def test_extraction_sign_flips_with_sharing():
    # The P3 motivation: extraction is +1 clause ALWAYS, but in bits it must
    # earn its keep. Hand-derived margins (goal occurrence ~6-7 bits, e1
    # dictionary charge 24 bits): FOUR rules sharing a FOUR-goal body pay
    # (delta ~ -19 bits); a single rule cannot (delta ~ +42, pure overhead).
    # NOTE deliberately chosen sizes: two rules sharing a three-goal body do
    # NOT pay under this code (delta ~ +25); that near-miss is the point of
    # pricing extraction honestly instead of exempting it.
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
