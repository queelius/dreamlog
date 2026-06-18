"""Operation I open-world partial-closure discovery (spec 2026-06-18).

Closed-world Op I fires only when the observed extension of R EXACTLY equals
transitive_closure(B). The open-world extension (open_world=True) relaxes that
gate: when R is a sufficiently complete STRICT subset of the closure, Op I
synthesizes the base + right-recursive rule and recovers the missing pairs.

These tests pin the spec's acceptance criteria:
  AC1 -- zero-drift sentinels (closed-world unchanged)
  AC2 -- recovery of missing closure pairs in open-world
  AC3 -- coverage threshold tau=0.5 on both sides
  AC4 -- soundness: an R pair outside the closure rejects the open-world path

Construction style mirrors test_recursive_discovery.py / ex35: factories
atom/compound, KnowledgeBase, KnowledgeBaseDreamer.
"""
from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, KnowledgeBase
from dreamlog.kb_dreamer import KnowledgeBaseDreamer
from dreamlog.evaluator import PrologEvaluator
from dreamlog.recursive_discovery import transitive_closure


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _chain_edges(chain_len):
    """Directed chain n0->n1->...->n{chain_len}."""
    nodes = ["n%d" % i for i in range(chain_len + 1)]
    return [(nodes[i], nodes[i + 1]) for i in range(chain_len)]


def _build_kb(base_pairs, r_pairs, base_functor="b", r_functor="r"):
    kb = KnowledgeBase()
    for x, y in base_pairs:
        kb.add_fact(Fact(compound(base_functor, atom(x), atom(y))))
    for x, y in r_pairs:
        kb.add_fact(Fact(compound(r_functor, atom(x), atom(y))))
    return kb


def _r_facts(kb, r_functor="r"):
    return [f for f in kb.facts if f.term.functor == r_functor]


def _has_recursion_rule(kb, r_functor="r"):
    return any(r.head.functor == r_functor and len(r.body) > 0 for r in kb.rules)


def _derivable(kb, functor, x, y, max_calls=10000):
    ev = PrologEvaluator(kb, max_total_calls=max_calls)
    return ev.has_solution(compound(functor, atom(x), atom(y)))


# ---------------------------------------------------------------------------
# AC2: recovery -- strict subset, coverage above tau, recovers missing pairs
# ---------------------------------------------------------------------------

def test_ac2_open_world_recovers_missing_closure_pairs():
    base = _chain_edges(4)                      # 4 base facts: n0->..->n4
    closure = transitive_closure(set(base))     # 10 closure pairs
    closure_sorted = sorted(closure)
    # STRICT subset of the closure: drop two pairs so coverage = 8/10 = 0.8 > 0.5
    missing = {closure_sorted[0], closure_sorted[-1]}
    observed = [p for p in closure_sorted if p not in missing]
    assert len(observed) < len(closure)         # strict subset
    assert len(observed) / len(closure) >= 0.5  # above tau

    kb = _build_kb(base, observed)
    dreamer = KnowledgeBaseDreamer(discover_recursion=True, open_world=True,
                                   min_base_facts=3)
    dreamer.dream(kb)

    # Op I produced a recursion candidate (recursive r-rule present, r-facts gone)
    assert _has_recursion_rule(kb)
    assert not _r_facts(kb)

    # Recovery > 0: every missing closure pair is now derivable
    for x, y in missing:
        assert _derivable(kb, "r", x, y), "missing closure pair (%s,%s) not recovered" % (x, y)

    # Precision preserved: no r-derivation OUTSIDE the closure. Probe a pair that
    # is provably not in the closure (reverse edge of the chain).
    rev = (base[0][1], base[0][0])              # (n1, n0): not reachable
    assert rev not in closure
    assert not _derivable(kb, "r", rev[0], rev[1])


# ---------------------------------------------------------------------------
# AC3: coverage threshold tau=0.5 on both sides
# ---------------------------------------------------------------------------

def test_ac3_fires_at_or_above_tau():
    base = _chain_edges(4)
    closure = transitive_closure(set(base))     # 10 pairs
    closure_sorted = sorted(closure)
    # coverage 0.6 (>= 0.5): keep 6 of 10
    observed = closure_sorted[:6]
    assert len(observed) / len(closure) >= 0.5

    kb = _build_kb(base, observed)
    dreamer = KnowledgeBaseDreamer(discover_recursion=True, open_world=True,
                                   min_base_facts=3)
    dreamer.dream(kb)
    assert _has_recursion_rule(kb)
    assert not _r_facts(kb)


def test_ac3_does_not_fire_below_tau():
    base = _chain_edges(4)
    closure = transitive_closure(set(base))     # 10 pairs
    closure_sorted = sorted(closure)
    # coverage 0.4 (< 0.5): keep 4 of 10, all genuine closure pairs (subset holds)
    observed = closure_sorted[:4]
    assert len(observed) / len(closure) < 0.5

    kb = _build_kb(base, observed)
    n_r_before = len(observed)
    dreamer = KnowledgeBaseDreamer(discover_recursion=True, open_world=True,
                                   min_base_facts=3)
    dreamer.dream(kb)
    # Sparse subset is left alone: no recursion rule, r-facts untouched
    assert not _has_recursion_rule(kb)
    assert len(_r_facts(kb)) == n_r_before


# ---------------------------------------------------------------------------
# AC4: soundness -- an observed R pair outside the closure rejects the path
# ---------------------------------------------------------------------------

def test_ac4_rejects_when_r_pair_outside_closure():
    base = _chain_edges(4)
    closure = transitive_closure(set(base))
    closure_sorted = sorted(closure)
    # High coverage subset...
    observed = list(closure_sorted[:7])
    # ...plus one r-pair that is NOT in the closure (uses an external node).
    spurious = ("z0", "z1")
    assert spurious not in closure
    observed.append(spurious)

    kb = _build_kb(base, observed)
    n_r_before = len(observed)
    dreamer = KnowledgeBaseDreamer(discover_recursion=True, open_world=True,
                                   min_base_facts=3)
    dreamer.dream(kb)
    # R is not a (partial) closure of B: open-world path must NOT fire.
    assert not _has_recursion_rule(kb)
    assert len(_r_facts(kb)) == n_r_before


# ---------------------------------------------------------------------------
# AC1: zero-drift sentinels (closed-world behavior unchanged)
# ---------------------------------------------------------------------------

def test_ac1_closed_world_strict_subset_does_not_fire():
    """With open_world=False (default), a strict-subset R must NOT trigger Op I:
    the exact `r_ext == TC(b_ext)` gate is preserved bit-for-bit."""
    base = _chain_edges(4)
    closure = transitive_closure(set(base))
    closure_sorted = sorted(closure)
    observed = closure_sorted[:8]               # strict subset, coverage 0.8
    assert len(observed) < len(closure)

    kb = _build_kb(base, observed)
    n_r_before = len(observed)
    dreamer = KnowledgeBaseDreamer(discover_recursion=True, open_world=False,
                                   min_base_facts=3)
    dreamer.dream(kb)
    assert not _has_recursion_rule(kb)
    assert len(_r_facts(kb)) == n_r_before


def test_ac1_exact_closure_fires_in_both_worlds():
    """An EXACT closure R triggers Op I regardless of open_world."""
    base = _chain_edges(4)
    closure = transitive_closure(set(base))
    observed = sorted(closure)                  # exact closure

    for open_world in (False, True):
        kb = _build_kb(base, observed)
        dreamer = KnowledgeBaseDreamer(discover_recursion=True,
                                       open_world=open_world, min_base_facts=3)
        dreamer.dream(kb)
        assert _has_recursion_rule(kb), "exact closure failed to fire (open_world=%s)" % open_world
        assert not _r_facts(kb)
