"""Op I final pipeline verify: open-world closure relaxation (EX37 gap).

The per-op gate (ClosurePolicy.verify) already relaxes synthetic negatives for
recovered partial closures. But dream()'s FINAL pipeline verify reused the
original unfiltered suite, so when a recovered closure pair coincided with a
synthetic negative the whole dream rolled back.

Fix: thread accepted open-world closures into the final verify via
filter_recovered_negatives so the same pairs are dropped there too.

Test IDs:
  AC-commit  -- the EX37 rollback case: open-world Op I fires, in-closure
                negative coincides with a synthetic negative -> must commit.
  AC-safety  -- a genuinely spurious R(a,b) (outside the closure) is NOT
                dropped by the filter -> final verify still enforces it.
  AC-zero-drift -- closed-world dream with exact Op I fires and commits;
                   final-verify path is byte-identical (empty recovered_closures).
"""

import pytest
from dreamlog.factories import atom, compound
from dreamlog.knowledge import Fact, KnowledgeBase
from dreamlog.kb_dreamer import (
    KnowledgeBaseDreamer, VerificationSuite, build_verification_suite,
)
from dreamlog.compression.util import filter_recovered_negatives
from dreamlog.evaluator import PrologEvaluator
from dreamlog.recursive_discovery import transitive_closure
import dreamlog.kb_dreamer as _kb_dreamer_mod


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------

def _chain_edges(n):
    """Directed linear chain n0->n1->...->n{n}."""
    nodes = ["n%d" % i for i in range(n + 1)]
    return [(nodes[i], nodes[i + 1]) for i in range(n)]


def _build_kb(base_pairs, r_pairs, base_functor="b", r_functor="r"):
    kb = KnowledgeBase()
    for x, y in base_pairs:
        kb.add_fact(Fact(compound(base_functor, atom(x), atom(y))))
    for x, y in r_pairs:
        kb.add_fact(Fact(compound(r_functor, atom(x), atom(y))))
    return kb


def _has_recursion_rule(kb, r_functor="r"):
    return any(r.head.functor == r_functor and len(r.body) > 0 for r in kb.rules)


def _derivable(kb, functor, x, y, max_calls=10_000):
    ev = PrologEvaluator(kb, max_total_calls=max_calls)
    return ev.has_solution(compound(functor, atom(x), atom(y)))


# ---------------------------------------------------------------------------
# AC-commit: EX37 rollback reproduction
#
# Chain 4: b-facts n0->n1->n2->n3->n4  (4 edges, 10 closure pairs)
# Observed r-pairs: all closure pairs EXCEPT those ending in n4 (6 of 10).
# Coverage = 0.6 >= tau=0.5 -> open-world subset path fires.
# Missing pairs: (n0,n4),(n1,n4),(n2,n4),(n3,n4).
#
# We inject r(n0,n4) into the suite's negative_queries to simulate the EX37
# scenario deterministically: build_verification_suite generates this negative
# when n4 appears in the atom pool (via b-facts) but NOT in any r-fact's arg1.
# We use a monkeypatch of build_verification_suite so the injected negative is
# present in the suite that dream() builds, which is the live version.
#
# Without the fix: dream() passes the unfiltered suite to the final verify.
# After Op I installs the recursive rule, r(n0,n4) becomes derivable -> false_positive
# -> verification fails -> rollback -> session.compressed == False.
# With the fix: filter_recovered_negatives drops r(n0,n4) from the final
# verify's negatives (it is inside the predicted closure) -> passes -> commits.
# ---------------------------------------------------------------------------

@pytest.fixture()
def chain4_partial_r():
    """Chain-4 KB with partial r (all closure pairs except those ending in n4)."""
    base = _chain_edges(4)                        # n0->n1->n2->n3->n4
    closure = transitive_closure(set(base))
    # Keep all closure pairs whose arg1 is not n4.
    observed = [(a, b) for (a, b) in sorted(closure) if b != "n4"]
    assert len(observed) == 6                     # 6 of 10
    assert len(observed) / len(closure) >= 0.5    # above tau
    return base, observed, closure


def _inject_r_n0_n4(original_bvs):
    """Return a patched build_verification_suite that appends r(n0,n4) to negatives."""
    def patched(kb):
        suite = original_bvs(kb)
        neg = compound("r", atom("n0"), atom("n4"))
        if neg not in suite.negative_queries:
            suite.negative_queries.append(neg)
        return suite
    return patched


def test_ac_commit_open_world_does_not_rollback(chain4_partial_r, monkeypatch):
    """Pre-fix this would roll back; post-fix it commits with recovery."""
    base, observed, closure = chain4_partial_r
    kb = _build_kb(base, observed)

    # Patch build_verification_suite so the in-closure negative is present.
    monkeypatch.setattr(
        _kb_dreamer_mod, "build_verification_suite",
        _inject_r_n0_n4(_kb_dreamer_mod.build_verification_suite))

    dreamer = KnowledgeBaseDreamer(
        discover_recursion=True, open_world=True, min_base_facts=3)
    session = dreamer.dream(kb)

    # Post-fix: Op I fires (partial closure accepted) and dream commits.
    assert session.compressed, (
        "dream() rolled back (pre-fix behavior): "
        "final verify flagged in-closure negative as false_positive")

    # Op I installed the recursive rule; r-facts were replaced.
    assert _has_recursion_rule(kb), "recursive r-rule not installed"
    assert not [f for f in kb.facts if f.term.functor == "r"], \
        "r-facts still present after Op I"

    # Recovery: all missing pairs (those ending in n4) are now derivable.
    missing = [(a, b) for (a, b) in sorted(closure) if b == "n4"]
    assert missing, "no missing pairs (test misconfigured)"
    for x, y in missing:
        assert _derivable(kb, "r", x, y), \
            "missing closure pair (%s,%s) not recovered" % (x, y)

    # Precision: no r(a,b) derivable OUTSIDE the closure.
    # Spot-check a pair provably not reachable (reverse of first base edge).
    not_in_closure = ("n1", "n0")
    assert not_in_closure not in closure
    assert not _derivable(kb, "r", not_in_closure[0], not_in_closure[1]), \
        "spurious r-pair derived outside closure"


def test_ac_commit_zero_spurious_no_r_outside_closure(chain4_partial_r, monkeypatch):
    """Enumerate all (ni,nj) pairs: none outside the full closure are derivable."""
    base, observed, closure = chain4_partial_r
    kb = _build_kb(base, observed)

    monkeypatch.setattr(
        _kb_dreamer_mod, "build_verification_suite",
        _inject_r_n0_n4(_kb_dreamer_mod.build_verification_suite))

    dreamer = KnowledgeBaseDreamer(
        discover_recursion=True, open_world=True, min_base_facts=3)
    dreamer.dream(kb)

    nodes = ["n%d" % i for i in range(5)]
    for x in nodes:
        for y in nodes:
            expected = (x, y) in closure
            got = _derivable(kb, "r", x, y)
            assert got == expected, \
                "r(%s,%s): expected %s got %s" % (x, y, expected, got)


# ---------------------------------------------------------------------------
# AC-safety: a genuinely spurious R(a,b) outside the closure is NOT dropped
# ---------------------------------------------------------------------------

def test_ac_safety_spurious_negative_outside_closure_is_retained():
    """filter_recovered_negatives must NOT drop negatives outside the closure."""
    closure_pairs = frozenset({("n0", "n1"), ("n0", "n2"), ("n1", "n2")})
    recovered = [("r", closure_pairs)]

    in_closure_neg = compound("r", atom("n0"), atom("n1"))   # inside -> dropped
    out_of_closure_neg = compound("r", atom("n2"), atom("n0"))  # outside -> kept

    negatives = [in_closure_neg, out_of_closure_neg]
    result = filter_recovered_negatives(negatives, recovered)

    assert in_closure_neg not in result, \
        "in-closure negative should have been dropped"
    assert out_of_closure_neg in result, \
        "out-of-closure negative must be retained by the filter"


def test_ac_safety_different_functor_negative_retained():
    """Negatives for a functor OTHER than R are never dropped."""
    closure_pairs = frozenset({("n0", "n1")})
    recovered = [("r", closure_pairs)]

    b_neg = compound("b", atom("n0"), atom("n1"))   # same pair, different functor
    result = filter_recovered_negatives([b_neg], recovered)
    assert b_neg in result, "different-functor negative must not be dropped"


# ---------------------------------------------------------------------------
# AC-safety: final verify still enforces spurious negatives end-to-end
#
# Even after the fix, if Op I derives a pair genuinely outside the closure
# (which the spec says cannot happen for a valid transitive-closure rule, but
# we test the filter does not over-drop), the final verify catches it.
# We simulate this by constructing a KB where the synthetic negative from
# build_verification_suite is NOT in the predicted closure, so the filter
# keeps it, and dream() rolls back because the rule cannot be sound.
# ---------------------------------------------------------------------------

def test_ac_safety_final_verify_enforces_out_of_closure_negative():
    """A synthetic negative truly outside the predicted closure is kept by the
    filter; dream() rolls back if the derived rule violates it.

    Here we build a KB where an exact closure exists (r = closure(b)) but we
    also add a SPURIOUS r-fact not in the closure, making r a superset, not a
    subset. Op I cannot fire (soundness guard in closure.run rejects r not
    issubset C). So dream() leaves the KB unchanged (no recursion rule).
    The synthetic negative remains enforced, not silently dropped.
    """
    base = _chain_edges(3)                         # n0->n1->n2->n3
    closure = transitive_closure(set(base))
    # R = full closure PLUS one spurious pair outside it.
    spurious = ("z0", "z1")
    assert spurious not in closure
    r_pairs = sorted(closure) + [spurious]

    kb = _build_kb(base, r_pairs)
    n_r_before = len(r_pairs)
    dreamer = KnowledgeBaseDreamer(
        discover_recursion=True, open_world=True, min_base_facts=3)
    session = dreamer.dream(kb)

    # Op I must NOT fire (r is not a subset of closure(b)).
    assert not _has_recursion_rule(kb), "Op I should not fire for superset R"
    assert len([f for f in kb.facts if f.term.functor == "r"]) == n_r_before


# ---------------------------------------------------------------------------
# AC-zero-drift: closed-world behavior unchanged
# ---------------------------------------------------------------------------

def test_ac_zero_drift_empty_recovered_closures_returns_same_list():
    """filter_recovered_negatives with empty recovered_closures returns the
    SAME list object (no copy), guaranteeing the closed-world fast path."""
    negatives = [compound("r", atom("a"), atom("b"))]
    result = filter_recovered_negatives(negatives, [])
    assert result is negatives, \
        "closed-world: must return the identical list object (no copy)"


def test_ac_zero_drift_closed_world_exact_op_i_commits():
    """closed-world (open_world=False) exact Op I fires and commits;
    the final-verify path is byte-identical (empty recovered_closures)."""
    base = _chain_edges(4)
    closure = transitive_closure(set(base))
    r_pairs = sorted(closure)          # EXACT closure -> closed-world fires

    kb = _build_kb(base, r_pairs)
    dreamer = KnowledgeBaseDreamer(
        discover_recursion=True, open_world=False, min_base_facts=3)
    session = dreamer.dream(kb)

    assert session.compressed, "closed-world exact Op I should commit"
    assert _has_recursion_rule(kb), "recursive rule not installed"
    # recovered_closures must be empty after a closed-world dream.
    assert dreamer._recovered_closures == [], \
        "closed-world must leave _recovered_closures empty"


def test_ac_zero_drift_no_discover_recursion():
    """With discover_recursion=False (default), _recovered_closures stays empty
    and the final verify is the original unmodified suite (zero drift)."""
    base = _chain_edges(4)
    r_pairs = sorted(transitive_closure(set(base)))
    kb = _build_kb(base, r_pairs)
    dreamer = KnowledgeBaseDreamer(discover_recursion=False, open_world=True)
    dreamer.dream(kb)
    assert dreamer._recovered_closures == []


def test_ac_zero_drift_filter_noop_for_unary_or_non_atom_args():
    """filter_recovered_negatives ignores non-binary or non-Atom queries."""
    from dreamlog.terms import Variable
    recovered = [("r", frozenset({("a", "b")}))]
    # Unary query
    unary = compound("r", atom("a"))
    # Binary with Variable arg
    binary_var = compound("r", atom("a"), Variable("X"))

    result = filter_recovered_negatives([unary, binary_var], recovered)
    assert unary in result
    assert binary_var in result
