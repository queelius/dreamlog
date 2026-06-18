"""Closure (Operation I): detect a binary predicate whose extension equals
the transitive closure of another and swap the extension for the base +
right-recursive pair. Run-form: at most one candidate per call; on rejection
the scan continues to the next (R, B) pair.
"""
from typing import List

from ...terms import Atom, Variable, Compound
from ...knowledge import KnowledgeBase, Rule
from ..util import _is_system_predicate
from ..proposal import Proposal
from ..gate import Accepted


def run(kb: KnowledgeBase, suite, gate_apply, policy,
        min_base_facts: int, rejections: list,
        open_world: bool = False,
        min_closure_coverage: float = 0.5,
        recovered_closures: list = None) -> List:
    """Discover transitive-closure recursive rules.

    For each binary predicate R whose ground extension equals the
    transitive closure of another binary predicate B, synthesize the
    right-recursive definition (base + recursive case), verify it against
    the suite with a bounded evaluator, and replace R's facts with the
    two rules. Returns at most one candidate per call (re-run on the next
    dream cycle to find further closures).

    Closed-world (``open_world=False``, default): only the EXACT path runs --
    accept iff ``r_ext == transitive_closure(b_ext)``. Byte-for-byte identical
    to the original gate; the subset branch below is never entered.

    Open-world (``open_world=True``): the exact path is still tried first; if
    it does not fire for an (R, B) pair, a relaxed SUBSET path may accept R as a
    PARTIAL closure of B when (spec 2026-06-18 Section 3):
      - r_ext is a strict subset of C = transitive_closure(b_ext) (soundness),
      - |r_ext| / |C| >= min_closure_coverage (confidence, tau default 0.5),
      - |b_ext| >= min_base_facts (base-size guard).
    On the subset path the synthesized clauses are identical to the exact path,
    but the Proposal carries ``notes={"predicted_closure": frozenset(C)}`` so the
    ClosurePolicy can relax the suite's negative check for the recovered pairs.

    ``suite`` lives in the policy; parameter kept for interface uniformity.
    """
    # suite lives in the policy; parameter kept for interface uniformity
    from ...recursive_discovery import transitive_closure

    # Collect binary predicate extensions over Atom-only argument pairs.
    ext: dict = {}
    facts_by_pred: dict = {}
    for fact in kb.facts:
        t = fact.term
        if (isinstance(t, Compound) and t.arity == 2
                and isinstance(t.args[0], Atom)
                and isinstance(t.args[1], Atom)):
            ext.setdefault(t.functor, set()).add(
                (t.args[0].value, t.args[1].value))
            facts_by_pred.setdefault(t.functor, []).append(fact)

    for R, r_ext in ext.items():
        if _is_system_predicate(R):
            continue
        for B, b_ext in ext.items():
            if R == B or len(b_ext) < min_base_facts:
                continue

            C = transitive_closure(b_ext)
            notes: dict = {}
            if r_ext == C:
                # Exact path (closed or open world): unchanged. notes stays empty.
                pass
            elif (open_world and r_ext != C and r_ext.issubset(C)
                  and len(r_ext) >= 1 and len(C) > 0
                  and len(r_ext) / len(C) >= min_closure_coverage):
                # Open-world subset path: R is a strict, sufficiently complete
                # subset of B's closure. Same clauses; carry C so the policy can
                # relax S- for the recovered pairs.
                notes = {"predicted_closure": frozenset(C)}
            else:
                continue

            X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
            base_rule = Rule(Compound(R, [X, Y]), [Compound(B, [X, Y])])
            rec_rule = Rule(
                Compound(R, [X, Z]),
                [Compound(B, [X, Y]), Compound(R, [Y, Z])])
            r_facts = facts_by_pred[R]
            new_clauses: List = [base_rule, rec_rule]

            # Verify on a copy with a bounded evaluator. A candidate that
            # does not terminate or over-runs the budget surfaces as a
            # verification FAILURE, not an exception: the evaluator catches
            # RecursionError internally and reports no solution, so the
            # closure facts stop being derivable and result.passed is False.
            # The try/except is a defensive backstop should that internal
            # handling ever change. Note: closure path length is bounded by
            # the evaluator's recursion-depth limit (~100), so very long
            # linear closures are rejected rather than compressed; keep
            # experiment domains well under that depth.
            proposal = Proposal(kind="recursion",
                                remove=tuple(r_facts),
                                add=tuple(new_clauses),
                                notes=notes)
            res = gate_apply(kb, proposal, policy)
            if not isinstance(res, Accepted):
                rejections.append((proposal.kind, res.reason))
                continue
            # Track accepted open-world partial closures so the final pipeline
            # verify in dream() can relax the same synthetic negatives the
            # per-op gate relaxed via ClosurePolicy.
            if recovered_closures is not None and notes.get("predicted_closure"):
                r_functor = proposal.add[0].head.functor
                recovered_closures.append((r_functor, notes["predicted_closure"]))
            return [res.candidate]

    return []
