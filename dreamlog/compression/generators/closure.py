"""Closure (Operation I): detect a binary predicate whose extension equals
the transitive closure of another and swap the extension for the base +
right-recursive pair. Run-form: at most one candidate per call; on rejection
the scan continues to the next (R, B) pair.
"""
from typing import List, Optional

from ...terms import Atom, Variable, Compound
from ...knowledge import KnowledgeBase, Fact, Rule
from ..util import _is_system_predicate
from ..proposal import Proposal
from ..gate import Accepted


def run(kb: KnowledgeBase, suite, gate_apply, policy,
        min_base_facts: int, rejections: list) -> List:
    """Discover transitive-closure recursive rules.

    For each binary predicate R whose ground extension equals the
    transitive closure of another binary predicate B, synthesize the
    right-recursive definition (base + recursive case), verify it against
    the suite with a bounded evaluator, and replace R's facts with the
    two rules. Returns at most one candidate per call (re-run on the next
    dream cycle to find further closures).
    """
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
            if r_ext != transitive_closure(b_ext):
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
                                add=tuple(new_clauses))
            res = gate_apply(kb, proposal, policy)
            if not isinstance(res, Accepted):
                rejections.append((proposal.kind, res.reason))
                continue
            return [res.candidate]

    return []
