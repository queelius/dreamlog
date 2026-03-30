"""
Anti-unification (least general generalization) for DreamLog terms.

Implements Plotkin's (1970) algorithm: the dual of unification.
Where unification finds the most general unifier of two terms,
anti-unification finds the least general generalization -- the most
specific term that subsumes both inputs.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from .terms import Term, Atom, Variable, Compound


@dataclass
class AntiUnificationResult:
    """Result of anti-unifying two or more terms."""
    generalized: Term
    substitutions: List[Dict[str, Term]]
    variables_introduced: int
    shared_structure: float


def node_count(term: Term) -> int:
    """Count nodes in a term tree."""
    if isinstance(term, (Atom, Variable)):
        return 1
    if isinstance(term, Compound):
        return 1 + sum(node_count(arg) for arg in term.args)
    raise TypeError(f"Unknown term type: {type(term)}")


def anti_unify(term1: Term, term2: Term) -> AntiUnificationResult:
    """Compute the least general generalization of two terms.

    Given two terms t1 and t2, returns a term g such that:
      - g is at least as general as both t1 and t2
      - there exist substitutions s1, s2 where g.substitute(s1) == t1
        and g.substitute(s2) == t2
      - g is the most specific such term (least general generalization)

    Args:
        term1: First term
        term2: Second term

    Returns:
        AntiUnificationResult with the generalized term, substitutions,
        count of fresh variables introduced, and shared structure ratio.
    """
    seen_pairs: Dict[Tuple[Term, Term], Variable] = {}
    sub1: Dict[str, Term] = {}
    sub2: Dict[str, Term] = {}
    counter = [0]

    def _fresh_var() -> Variable:
        name = f"_G{counter[0]}"
        counter[0] += 1
        return Variable(name)

    def _anti_unify_rec(t1: Term, t2: Term) -> Term:
        if t1 == t2:
            return t1
        pair_key = (t1, t2)
        if pair_key in seen_pairs:
            return seen_pairs[pair_key]
        if (isinstance(t1, Compound) and isinstance(t2, Compound)
                and t1.functor == t2.functor and t1.arity == t2.arity):
            new_args = [_anti_unify_rec(a1, a2)
                        for a1, a2 in zip(t1.args, t2.args)]
            return Compound(t1.functor, new_args)
        v = _fresh_var()
        seen_pairs[pair_key] = v
        sub1[v.name] = t1
        sub2[v.name] = t2
        return v

    generalized = _anti_unify_rec(term1, term2)
    variables_introduced = counter[0]
    total_nodes = node_count(generalized)
    shared_nodes = total_nodes - variables_introduced
    shared_structure = shared_nodes / total_nodes if total_nodes > 0 else 1.0

    return AntiUnificationResult(
        generalized=generalized,
        substitutions=[sub1, sub2],
        variables_introduced=variables_introduced,
        shared_structure=shared_structure,
    )


def anti_unify_many(terms: List[Term]) -> AntiUnificationResult:
    """Compute the lgg of multiple terms via pairwise folding.

    Folds `anti_unify` across the list, composing substitutions at each
    step so that every original term can be recovered from the final
    generalization.

    Args:
        terms: Non-empty list of terms to anti-unify.

    Returns:
        AntiUnificationResult where substitutions[i] recovers terms[i].

    Raises:
        ValueError: If the list is empty.
    """
    if not terms:
        raise ValueError("Cannot anti-unify empty list")
    if len(terms) == 1:
        return AntiUnificationResult(
            generalized=terms[0], substitutions=[{}],
            variables_introduced=0, shared_structure=1.0)

    acc = anti_unify(terms[0], terms[1])
    all_subs = [acc.substitutions[0], acc.substitutions[1]]

    for i in range(2, len(terms)):
        acc = anti_unify(acc.generalized, terms[i])
        bridge = acc.substitutions[0]
        all_subs = [
            {v: t.substitute(old_sub) for v, t in bridge.items()}
            for old_sub in all_subs
        ] + [acc.substitutions[1]]

    return AntiUnificationResult(
        generalized=acc.generalized, substitutions=all_subs,
        variables_introduced=acc.variables_introduced,
        shared_structure=acc.shared_structure)
