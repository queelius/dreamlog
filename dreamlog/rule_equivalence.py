"""Structural equivalence of Horn clauses: identical up to variable renaming
and body-literal reordering, but connectivity-sensitive (a consistent variable
bijection must exist), so left- and right-recursion are distinguished."""
from itertools import permutations
from typing import Optional, Dict
from .terms import Term, Variable, Atom, Compound
from .knowledge import Rule


def _match(a: Term, b: Term, vmap: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Extend vmap (a-var name -> b-var name) so a matches b; None on failure.
    The bijection is enforced (no two a-vars map to the same b-var)."""
    if isinstance(a, Variable) and isinstance(b, Variable):
        if a.name in vmap:
            return vmap if vmap[a.name] == b.name else None
        if b.name in vmap.values():
            return None
        m = dict(vmap)
        m[a.name] = b.name
        return m
    if isinstance(a, Atom) and isinstance(b, Atom):
        return vmap if a.value == b.value else None
    if isinstance(a, Compound) and isinstance(b, Compound):
        if a.functor != b.functor or a.arity != b.arity:
            return None
        for x, y in zip(a.args, b.args):
            vmap = _match(x, y, vmap)
            if vmap is None:
                return None
        return vmap
    return None


def rules_structurally_equivalent(r1: Rule, r2: Rule) -> bool:
    """True iff r1 and r2 are the same clause up to variable renaming and body
    reordering (body as a multiset, single consistent variable bijection)."""
    if len(r1.body) != len(r2.body):
        return False
    base = _match(r1.head, r2.head, {})
    if base is None:
        return False
    n = len(r1.body)
    for perm in permutations(range(n)):
        vmap = dict(base)
        ok = True
        for i in range(n):
            vmap = _match(r1.body[i], r2.body[perm[i]], vmap)
            if vmap is None:
                ok = False
                break
        if ok:
            return True
    return False
