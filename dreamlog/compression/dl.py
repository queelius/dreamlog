"""Single source of truth for description length.

Two modes behind stable signatures:
- "clauses" (default): clause count, exactly the P1 behavior. Ignores kb.
- "bits": the P3 prefix code over symbol tables (spec
  2026-06-10-p3-bits-dl-design.md Section 3). Requires kb context because
  payload costs are log2 of the KB's symbol-table sizes and dictionary
  membership decides the one-time signature charge.

RENAME INVARIANCE (P3b, spec recalibration): the dictionary cost charges per
unique symbol by ARITY only, never by name spelling. The decision-diff under
the original 8*len(name)+8 charge showed it was rename-variant: a verbose
auto-generated name like exception_likes_chocolate_person cost 272 bits and
swamped the structural savings, so abstraction never paid. What an MDL code
should count is how many distinct symbols and clauses exist, not how their
labels are spelled. The fix: a symbol of arity a costs the Elias gamma code of
(a+1) bits to declare (the minimal info a two-part code needs: its arity; its
index is implicit by first-occurrence order). This is parameter-free and
invariant under any consistent renaming.

Formulas (pinned; do not simplify):
  decl(symbol) = elias_gamma_len(arity + 1)   [arity 0 -> 1b, 1,2 -> 3b, 3,4 -> 5b]
  L(D)        = sum over entries of decl(symbol)   [F keyed by (name, arity)]
  L(clause|D) = 4 + sum over occurrences of (2 + payload)
    functor: log2(max(1,|F|)); constant: log2(max(1,|C|));
    variable: log2(max(1,|V_clause|))
  DL(kb)      = L(D) + sum clause costs
  delta(p)    = DL(kb after p) - DL(kb before p), computed exactly: changing
  a table size re-prices every occurrence in the KB (log2(n+1) vs log2(n)),
  and adding/removing a symbol's last occurrence moves its declaration in/out
  of the dictionary.
"""
import math
from collections import Counter
from typing import Iterable, List, Optional, Tuple, Union

from ..knowledge import Fact, KnowledgeBase, Rule
from ..terms import Atom, Compound, Variable
from .proposal import Proposal, Clause


# -- clauses mode (P1, unchanged) --

def _clause_count_cost(clause: Clause) -> int:
    return 1


# -- bits mode --

def _elias_gamma_len(n: int) -> int:
    """Bit length of the Elias gamma code for an integer n >= 1."""
    return 2 * int(math.floor(math.log2(n))) + 1


def _symbol_decl_bits(arity: int) -> int:
    """Rename-invariant declaration cost of one dictionary symbol: the Elias
    gamma length of (arity + 1). Depends ONLY on arity, never on the symbol's
    spelling, so the description length is invariant under renaming (P3b).
    arity 0 (constants) -> 1 bit; arity 1, 2 -> 3 bits; arity 3, 4 -> 5 bits."""
    return _elias_gamma_len(arity + 1)


def _walk(term) -> Iterable[Tuple[str, object]]:
    """Yield ('f', (name, arity)) / ('c', value) / ('v', name) occurrences."""
    if isinstance(term, Compound):
        yield ("f", (term.functor, term.arity))
        for a in term.args:
            yield from _walk(a)
    elif isinstance(term, Atom):
        yield ("c", term.value)
    elif isinstance(term, Variable):
        yield ("v", term.name)


def _clause_terms(clause: Clause):
    if isinstance(clause, Rule):
        yield clause.head
        for g in clause.body:
            yield g
    else:
        yield clause.term


class _SymbolTables:
    """Occurrence-counted functor and constant tables for a clause set."""

    def __init__(self):
        self.functors: Counter = Counter()
        self.constants: Counter = Counter()

    @classmethod
    def from_clauses(cls, clauses: Iterable[Clause]) -> "_SymbolTables":
        t = cls()
        for clause in clauses:
            t.add_clause(clause)
        return t

    def add_clause(self, clause: Clause) -> None:
        for term in _clause_terms(clause):
            for kind, sym in _walk(term):
                if kind == "f":
                    self.functors[sym] += 1
                elif kind == "c":
                    self.constants[sym] += 1

    def remove_clause(self, clause: Clause) -> None:
        for term in _clause_terms(clause):
            for kind, sym in _walk(term):
                if kind == "f":
                    self.functors[sym] -= 1
                    if self.functors[sym] <= 0:
                        del self.functors[sym]
                elif kind == "c":
                    self.constants[sym] -= 1
                    if self.constants[sym] <= 0:
                        del self.constants[sym]

    def dictionary_cost(self) -> float:
        cost = 0.0
        for (_name, arity) in self.functors:
            cost += _symbol_decl_bits(arity)
        for _value in self.constants:
            cost += _symbol_decl_bits(0)   # constants are arity-0 leaves
        return cost

    def clause_cost(self, clause: Clause) -> float:
        n_f = max(1, len(self.functors))
        n_c = max(1, len(self.constants))
        n_v = max(1, len(clause.get_variables()))
        cost = 4.0
        for term in _clause_terms(clause):
            for kind, _sym in _walk(term):
                if kind == "f":
                    cost += 2 + math.log2(n_f)
                elif kind == "c":
                    cost += 2 + math.log2(n_c)
                else:
                    cost += 2 + math.log2(n_v)
        return cost


def _kb_clauses(kb: KnowledgeBase) -> List[Clause]:
    return list(kb.facts) + list(kb.rules)


def _dl_bits(clauses: List[Clause]) -> float:
    tables = _SymbolTables.from_clauses(clauses)
    return tables.dictionary_cost() + sum(
        tables.clause_cost(c) for c in clauses)


# -- public API (stable signatures) --

def clause_cost(clause: Clause, kb: Optional[KnowledgeBase] = None,
                mode: str = "clauses") -> Union[int, float]:
    if mode == "clauses":
        return _clause_count_cost(clause)
    if kb is None:
        raise ValueError("bits mode requires kb context")
    return _SymbolTables.from_clauses(_kb_clauses(kb)).clause_cost(clause)


def description_length(kb: KnowledgeBase,
                       mode: str = "clauses") -> Union[int, float]:
    if mode == "clauses":
        return len(kb)
    if kb is None:
        raise ValueError("bits mode requires kb context")
    return _dl_bits(_kb_clauses(kb))


def proposal_delta(p: Proposal, kb: Optional[KnowledgeBase] = None,
                   mode: str = "clauses") -> Union[int, float]:
    if mode == "clauses":
        return (sum(_clause_count_cost(c) for c in p.add)
                - sum(_clause_count_cost(c) for c in p.remove))
    if kb is None:
        raise ValueError("bits mode requires kb context")
    before = _kb_clauses(kb)
    after = [c for c in before]
    for r in p.remove:
        after.remove(r)          # first equal occurrence; ValueError if absent
    after.extend(p.add)
    return _dl_bits(after) - _dl_bits(before)


def correction_cost(head_functor: str, n_corrections: int,
                    kb: KnowledgeBase) -> float:
    """Bits to exclude n over-derivations from a rule via a fresh exception
    predicate: the rename-invariant declaration charge for the exception
    functor (arity 1), one not/1 + exception goal added to the rule body, and
    one exception fact per over-derivation (spec Section 4). Priced with the
    AFTER tables (exception functor and `not` added to F)."""
    if n_corrections <= 0:
        return 0.0
    exc = "exception_" + head_functor
    tables = _SymbolTables.from_clauses(_kb_clauses(kb))
    tables.functors[(exc, 1)] += 1
    tables.functors[("not", 1)] += 1
    n_f = max(1, len(tables.functors))
    n_c = max(1, len(tables.constants))
    name_charge = _symbol_decl_bits(1)   # fresh exception functor, arity 1
    # the added body literal not(exc(X)): two functor occurrences + one var
    body_literal = (2 + math.log2(n_f)) * 2 + (2 + math.log2(max(1, 1)))
    # each exception fact: 4 + functor occurrence + one constant occurrence
    per_fact = 4 + (2 + math.log2(n_f)) + (2 + math.log2(n_c))
    return name_charge + body_literal + n_corrections * per_fact
