"""Generalize (Operation C): facts -> guarded rule with not/1 exceptions.

Interleaved run-form generator: detection reads the live KB between accepted
candidates, so all mutation flows through the gate but iteration order is
owned here, byte-identical to the original op.
"""
from typing import List, Optional, Union

from ...terms import Atom, Variable, Compound
from ...knowledge import KnowledgeBase, Fact, Rule
from ..util import _is_system_predicate
from ..proposal import Proposal
from ..gate import Accepted


def find_guard(kb: KnowledgeBase, functor: str,
               needed_values: set) -> Optional[tuple]:
    """Find a unary guard predicate whose extension covers needed_values."""
    candidates = []
    unary_groups: dict = {}
    for fact in kb.facts:
        term = fact.term
        if (isinstance(term, Compound) and term.arity == 1
                and isinstance(term.args[0], Atom)):
            f = term.functor
            if f == functor or _is_system_predicate(f):
                continue
            unary_groups.setdefault(f, set()).add(term.args[0].value)

    for guard_f, guard_vals in unary_groups.items():
        if needed_values.issubset(guard_vals):
            excess = len(guard_vals - needed_values)
            candidates.append((guard_f, guard_vals, excess))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[2])
    return (candidates[0][0], candidates[0][1])


def run(kb: KnowledgeBase, suite, gate_apply, policy,
        min_group_size: int, rejections: list) -> List:
    """Run Operation C on kb, routing each candidate through gate_apply.

    Interleaved: each accepted candidate mutates kb immediately; later
    subgroups within the same functor group see the updated fact list.
    The functor-group snapshot (groups) is taken once at entry and
    deliberately excludes the exception_* predicates introduced here.

    ``suite`` lives in the policy; parameter kept for interface uniformity.
    """
    # suite lives in the policy; parameter kept for interface uniformity
    ops = []

    # Group facts by functor/arity
    groups: dict = {}
    for fact in kb.facts:
        term = fact.term
        if isinstance(term, Compound):
            key = (term.functor, term.arity)
            groups.setdefault(key, []).append(fact)

    for (functor, arity), all_facts in groups.items():
        if _is_system_predicate(functor):
            continue
        if len(all_facts) < min_group_size:
            continue

        # Try each argument position as the varying one
        for var_pos in range(arity):
            # Partition by constant pattern (all args except var_pos)
            subgroups: dict = {}
            for fact in all_facts:
                term = fact.term
                pattern = tuple(
                    term.args[i] for i in range(arity) if i != var_pos)
                subgroups.setdefault(pattern, []).append(fact)

            for pattern, facts in subgroups.items():
                if len(facts) < min_group_size:
                    continue

                # Check all values at var_pos are atoms
                fact_values = set()
                all_atoms = True
                for fact in facts:
                    arg = fact.term.args[var_pos]
                    if isinstance(arg, Atom):
                        fact_values.add(arg.value)
                    else:
                        all_atoms = False
                        break
                if not all_atoms:
                    continue

                # Find guard predicate
                guard = find_guard(kb, functor, fact_values)
                if guard is None:
                    continue

                guard_functor, guard_values = guard
                exceptions = guard_values - fact_values
                cost_before = len(facts)
                cost_after = 1 + len(exceptions)
                if cost_after >= cost_before:
                    continue

                # Build the constant args for the rule head
                constant_args = list(pattern)

                # Build exception functor name from constants
                constant_str = "_".join(
                    str(a.value) if isinstance(a, Atom) else str(a)
                    for a in constant_args) if constant_args else ""
                exception_functor = (
                    f"exception_{functor}_{constant_str}_{guard_functor}"
                    if constant_str
                    else f"exception_{functor}_{guard_functor}")

                new_clauses: List[Union[Fact, Rule]] = []

                # Build generalizing rule
                rule_var = Variable("X")
                rule_args = []
                pattern_idx = 0
                for i in range(arity):
                    if i == var_pos:
                        rule_args.append(rule_var)
                    else:
                        rule_args.append(constant_args[pattern_idx])
                        pattern_idx += 1

                rule_head = Compound(functor, rule_args)
                rule_body = [
                    Compound(guard_functor, [rule_var]),
                    Compound("not", [
                        Compound(exception_functor, [rule_var])]),
                ]
                new_clauses.append(Rule(rule_head, rule_body))

                for exc_val in sorted(exceptions, key=str):
                    new_clauses.append(
                        Fact(Compound(exception_functor, [Atom(exc_val)])))

                proposal = Proposal(kind="generalization",
                                    remove=tuple(facts),
                                    add=tuple(new_clauses))
                res = gate_apply(kb, proposal, policy)
                if not isinstance(res, Accepted):
                    rejections.append((proposal.kind, res.reason))
                    continue
                ops.append(res.candidate)

                # Rebuild all_facts since KB changed
                all_facts = [
                    f for f in kb.facts
                    if isinstance(f.term, Compound)
                    and f.term.functor == functor
                    and f.term.arity == arity]
                break  # restart position scan with updated facts

    return ops
