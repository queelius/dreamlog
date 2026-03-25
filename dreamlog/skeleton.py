"""
Rule-set skeleton extraction for structural comparison.

A skeleton captures the structure of a predicate's rule set while abstracting
away functor names. Two predicates with identical skeletons are structurally
identical and candidates for predicate invention.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from .terms import Term, Atom, Variable, Compound
from .knowledge import Rule


@dataclass(frozen=True)
class RuleSkeleton:
    """Skeleton of a single rule."""
    head_arity: int
    body: tuple  # tuple of (functor_role: str, arity: int)
    variable_map: tuple  # tuple of (tuple of (goal_idx, arg_idx), ...)


@dataclass(frozen=True)
class RuleSetSkeleton:
    """Skeleton of a complete rule set."""
    rules: tuple  # tuple of RuleSkeleton, sorted
    param_count: int


def extract_skeleton(
    predicate: str, rules: List[Rule]
) -> Tuple[RuleSetSkeleton, Dict[str, str]]:
    """Extract skeleton and functor mapping from a rule set."""
    param_counter = [0]
    functor_to_role: Dict[str, str] = {}
    functor_map: Dict[str, str] = {}

    def _get_role(functor_name: str) -> str:
        if functor_name == predicate:
            return "SELF"
        if functor_name in functor_to_role:
            return functor_to_role[functor_name]
        role = f"PARAM_{param_counter[0]}"
        param_counter[0] += 1
        functor_to_role[functor_name] = role
        functor_map[role] = functor_name
        return role

    # First pass: discover functor roles in deterministic order
    sorted_rules = sorted(rules, key=lambda r: (len(r.body), str(r)))
    for rule in sorted_rules:
        for goal in rule.body:
            if isinstance(goal, Compound):
                _get_role(goal.functor)

    # Second pass: build skeletons
    rule_skeletons = []
    for rule in sorted_rules:
        head_arity = rule.head.arity if isinstance(rule.head, Compound) else 0
        var_order = _determine_variable_order(rule)
        var_rename = {old: f"_V{i}" for i, old in enumerate(var_order)}
        var_positions = _collect_variable_positions(rule)
        var_map = _build_variable_map(var_positions, var_rename)

        body_parts = []
        for goal in rule.body:
            if isinstance(goal, Compound):
                role = _get_role(goal.functor)
                body_parts.append((role, goal.arity))
            else:
                body_parts.append(("UNKNOWN", 0))

        rule_skeletons.append(RuleSkeleton(
            head_arity=head_arity,
            body=tuple(body_parts),
            variable_map=var_map,
        ))

    rule_skeletons.sort(key=lambda s: (len(s.body), s.body, s.variable_map))

    return (
        RuleSetSkeleton(rules=tuple(rule_skeletons), param_count=param_counter[0]),
        functor_map,
    )


def _determine_variable_order(rule: Rule) -> List[str]:
    """Variable ordering: left-to-right through head then body."""
    seen = []
    for name in _iter_variables(rule.head):
        if name not in seen:
            seen.append(name)
    for goal in rule.body:
        for name in _iter_variables(goal):
            if name not in seen:
                seen.append(name)
    return seen


def _iter_variables(term: Term):
    """Yield variable names left-to-right."""
    if isinstance(term, Variable):
        yield term.name
    elif isinstance(term, Compound):
        for arg in term.args:
            yield from _iter_variables(arg)


def _collect_variable_positions(rule: Rule) -> Dict[str, List[Tuple[int, int]]]:
    """Collect (goal_index, arg_index) positions per variable. Head = -1."""
    positions: Dict[str, List[Tuple[int, int]]] = {}

    def _record(term: Term, goal_idx: int, arg_idx: int):
        if isinstance(term, Variable):
            positions.setdefault(term.name, []).append((goal_idx, arg_idx))
        elif isinstance(term, Compound):
            for i, arg in enumerate(term.args):
                _record(arg, goal_idx, i)

    if isinstance(rule.head, Compound):
        for i, arg in enumerate(rule.head.args):
            _record(arg, -1, i)

    for g_idx, goal in enumerate(rule.body):
        if isinstance(goal, Compound):
            for i, arg in enumerate(goal.args):
                _record(arg, g_idx, i)

    return positions


def _build_variable_map(
    positions: Dict[str, List[Tuple[int, int]]],
    var_rename: Dict[str, str],
) -> tuple:
    """Build normalized variable connectivity map."""
    renamed = {}
    for old_name, pos_list in positions.items():
        new_name = var_rename.get(old_name, old_name)
        renamed[new_name] = tuple(sorted(pos_list))
    sorted_vars = sorted(renamed.keys())
    return tuple(renamed[v] for v in sorted_vars)
