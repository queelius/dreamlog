"""Factor (Operations D and E): introduce a definition for shared structure
and rewrite users. Two detectors behind one module; both interleaved
run-form (D allocates invented names from the live KB pre-verification;
E is round-based over the live KB).
"""
from typing import Dict, List, Optional

from ...terms import Term, Atom, Variable, Compound
from ...knowledge import KnowledgeBase, Fact, Rule
from ..util import _is_system_predicate, _next_generated_name
from ..proposal import Proposal
from ..gate import Accepted


# ── Operation E helpers (moved verbatim from KnowledgeBaseDreamer) ──


def _iter_var_names(term: Term):
    """Yield variable names left-to-right in a term."""
    if isinstance(term, Variable):
        yield term.name
    elif isinstance(term, Compound):
        for arg in term.args:
            yield from _iter_var_names(arg)


def _subseq_structural_key(subseq: list) -> tuple:
    """Compute structural key for a contiguous sub-sequence of body goals.

    Captures: length, functor names, arities, and normalized variable
    connectivity.
    """
    length = len(subseq)
    functors = []
    for goal in subseq:
        if isinstance(goal, Compound):
            functors.append((goal.functor, goal.arity))
        else:
            functors.append((str(goal), 0))

    # Normalize variables within the sub-sequence
    var_order = []
    for goal in subseq:
        for name in _iter_var_names(goal):
            if name not in var_order:
                var_order.append(name)

    var_rename = {old: f"_S{i}" for i, old in enumerate(var_order)}

    # Build variable connectivity
    var_positions: Dict[str, list] = {}
    for g_idx, goal in enumerate(subseq):
        if isinstance(goal, Compound):
            for a_idx, arg in enumerate(goal.args):
                if isinstance(arg, Variable):
                    renamed = var_rename.get(arg.name, arg.name)
                    var_positions.setdefault(renamed, []).append((g_idx, a_idx))

    sorted_vars = sorted(var_positions.keys())
    var_map = tuple(tuple(sorted(var_positions[v])) for v in sorted_vars)

    return (length, tuple(functors), var_map)


def _compute_interface_vars(rule: Rule, start: int,
                             length: int) -> List[str]:
    """Compute interface variables for a sub-sequence extraction.

    Interface vars = variables appearing in both the sub-sequence AND
    the rest of the rule (head + remaining body goals).
    """
    body = list(rule.body)
    subseq_goals = body[start:start + length]
    rest_goals = [rule.head] + body[:start] + body[start + length:]

    subseq_vars = set()
    for goal in subseq_goals:
        subseq_vars.update(goal.get_variables())

    rest_vars = set()
    for goal in rest_goals:
        rest_vars.update(goal.get_variables())

    interface = subseq_vars & rest_vars

    # Order by first appearance in sub-sequence
    ordered = []
    for goal in subseq_goals:
        for name in _iter_var_names(goal):
            if name in interface and name not in ordered:
                ordered.append(name)

    return ordered


def _map_interface_vars(rule: Rule, start: int, length: int,
                        template_interface: List[str],
                        template_rule: Rule, template_start: int
                        ) -> list:
    """Map this occurrence's variables to the extracted predicate's interface."""
    body = list(rule.body)
    subseq = body[start:start + length]
    template_body = list(template_rule.body)
    template_subseq = template_body[template_start:template_start + length]

    # Build mapping from template var names to this occurrence's var names
    var_map = {}
    for t_goal, o_goal in zip(template_subseq, subseq):
        if isinstance(t_goal, Compound) and isinstance(o_goal, Compound):
            for t_arg, o_arg in zip(t_goal.args, o_goal.args):
                if isinstance(t_arg, Variable) and isinstance(o_arg, Variable):
                    var_map[t_arg.name] = o_arg.name

    return [Variable(var_map.get(v, v)) for v in template_interface]


def _find_best_body_pattern(kb: KnowledgeBase,
                            exclude_keys: set = None):
    """Find the best common contiguous sub-sequence across rule bodies."""
    if exclude_keys is None:
        exclude_keys = set()
    # Collect rules to scan (skip generated predicates)
    rules = []
    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            if _is_system_predicate(rule.head.functor):
                continue
        if len(rule.body) >= 2:
            rules.append(rule)

    if not rules:
        return None

    # Index all sub-sequences by structural key
    subseq_index: Dict[tuple, list] = {}
    for rule in rules:
        body = list(rule.body)
        for length in range(2, len(body) + 1):
            for start in range(len(body) - length + 1):
                subseq = body[start:start + length]
                key = _subseq_structural_key(subseq)
                subseq_index.setdefault(key, []).append((rule, start))

    # Find best candidate: highest savings
    best = None
    best_savings = 0
    for key, occurrences in subseq_index.items():
        # Deduplicate: one occurrence per rule
        seen_rules = set()
        unique_occs = []
        for rule, start in occurrences:
            rule_id = id(rule)
            if rule_id not in seen_rules:
                seen_rules.add(rule_id)
                unique_occs.append((rule, start))

        n = len(unique_occs)
        k = key[0]  # length stored as first element of key
        if n < 2 or k < 2:
            continue
        if key in exclude_keys:
            continue
        body_goal_savings = (k - 1) * (n - 1) - 1
        # Also check clause count: extraction adds 1 rule but may enable
        # trivial wrapper elimination. Count rules that become single-goal
        # wrappers (body = just the extracted call).
        wrappers_created = sum(1 for rule, start in unique_occs
                               if len(rule.body) == k)  # entire body is the subseq
        # Net clause change: +1 (extracted) - wrappers_created (if they become
        # redundant with the extracted predicate, a later dream cycle removes them)
        # For now, require positive body-goal savings
        savings = body_goal_savings
        if savings > best_savings:
            best_savings = savings
            best = (key, unique_occs)

    if best is None:
        return None

    key, occurrences = best
    # Recover the actual sub-sequence goals from first occurrence
    first_rule, first_start = occurrences[0]
    subseq_len = key[0]
    subseq = list(first_rule.body)[first_start:first_start + subseq_len]
    return (subseq, occurrences, best[0])


# ── Operation D: Predicate invention ──


def run_invention(kb: KnowledgeBase, suite, gate_apply, policy,
                  rejections: list) -> list:
    """Operation D: invent predicates from structurally identical rule sets.

    Allocates the invented name from the LIVE KB before verification;
    enforces strict reduction (k + n >= n*k guard) unchanged.
    """
    from ...skeleton import extract_skeleton
    from ...kb_dreamer import CompressionCandidate

    ops = []

    # Group rules by head functor
    pred_rules: Dict[str, List[Rule]] = {}
    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            f = rule.head.functor
            if _is_system_predicate(f):
                continue
            pred_rules.setdefault(f, []).append(rule)

    # Extract skeletons and group
    skeleton_groups: Dict = {}
    for pred, rules in pred_rules.items():
        if not rules:
            continue
        skeleton, fmap = extract_skeleton(pred, rules)
        if skeleton.param_count != 1:
            continue
        skeleton_groups.setdefault(skeleton, []).append((pred, rules, fmap))

    for skeleton, members in skeleton_groups.items():
        n = len(members)
        k = len(skeleton.rules)
        if k <= 1:
            continue
        if k + n >= n * k:
            continue

        invented_name = _next_generated_name(kb, "_invented_")
        template_pred, template_rules, template_fmap = members[0]
        param_functor = template_fmap["PARAM_0"]

        # Sort template rules by body length for determinism
        sorted_template = sorted(template_rules, key=lambda r: len(r.body))

        invented_rules = []
        for rule in sorted_template:
            param_var = Variable("R")
            old_head = rule.head
            new_head_args = [param_var] + list(old_head.args)
            new_head = Compound(invented_name, new_head_args)

            new_body = []
            for goal in rule.body:
                if isinstance(goal, Compound):
                    if goal.functor == template_pred:
                        new_args = [param_var] + list(goal.args)
                        new_body.append(Compound(invented_name, new_args))
                    elif goal.functor == param_functor:
                        call_args = [param_var] + list(goal.args)
                        new_body.append(Compound("call", call_args))
                    else:
                        new_body.append(goal)
                else:
                    new_body.append(goal)

            invented_rules.append(Rule(new_head, new_body))

        wrapper_rules = []
        for pred, rules, fmap in members:
            actual_functor = fmap["PARAM_0"]
            head_arity = rules[0].head.arity if isinstance(rules[0].head, Compound) else 0
            wrapper_vars = [Variable(f"_W{i}") for i in range(head_arity)]
            wrapper_head = Compound(pred, wrapper_vars)
            wrapper_body_args = [Atom(actual_functor)] + wrapper_vars
            wrapper_body = [Compound(invented_name, wrapper_body_args)]
            wrapper_rules.append(Rule(wrapper_head, wrapper_body))

        all_original = []
        for pred, rules, fmap in members:
            all_original.extend(rules)
        all_new = invented_rules + wrapper_rules

        proposal = Proposal(kind="invention",
                            remove=tuple(all_original),
                            add=tuple(all_new))
        res = gate_apply(kb, proposal, policy)
        if not isinstance(res, Accepted):
            rejections.append((proposal.kind, res.reason))
            continue
        ops.append(res.candidate)

    return ops


# ── Operation E: Body pattern extraction ──


def run_extraction(kb: KnowledgeBase, suite, gate_apply, policy,
                   rejections: list, max_rounds: int = 10) -> list:
    """Operation E: extract common contiguous sub-goal sequences from rule bodies.

    Delta is always +1 in clause-count terms (one extracted rule added,
    occurrences rewritten 1:1). ExtractionPolicy sets require_negative_delta=False
    to exempt this operation from the strict-delta gate check.
    """
    from ...kb_dreamer import CompressionCandidate

    all_ops = []
    failed_keys: set = set()

    for _round in range(max_rounds):
        candidate = _find_best_body_pattern(kb, exclude_keys=failed_keys)
        if candidate is None:
            break

        subseq, occurrences, pattern_key = candidate

        # Compute interface variables for each occurrence
        extracted_name = _next_generated_name(kb, "_extracted_")
        subseq_len = len(subseq)

        first_rule, first_start = occurrences[0]
        interface_vars = _compute_interface_vars(
            first_rule, first_start, subseq_len)

        template_rule = first_rule
        template_body = list(template_rule.body)
        subseq_goals = template_body[first_start:first_start + subseq_len]

        extracted_head = Compound(extracted_name,
                                  [Variable(v) for v in interface_vars])
        extracted_rule = Rule(extracted_head, subseq_goals)

        # Compute rewrites once (same as apply path in original)
        original_rules = []
        new_rules = []
        for rule, start in occurrences:
            body = list(rule.body)
            call_args = _map_interface_vars(
                rule, start, subseq_len, interface_vars, first_rule, first_start)
            new_body = (body[:start]
                        + [Compound(extracted_name, call_args)]
                        + body[start + subseq_len:])
            new_rule = Rule(rule.head, new_body)
            original_rules.append(rule)
            new_rules.append(new_rule)

        proposal = Proposal(kind="extraction",
                            remove=tuple(original_rules),
                            add=(extracted_rule,) + tuple(new_rules))
        res = gate_apply(kb, proposal, policy)
        if not isinstance(res, Accepted):
            rejections.append((proposal.kind, res.reason))
            failed_keys.add(pattern_key)
            continue
        all_ops.append(res.candidate)

    return all_ops
