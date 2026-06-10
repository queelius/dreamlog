"""
Knowledge Base Dreamer - Sleep phase symbolic compression.

Implements the sleep/dream cycle via eight operations:
A. Subsumption elimination
B. Redundant fact pruning
C. Fact generalization with exceptions
D. Predicate invention
E. Rule body pattern extraction
F. Dead clause pruning
G. LLM-assisted compression
H. Lemma caching

All operations except G are purely symbolic (no LLM). Compression is
guided by Minimum Description Length: compress only when the result is
shorter.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from .terms import Term, Atom, Variable, Compound
from .knowledge import KnowledgeBase, Fact, Rule
from .unification import clause_subsumes, subsumes
from .evaluator import PrologEvaluator

from .compression.util import (_is_system_predicate, _next_generated_name,
                               _strip_llm_noise, _filter_cyclic_rules,
                               _collect_user_functors)


@dataclass
class CompressionCandidate:
    """A single compression operation that was applied."""
    operation: str
    original_clauses: List[Union[Fact, Rule]]
    new_clauses: List[Union[Fact, Rule]] = field(default_factory=list)

    @property
    def mdl_delta(self) -> int:
        return len(self.new_clauses) - len(self.original_clauses)

    @property
    def is_worth_it(self) -> bool:
        return self.mdl_delta < 0


@dataclass
class VerificationResult:
    """Result of verifying a compressed KB."""
    passed: bool
    failures: List[tuple]
    positive_count: int
    negative_count: int


@dataclass
class VerificationSuite:
    """Test suite for verifying KB compression preserves semantics."""
    positive_queries: List[Term]
    negative_queries: List[Term]

    def verify(self, kb: KnowledgeBase, evaluator_factory) -> VerificationResult:
        failures = []
        for q in self.positive_queries:
            ev = evaluator_factory(kb)
            if not ev.has_solution(q):
                failures.append(("false_negative", q))
        for q in self.negative_queries:
            ev = evaluator_factory(kb)
            if ev.has_solution(q):
                failures.append(("false_positive", q))
        return VerificationResult(
            passed=len(failures) == 0, failures=failures,
            positive_count=len(self.positive_queries),
            negative_count=len(self.negative_queries))


def build_verification_suite(kb: KnowledgeBase) -> VerificationSuite:
    """Build verification suite from current KB state."""
    positive = [fact.term for fact in kb.facts]

    atom_pool: Set[Any] = set()
    fact_terms_by_key = {}
    for fact in kb.facts:
        term = fact.term
        if isinstance(term, Compound):
            key = (term.functor, term.arity)
            fact_terms_by_key.setdefault(key, []).append(term)
            for arg in term.args:
                if isinstance(arg, Atom):
                    atom_pool.add(arg.value)
        elif isinstance(term, Atom):
            atom_pool.add(term.value)

    negative: List[Term] = []
    max_negative = 2 * len(positive)
    for (functor, arity), terms in fact_terms_by_key.items():
        if len(negative) >= max_negative:
            break
        for pos in range(arity):
            existing_values = set()
            for t in terms:
                if isinstance(t.args[pos], Atom):
                    existing_values.add(t.args[pos].value)
            novel_values = atom_pool - existing_values
            if not novel_values:
                continue
            novel = sorted(novel_values, key=str)[0]
            base = terms[0]
            new_args = list(base.args)
            new_args[pos] = Atom(novel)
            neg_term = Compound(functor, new_args)
            if neg_term not in [f.term for f in kb.facts]:
                negative.append(neg_term)
            if len(negative) >= max_negative:
                break

    return VerificationSuite(positive_queries=positive,
                             negative_queries=negative)


def extend_verification_for_rules(suite: VerificationSuite,
                                  kb: KnowledgeBase,
                                  max_queries: int = 50) -> None:
    """Extend verification suite with rule-derived positive/negative queries.

    For each rule-defined predicate, generate ground queries using atom values
    from the KB and test which are derivable.
    """
    ev = PrologEvaluator(kb)

    atom_values = set()
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            for arg in fact.term.args:
                if isinstance(arg, Atom):
                    atom_values.add(arg.value)

    if not atom_values:
        return

    atoms = sorted(atom_values, key=str)

    rule_preds = {}
    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            key = (rule.head.functor, rule.head.arity)
            rule_preds[key] = True

    added = 0
    for (functor, arity) in rule_preds:
        if added >= max_queries:
            break
        if _is_system_predicate(functor):
            continue
        # Sample size scales inversely with atom pool size to keep total bounded
        sample = min(5, len(atoms))
        if arity == 1:
            candidates = [Compound(functor, [Atom(a)]) for a in atoms[:sample]]
        elif arity == 2:
            candidates = [Compound(functor, [Atom(a), Atom(b)])
                          for a in atoms[:sample] for b in atoms[:sample]]
        else:
            continue

        for candidate in candidates:
            if added >= max_queries:
                break
            if ev.has_solution(candidate):
                if candidate not in suite.positive_queries:
                    suite.positive_queries.append(candidate)
                    added += 1
            else:
                if candidate not in suite.negative_queries:
                    suite.negative_queries.append(candidate)
                    added += 1


@dataclass
class DreamSession:
    """Results from a dream cycle."""
    compressed: bool
    operations: List[CompressionCandidate]
    compression_ratio: float
    verification: Optional[VerificationResult]

    @property
    def clauses_removed(self) -> int:
        return sum(-op.mdl_delta for op in self.operations)


class KnowledgeBaseDreamer:
    """Symbolic sleep-phase compression via anti-unification and MDL."""

    def __init__(self, min_group_size: int = 3,
                 shared_structure_threshold: float = 0.1,
                 llm_client=None,
                 max_prompt_facts: int = 50,
                 open_world: bool = False,
                 discover_recursion: bool = False,
                 min_base_facts: int = 3,
                 disable_op_c: bool = False):
        self.min_group_size = min_group_size
        self.shared_structure_threshold = shared_structure_threshold
        self.llm_client = llm_client
        self.max_prompt_facts = max_prompt_facts
        # Open-world mode: when True, Op G's false-positive check accepts
        # rules that derive ground terms not in the KB (instead of rejecting).
        # This is necessary for holdout-style evaluations where the goal is
        # precisely to recover absent facts. Default False (closed-world)
        # preserves the safety guarantee for the standard compression setting.
        self.open_world = open_world
        # Operation I (recursive closure discovery): off by default so the
        # standard compression pipeline is unchanged (zero drift).
        self.discover_recursion = discover_recursion
        self.min_base_facts = min_base_facts
        # Skip Operation C (fact generalization). Off by default; used by the
        # EX28 within-predicate LLM-only ablation condition.
        self.disable_op_c = disable_op_c

    def dream(self, kb: KnowledgeBase, verify: bool = True) -> DreamSession:
        original_size = len(kb)
        if original_size == 0:
            return DreamSession(compressed=False, operations=[],
                                compression_ratio=1.0, verification=None)

        snapshot = kb.copy()
        # Save wake-phase tracking data before dream operations inflate it
        wake_usage = dict(kb._usage_counts)
        wake_derivation_counts = dict(kb._derivation_counts)
        wake_derivation_terms = dict(kb._derivation_terms)
        suite = build_verification_suite(kb) if verify else None
        result = None
        ops: List[CompressionCandidate] = []

        # Operation F runs first, using only wake-phase usage data
        # (before verification queries inflate usage counts).
        # Protect user-provided seed facts from pruning — only prune
        # facts/rules added during previous dream cycles (lemmas, LLM rules).
        seed_terms = {f.term for f in kb.facts}
        seed_rules = {(r.head, tuple(r.body)) for r in kb.rules}
        dead_ops = self._prune_dead_clauses(kb, seed_terms=seed_terms,
                                             seed_rules=seed_rules)
        ops.extend(dead_ops)
        # Remove dead clauses from verification suite (they're dead by definition)
        if suite and dead_ops:
            dead_terms = set()
            for op in dead_ops:
                for clause in op.original_clauses:
                    if isinstance(clause, Fact):
                        dead_terms.add(clause.term)
            suite.positive_queries = [
                q for q in suite.positive_queries if q not in dead_terms]

        ops.extend(self._eliminate_subsumed(kb))
        ops.extend(self._prune_redundant_facts(kb))
        if not self.disable_op_c:
            ops.extend(self._generalize_facts(kb, suite))

        # Extend verification for rule-derived queries before Operation D
        if verify and suite:
            extend_verification_for_rules(suite, kb)
        ops.extend(self._invent_predicates(kb, suite))
        ops.extend(self._extract_body_patterns(kb, suite))

        # Operation I: recursive closure discovery (flag-gated, off by default).
        # Runs in the symbolic phase so symbolic-only and full-pipeline differ
        # only by Operation G.
        if self.discover_recursion:
            ops.extend(self._discover_recursion(kb, suite))

        # LLM-assisted naming (after symbolic ops, before final verify)
        self._name_invented_predicates(kb)

        # Operation G: LLM-assisted compression
        llm_ops = self._llm_compress(kb, suite)
        ops.extend(llm_ops)

        # After LLM-proposed rules, use bounded evaluators to prevent
        # combinatorial explosion from loops (e.g. parent←father + father←parent)
        if llm_ops:
            ops.extend(self._prune_redundant_facts(kb, max_calls=500))

        # Operation H: Lemma caching (add frequently-derived terms as facts)
        ops.extend(self._cache_lemmas(kb))

        if verify and suite:
            max_calls = 500 if llm_ops else 0
            result = suite.verify(
                kb, lambda k, _mc=max_calls: PrologEvaluator(k, max_total_calls=_mc))
            if not result.passed:
                kb.restore_from(snapshot)
                return DreamSession(compressed=False, operations=[],
                                    compression_ratio=1.0,
                                    verification=result)

        # Restore wake-phase usage data (discard usage from verification queries)
        kb._usage_counts = wake_usage
        kb._derivation_counts = wake_derivation_counts
        kb._derivation_terms = wake_derivation_terms

        new_size = len(kb)
        return DreamSession(
            compressed=new_size < original_size, operations=ops,
            compression_ratio=new_size / original_size if original_size > 0 else 1.0,
            verification=result)

    def _eliminate_subsumed(self, kb: KnowledgeBase) -> List[CompressionCandidate]:
        """Operation A: Remove clauses subsumed by more general clauses."""
        ops = []

        # A1: Rule-vs-rule subsumption (same body length)
        rules_to_remove = set()
        rules = kb.rules
        for i, rule_a in enumerate(rules):
            for j, rule_b in enumerate(rules):
                if i == j or j in rules_to_remove:
                    continue
                if clause_subsumes(rule_a, rule_b) and rule_a != rule_b:
                    rules_to_remove.add(j)

        for idx in sorted(rules_to_remove, reverse=True):
            removed = rules[idx]
            kb.remove_rule_by_value(removed)
            ops.append(CompressionCandidate(
                operation="subsumption", original_clauses=[removed]))

        # A2: Bodyless-rule-vs-fact subsumption
        facts_to_remove = []
        bodyless_rules = [r for r in kb.rules if len(r.body) == 0]
        for fact in kb.facts:
            for rule in bodyless_rules:
                if subsumes(rule.head, fact.term):
                    facts_to_remove.append(fact)
                    break

        for fact in facts_to_remove:
            kb.remove_fact_by_value(fact)
            ops.append(CompressionCandidate(
                operation="subsumption", original_clauses=[fact]))

        return ops

    def _prune_redundant_facts(self, kb: KnowledgeBase,
                               max_calls: int = 0) -> List[CompressionCandidate]:
        """Operation B: Remove facts derivable from remaining KB."""
        ops = []
        facts = kb.facts

        def _make_ev(k):
            return PrologEvaluator(k, max_total_calls=max_calls)

        # Phase 1: Find independently redundant facts
        redundant = []
        for fact in facts:
            kb.remove_fact_by_value(fact)
            ev = _make_ev(kb)
            try:
                is_derivable = ev.has_solution(fact.term)
            except RecursionError:
                is_derivable = False
            if is_derivable:
                redundant.append(fact)
            kb.add_fact(fact)

        if not redundant:
            return ops

        # Phase 2: Batch remove
        for fact in redundant:
            kb.remove_fact_by_value(fact)

        # Phase 3: Verify all still derivable
        ev = _make_ev(kb)
        try:
            still_ok = all(ev.has_solution(f.term) for f in redundant)
        except RecursionError:
            still_ok = False

        if still_ok:
            for fact in redundant:
                ops.append(CompressionCandidate(
                    operation="pruning", original_clauses=[fact]))
            return ops

        # Phase 4: Fallback to one-at-a-time
        for fact in redundant:
            kb.add_fact(fact)
        for fact in redundant:
            kb.remove_fact_by_value(fact)
            ev = _make_ev(kb)
            try:
                is_derivable = ev.has_solution(fact.term)
            except RecursionError:
                is_derivable = False
            if is_derivable:
                ops.append(CompressionCandidate(
                    operation="pruning", original_clauses=[fact]))
            else:
                kb.add_fact(fact)

        return ops

    def _generalize_facts(self, kb: KnowledgeBase,
                          suite: Optional['VerificationSuite'] = None
                          ) -> List[CompressionCandidate]:
        """Operation C: Generalize fact subgroups into rules with exceptions.

        Uses argument-position partitioning to find compressible subgroups
        within each functor/arity group. For each argument position p, facts
        are grouped by their constant pattern (all args except position p).
        Each subgroup where exactly position p varies is a compression candidate.
        """
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
            if len(all_facts) < self.min_group_size:
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
                    if len(facts) < self.min_group_size:
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
                    guard = self._find_guard(kb, functor, fact_values)
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

                    # Verify candidate
                    if suite is not None:
                        test_kb = kb.copy()
                        test_kb.replace_facts(facts, new_clauses)
                        result = suite.verify(
                            test_kb, lambda k: PrologEvaluator(k))
                        if not result.passed:
                            continue

                    # Apply (greedy: remove these facts, later subgroups
                    # will work on remaining facts)
                    kb.replace_facts(facts, new_clauses)
                    ops.append(CompressionCandidate(
                        operation="generalization",
                        original_clauses=list(facts),
                        new_clauses=list(new_clauses)))

                    # Rebuild all_facts since KB changed
                    all_facts = [
                        f for f in kb.facts
                        if isinstance(f.term, Compound)
                        and f.term.functor == functor
                        and f.term.arity == arity]
                    break  # restart position scan with updated facts

        return ops

    def _find_guard(self, kb: KnowledgeBase, functor: str,
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

    def _invent_predicates(self, kb: KnowledgeBase,
                           suite: Optional['VerificationSuite'] = None
                           ) -> List[CompressionCandidate]:
        """Operation D: Invent predicates from structurally identical rule sets."""
        from .skeleton import extract_skeleton

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

            if suite is not None:
                test_kb = kb.copy()
                for rule in all_original:
                    test_kb.remove_rule_by_value(rule)
                for rule in all_new:
                    test_kb.add_rule(rule)
                result = suite.verify(test_kb, lambda k: PrologEvaluator(k))
                if not result.passed:
                    continue

            for rule in all_original:
                kb.remove_rule_by_value(rule)
            for rule in all_new:
                kb.add_rule(rule)

            ops.append(CompressionCandidate(
                operation="invention",
                original_clauses=all_original,
                new_clauses=all_new))

        return ops

    # ── Operation E: Rule body pattern extraction ──

    def _extract_body_patterns(self, kb: KnowledgeBase,
                               suite: Optional['VerificationSuite'] = None,
                               max_rounds: int = 10
                               ) -> List[CompressionCandidate]:
        """Operation E: Extract common contiguous sub-goal sequences from rule bodies."""
        all_ops = []
        failed_keys: set = set()

        for _round in range(max_rounds):
            candidate = self._find_best_body_pattern(kb, exclude_keys=failed_keys)
            if candidate is None:
                break

            subseq, occurrences, pattern_key = candidate

            # Compute interface variables for each occurrence
            extracted_name = _next_generated_name(kb, "_extracted_")
            subseq_len = len(subseq)

            first_rule, first_start = occurrences[0]
            interface_vars = self._compute_interface_vars(
                first_rule, first_start, subseq_len)

            template_rule = first_rule
            template_body = list(template_rule.body)
            subseq_goals = template_body[first_start:first_start + subseq_len]

            extracted_head = Compound(extracted_name,
                                     [Variable(v) for v in interface_vars])
            extracted_rule = Rule(extracted_head, subseq_goals)

            # Verify before applying
            if suite is not None:
                test_kb = kb.copy()
                test_kb.add_rule(extracted_rule)
                for rule, start in occurrences:
                    body = list(rule.body)
                    call_args = self._map_interface_vars(
                        rule, start, subseq_len, interface_vars, first_rule, first_start)
                    new_body = (body[:start]
                                + [Compound(extracted_name, call_args)]
                                + body[start + subseq_len:])
                    new_rule = Rule(rule.head, new_body)
                    test_kb.remove_rule_by_value(rule)
                    test_kb.add_rule(new_rule)
                result = suite.verify(test_kb, lambda k: PrologEvaluator(k))
                if not result.passed:
                    failed_keys.add(pattern_key)
                    continue

            # Apply
            kb.add_rule(extracted_rule)
            original_rules = []
            new_rules = []
            for rule, start in occurrences:
                body = list(rule.body)
                call_args = self._map_interface_vars(
                    rule, start, subseq_len, interface_vars, first_rule, first_start)
                new_body = (body[:start]
                            + [Compound(extracted_name, call_args)]
                            + body[start + subseq_len:])
                new_rule = Rule(rule.head, new_body)
                kb.remove_rule_by_value(rule)
                kb.add_rule(new_rule)
                original_rules.append(rule)
                new_rules.append(new_rule)

            all_ops.append(CompressionCandidate(
                operation="extraction",
                original_clauses=original_rules,
                new_clauses=[extracted_rule] + new_rules))

        return all_ops

    def _discover_recursion(self, kb: KnowledgeBase,
                            suite: Optional['VerificationSuite'] = None,
                            max_calls: int = 5000
                            ) -> List[CompressionCandidate]:
        """Operation I: discover transitive-closure recursive rules.

        For each binary predicate R whose ground extension equals the
        transitive closure of another binary predicate B, synthesize the
        right-recursive definition (base + recursive case), verify it against
        the suite with a bounded evaluator, and replace R's facts with the
        two rules. Returns at most one candidate per call (re-run on the next
        dream cycle to find further closures).
        """
        from .recursive_discovery import transitive_closure

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
                if R == B or len(b_ext) < self.min_base_facts:
                    continue
                if r_ext != transitive_closure(b_ext):
                    continue

                X, Y, Z = Variable("X"), Variable("Y"), Variable("Z")
                base_rule = Rule(Compound(R, [X, Y]), [Compound(B, [X, Y])])
                rec_rule = Rule(
                    Compound(R, [X, Z]),
                    [Compound(B, [X, Y]), Compound(R, [Y, Z])])
                r_facts = facts_by_pred[R]
                new_clauses: List[Union[Fact, Rule]] = [base_rule, rec_rule]

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
                if suite is not None:
                    test_kb = kb.copy()
                    test_kb.replace_facts(r_facts, new_clauses)
                    try:
                        result = suite.verify(
                            test_kb,
                            lambda k: PrologEvaluator(k, max_total_calls=max_calls))
                    except RecursionError:
                        continue
                    if not result.passed:
                        continue

                kb.replace_facts(r_facts, new_clauses)
                return [CompressionCandidate(
                    operation="recursion",
                    original_clauses=list(r_facts),
                    new_clauses=list(new_clauses))]

        return []

    def _find_best_body_pattern(self, kb: KnowledgeBase,
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
                    key = self._subseq_structural_key(subseq)
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

    def _subseq_structural_key(self, subseq: list) -> tuple:
        """Compute structural key for a contiguous sub-sequence of body goals.

        Captures: length, functor names, arities, and normalized variable connectivity.
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
            for name in self._iter_var_names(goal):
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

    def _iter_var_names(self, term: Term):
        """Yield variable names left-to-right in a term."""
        if isinstance(term, Variable):
            yield term.name
        elif isinstance(term, Compound):
            for arg in term.args:
                yield from self._iter_var_names(arg)

    def _compute_interface_vars(self, rule: Rule, start: int,
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
            for name in self._iter_var_names(goal):
                if name in interface and name not in ordered:
                    ordered.append(name)

        return ordered

    def _map_interface_vars(self, rule: Rule, start: int, length: int,
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

    # ── LLM-assisted naming ──

    def _name_invented_predicates(self, kb: KnowledgeBase) -> None:
        """Ask LLM to suggest names for _invented_N and _extracted_N predicates."""
        if not self.llm_client:
            return

        for prefix in ("_invented_", "_extracted_"):
            generated: Dict[str, List[Rule]] = {}
            for rule in kb.rules:
                if isinstance(rule.head, Compound) and rule.head.functor.startswith(prefix):
                    generated.setdefault(rule.head.functor, []).append(rule)

            for old_name, rules in generated.items():
                rules_str = "\n".join(str(r) for r in rules)
                wrappers = [r for r in kb.rules
                            if isinstance(r.head, Compound)
                            and any(isinstance(g, Compound) and g.functor == old_name
                                    for g in r.body)]
                wrappers_str = "\n".join(str(r) for r in wrappers)

                prompt = (
                    f"This predicate was discovered by compressing a knowledge base:\n\n"
                    f"{rules_str}\n\n"
                    f"It is used as:\n{wrappers_str}\n\n"
                    f"Suggest a short, descriptive name (lowercase, underscores, "
                    f"no spaces). Reply with just the name."
                )

                try:
                    suggested = _strip_llm_noise(
                        self.llm_client.complete(prompt)).lower()
                    if not re.match(r'^[a-z][a-z0-9_]*$', suggested):
                        continue
                    if len(suggested) > 50:
                        continue
                    existing = {r.head.functor for r in kb.rules
                                if isinstance(r.head, Compound)}
                    existing.update(f.term.functor for f in kb.facts
                                    if isinstance(f.term, Compound))
                    if suggested in existing:
                        continue
                    self._rename_predicate(kb, old_name, suggested)
                except Exception:
                    continue

    def _rename_predicate(self, kb: KnowledgeBase, old_name: str,
                          new_name: str) -> None:
        """Rename a predicate throughout the KB."""
        def rename_term(term):
            if isinstance(term, Compound):
                functor = new_name if term.functor == old_name else term.functor
                new_args = [rename_term(a) for a in term.args]
                return Compound(functor, new_args)
            return term

        # Rename in rules
        new_rules = []
        old_rules = list(kb.rules)
        for rule in old_rules:
            new_head = rename_term(rule.head)
            new_body = [rename_term(g) for g in rule.body]
            new_rules.append((rule, Rule(new_head, new_body)))

        for old_rule, new_rule in new_rules:
            if old_rule != new_rule:
                kb.remove_rule_by_value(old_rule)
                kb.add_rule(new_rule)

    # ── Operation G: LLM-assisted compression ──

    def _build_op_g_prompt(self, kb: KnowledgeBase) -> Optional[str]:
        """Build the Operation G prompt string from a knowledge base.

        Samples facts round-robin by predicate, computes predicate fact
        counts, and assembles the full prompt. Returns None when there are
        no facts to prompt with. Reads ``self.max_prompt_facts`` and ``kb``
        only; mutates nothing.
        """
        # Sample facts for the prompt, using Prolog notation
        # Sample facts ensuring every predicate is represented
        max_facts = self.max_prompt_facts
        facts_by_pred: Dict[str, list] = {}
        for fact in kb.facts:
            if isinstance(fact.term, Compound):
                facts_by_pred.setdefault(fact.term.functor, []).append(fact)
        sampled: list = []
        # Round-robin: take up to max_facts / n_preds from each
        per_pred = max(2, max_facts // max(len(facts_by_pred), 1))
        for fn in sorted(facts_by_pred):
            sampled.extend(facts_by_pred[fn][:per_pred])
        sampled = sampled[:max_facts]

        fact_lines = []
        for fact in sampled:
            term = fact.term
            if isinstance(term, Compound):
                args = ", ".join(
                    str(a.value) if isinstance(a, Atom) else str(a)
                    for a in term.args)
                fact_lines.append(f"{term.functor}({args}).")
            else:
                fact_lines.append(f"{term}.")
        if not fact_lines:
            return None

        # Compute predicate fact counts to guide directionality
        pred_counts: Dict[str, int] = {}
        for fact in kb.facts:
            if isinstance(fact.term, Compound):
                pred_counts[fact.term.functor] = (
                    pred_counts.get(fact.term.functor, 0) + 1)
        count_lines = "\n".join(
            f"  {fn}: {cnt} facts" for fn, cnt in
            sorted(pred_counts.items(), key=lambda x: -x[1]))

        prompt = (
            "Given these facts from a knowledge base:\n\n"
            + "\n".join(fact_lines)
            + f"\n\nPredicate fact counts:\n{count_lines}\n\n"
            "Propose rules that derive SPECIFIC predicates from MORE GENERAL ones. "
            "A rule should EXPLAIN why a fact is true using simpler building blocks.\n\n"
            "IMPORTANT constraints:\n"
            "- Rules must go in ONE direction only: specific <- general.\n"
            "- NEVER propose reverse/inverse rules (if father <- parent+male, "
            "do NOT also propose parent <- father).\n"
            "- Body predicates should have MORE facts than the head predicate.\n"
            "- Each rule must derive at least 2 existing facts.\n"
            "- For 'all X satisfy P' patterns (e.g., vegan_recipe requires ALL "
            "ingredients to be vegan), use a helper predicate with not/1:\n"
            '  ["rule", ["has_non_vegan", "X"], [["uses", "X", "Y"], ["vegan", "Y", "false"]]]\n'
            '  ["rule", ["vegan_recipe", "X"], [["recipe", "X"], ["not", ["has_non_vegan", "X"]]]]\n\n'
            "Example format:\n"
            '  ["rule", ["father", "X", "Y"], [["parent", "X", "Y"], ["male", "X"]]]\n'
            '  For not/1: ["not", ["predicate", "X"]] as a body goal.\n\n'
            "If a relation appears to be the transitive closure of another "
            "relation (its facts are exactly the reachable pairs over a base "
            "relation), propose BOTH a base rule and a right-recursive rule, "
            "for example:\n"
            "  ancestor(X, Y) :- parent(X, Y).\n"
            "  ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).\n"
            "Always include the base case. Never write a left-recursive body "
            "(do not put the recursive call first).\n\n"
            "Reply with ONLY a JSON array of rules. "
            "No explanation, no markdown.\n\n"
            "Rules:"
        )
        return prompt

    def _llm_propose(self, kb: KnowledgeBase) -> List[Rule]:
        """Operation G, proposal stage: prompt + parse + validate.

        Builds the prompt, calls the LLM, parses the response, then applies
        Phase 1 structural validation and Phase 2 cyclic filtering. Returns
        the proposed, validated, cycle-filtered rules. Mutates nothing (reads
        ``kb`` only).
        """
        if not self.llm_client:
            return []

        from .llm_response_parser import parse_llm_response

        prompt = self._build_op_g_prompt(kb)
        if prompt is None:
            return []

        raw_rules = self._parse_llm_rules(prompt, parse_llm_response)
        if not raw_rules:
            return []

        # Collect user-defined (non-system) functors once for validation
        kb_functors = _collect_user_functors(kb)

        # Phase 1: Parse and structurally validate all proposed rules
        parsed_rules: List[Rule] = []
        for rule_data in raw_rules:
            try:
                if isinstance(rule_data, Rule):
                    rule = rule_data
                else:
                    rule = self._build_rule_from_parsed(rule_data)
                    if rule is None:
                        continue
                if not isinstance(rule.head, Compound) or not rule.head.functor:
                    continue
                if not rule.body:
                    continue
                if any(not isinstance(g, Compound) or not g.functor
                       for g in rule.body):
                    continue
                # Body functors must be in KB or be builtins (not, call).
                # Also allow functors that appear as heads of OTHER
                # proposed rules (helper predicates like has_non_vegan).
                _BUILTIN_FUNCTORS = {"not", "call"}
                if any(g.functor not in kb_functors
                       and g.functor not in _BUILTIN_FUNCTORS
                       for g in rule.body):
                    continue
                # Reject unstratified negation: head functor inside not/1
                # (e.g., pet(X) :- ..., not(pet(X))) creates a fixed-point
                # paradox where the rule defines pet in terms of not(pet).
                head_fn = rule.head.functor
                if any(g.functor == "not" and g.arity == 1
                       and isinstance(g.args[0], Compound)
                       and g.args[0].functor == head_fn
                       for g in rule.body):
                    continue
                parsed_rules.append(rule)
            except Exception:
                continue

        # Phase 2: Filter out rules that create cross-functor cycles
        # (e.g., parent←father + father←parent). Self-recursion is fine.
        parsed_rules = _filter_cyclic_rules(parsed_rules)

        return parsed_rules

    def _llm_compress(self, kb: KnowledgeBase,
                      suite: Optional['VerificationSuite'] = None
                      ) -> List[CompressionCandidate]:
        """Operation G: Ask LLM to propose cross-functor rules."""
        if not self.llm_client:
            return []

        ops = []

        parsed_rules = self._llm_propose(kb)
        if not parsed_rules:
            return ops

        # Scale call budget with KB size — small KBs need ~500, but
        # transitive predicates in large KBs need proportionally more.
        MAX_CALLS = max(500, len(kb) * 10)

        # Phase 3: Separate helper rules (new predicates) from main rules
        # (derive existing facts). Helpers support main rules (e.g.,
        # has_non_vegan for not(has_non_vegan(X)) in vegan_recipe).
        #
        # An "existing" predicate is one already defined by a fact OR a rule
        # head. Including rule heads matters after a compression op converts a
        # predicate's facts into a rule (Operation I's recursive closure, or
        # Operation C's generalization): that predicate must still count as
        # main, so an LLM proposal for it faces the false-positive check rather
        # than slipping through the unconditional helper path.
        existing_functors = {f.term.functor for f in kb.facts
                             if isinstance(f.term, Compound)}
        existing_functors |= {r.head.functor for r in kb.rules
                              if isinstance(r.head, Compound)}
        helper_rules = [r for r in parsed_rules
                        if r.head.functor not in existing_functors]
        main_rules = [r for r in parsed_rules
                      if r.head.functor in existing_functors]

        # Phase 4: Evaluate main rules (with helpers available)
        accepted_rules: List[Rule] = list(helper_rules)
        for rule in main_rules:
            try:
                # Build test KB with helpers + this rule
                test_kb = kb.copy()
                for h in helper_rules:
                    test_kb.add_rule(h)
                test_kb.add_rule(rule)
                ev = PrologEvaluator(test_kb, max_total_calls=MAX_CALLS)

                derivable_facts = []
                try:
                    for fact in kb.facts:
                        if (isinstance(fact.term, Compound)
                                and fact.term.functor == rule.head.functor):
                            if ev.has_solution(fact.term):
                                derivable_facts.append(fact)
                except RecursionError:
                    continue

                if len(derivable_facts) < 2:
                    continue

                # Verify: adding rule must not break anything
                if suite is not None:
                    def ev_factory(k, _mc=MAX_CALLS):
                        return PrologEvaluator(k, max_total_calls=_mc)
                    try:
                        result = suite.verify(test_kb, ev_factory)
                    except RecursionError:
                        continue
                    if not result.passed:
                        continue

                # False-positive check: the rule must not derive ground
                # terms absent from the KB. Enumerate solutions for the
                # head pattern and reject if any is a novel (spurious) fact.
                #
                # Skipped in open-world mode (e.g., during holdout evaluation)
                # where the goal is precisely to recover absent facts.
                if not self.open_world:
                    existing_terms = {f.term for f in kb.facts
                                      if isinstance(f.term, Compound)
                                      and f.term.functor == rule.head.functor}
                    ev_fp = PrologEvaluator(test_kb, max_total_calls=MAX_CALLS)
                    head_query = rule.head  # has variables
                    has_false_positive = False
                    try:
                        for sol in ev_fp.query([head_query]):
                            ground = head_query.substitute(sol.bindings)
                            if (not ground.get_variables()
                                    and ground not in existing_terms):
                                has_false_positive = True
                                break
                    except RecursionError:
                        has_false_positive = True
                    if has_false_positive:
                        continue

                accepted_rules.append(rule)

            except Exception:
                continue

        # Add all accepted rules and verify the combined set
        if accepted_rules and suite is not None:
            test_kb = kb.copy()
            for rule in accepted_rules:
                test_kb.add_rule(rule)
            def ev_factory(k, _mc=MAX_CALLS):
                return PrologEvaluator(k, max_total_calls=_mc)
            try:
                result = suite.verify(test_kb, ev_factory)
            except RecursionError:
                accepted_rules = []  # Combined set explodes — reject all
            else:
                if not result.passed:
                    accepted_rules = []

        for rule in accepted_rules:
            kb.add_rule(rule)
            ops.append(CompressionCandidate(
                operation="llm_compression",
                original_clauses=[],
                new_clauses=[rule]))

        return ops

    def _parse_llm_rules(self, prompt: str, parse_llm_response) -> list:
        """Send prompt to LLM and parse the response into raw rule data."""
        try:
            response = _strip_llm_noise(self.llm_client.complete(prompt))
            try:
                parsed_json = json.loads(response)
                if isinstance(parsed_json, list):
                    return parsed_json
            except (json.JSONDecodeError, ValueError):
                pass
            # Try line-by-line extraction for partially valid JSON
            raw_rules = []
            for line in response.split("\n"):
                line = line.strip().rstrip(",")
                if line.startswith("[") and "rule" in line:
                    try:
                        raw_rules.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError):
                        continue
            if raw_rules:
                return raw_rules
            # Fall back to the structured parser
            try:
                parsed_resp, _ = parse_llm_response(response)
                return parsed_resp.rules if parsed_resp else []
            except Exception:
                return []
        except Exception:
            return []

    def _build_rule_from_parsed(self, rule_data):
        """Build a Rule from parsed LLM output (raw list format)."""
        try:
            if not isinstance(rule_data, (list, tuple)) or len(rule_data) < 3:
                return None
            head_data = rule_data[1] if isinstance(rule_data[1], list) else rule_data
            body_data = rule_data[2] if len(rule_data) > 2 else []

            def make_term(data):
                if not isinstance(data, list) or len(data) == 0:
                    return None
                functor = data[0]
                if not isinstance(functor, str) or len(functor) == 0:
                    return None
                args = []
                for a in data[1:]:
                    if isinstance(a, str) and len(a) > 0:
                        if a[0].isupper():
                            args.append(Variable(a))
                        else:
                            args.append(Atom(a))
                    elif isinstance(a, list):
                        # Nested term (e.g., not(inner_goal))
                        inner = make_term(a)
                        if inner is None:
                            return None
                        args.append(inner)
                    else:
                        return None
                return Compound(functor, args)

            head = make_term(head_data)
            if head is None:
                return None
            body = []
            for b in body_data:
                bt = make_term(b)
                if bt is None:
                    return None
                body.append(bt)
            if not body:
                return None
            return Rule(head, body)
        except Exception:
            return None

    # ── Operation F: Dead clause pruning ──

    def _prune_dead_clauses(self, kb: KnowledgeBase,
                            min_query_threshold: int = 10,
                            seed_terms: Optional[Set] = None,
                            seed_rules: Optional[Set] = None,
                            ) -> List[CompressionCandidate]:
        """Operation F: Remove clauses with 0 usage after sufficient queries.

        Requires both a minimum total query count AND that at least 50% of
        distinct predicates have been queried. This prevents pruning in KBs
        where the wake phase only exercised a narrow subset of predicates.

        User-provided seed facts/rules (present when dream() was called) are
        never pruned — only derived clauses (lemmas, LLM rules from prior
        cycles) are eligible for dead-clause removal.
        """
        if kb.total_queries_tracked() < min_query_threshold:
            return []

        # Check predicate coverage: at least 50% must have usage
        all_functors = _collect_user_functors(kb)
        if not all_functors:
            return []

        used_functors = set()
        for fact in kb.facts:
            if isinstance(fact.term, Compound) and not _is_system_predicate(fact.term.functor):
                if kb.get_usage(fact) > 0:
                    used_functors.add(fact.term.functor)
        for rule in kb.rules:
            if isinstance(rule.head, Compound) and not _is_system_predicate(rule.head.functor):
                if kb.get_usage(rule) > 0:
                    used_functors.add(rule.head.functor)

        if len(used_functors) / len(all_functors) < 0.5:
            return []

        # Collect dead clauses (unused, non-system, non-seed)
        ops = []
        for fact in kb.facts:
            if isinstance(fact.term, Compound) and _is_system_predicate(fact.term.functor):
                continue
            if seed_terms and fact.term in seed_terms:
                continue
            if kb.get_usage(fact) == 0:
                kb.remove_fact_by_value(fact)
                ops.append(CompressionCandidate(
                    operation="dead_clause", original_clauses=[fact]))

        for rule in kb.rules:
            if isinstance(rule.head, Compound) and _is_system_predicate(rule.head.functor):
                continue
            if seed_rules and (rule.head, tuple(rule.body)) in seed_rules:
                continue
            if kb.get_usage(rule) == 0:
                kb.remove_rule_by_value(rule)
                ops.append(CompressionCandidate(
                    operation="dead_clause", original_clauses=[rule]))

        return ops

    def _frequency_score(self, kb: KnowledgeBase,
                         clauses: List[Union[Fact, Rule]]) -> float:
        """Compute frequency-weighted score for a set of clauses."""
        total = sum(kb.get_usage(c) for c in clauses)
        return 1.0 + math.log2(total + 1)

    # -- Operation H: Lemma caching --

    def _cache_lemmas(self, kb: KnowledgeBase,
                      min_derivation_count: int = 5
                      ) -> List[CompressionCandidate]:
        """Operation H: Cache frequently-derived terms as facts (lemmas).

        Adds ground terms that are derived >= min_derivation_count times
        but not already stored as facts. This speeds up future queries
        by providing direct fact lookup instead of re-derivation.
        """
        ops = []
        frequent = kb.get_frequent_derivations(min_count=min_derivation_count)

        for term, count in frequent:
            if isinstance(term, Compound) and _is_system_predicate(term.functor):
                continue

            try:
                new_fact = Fact(term)
                kb.add_fact(new_fact)
                ops.append(CompressionCandidate(
                    operation="lemma_cache",
                    original_clauses=[],
                    new_clauses=[new_fact]))
            except (ValueError, TypeError):
                continue  # Not a valid fact (e.g., contains variables)

        return ops
