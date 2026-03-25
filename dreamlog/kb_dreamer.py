"""
Knowledge Base Dreamer - Sleep phase symbolic compression.

Implements the sleep/dream cycle via three operations:
A. Subsumption elimination
B. Redundant fact pruning
C. Fact generalization with exceptions (added in Task 7)

All operations are purely symbolic (no LLM). Compression is guided by
Minimum Description Length: compress only when the result is shorter.
"""

from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from .terms import Term, Atom, Variable, Compound
from .knowledge import KnowledgeBase, Fact, Rule
from .unification import clause_subsumes, subsumes


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
        evaluator = evaluator_factory(kb)
        failures = []
        for q in self.positive_queries:
            if not evaluator.has_solution(q):
                failures.append(("false_negative", q))
        for q in self.negative_queries:
            if evaluator.has_solution(q):
                failures.append(("false_positive", q))
        return VerificationResult(
            passed=len(failures) == 0, failures=failures,
            positive_count=len(self.positive_queries),
            negative_count=len(self.negative_queries))


def build_verification_suite(kb: KnowledgeBase) -> VerificationSuite:
    """Build verification suite from current KB state."""
    positive = [fact.term for fact in kb.facts]

    atom_pool: Set[str] = set()
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
            novel = sorted(novel_values)[0]
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
    from .evaluator import PrologEvaluator

    ev = PrologEvaluator(kb)

    atom_values = set()
    for fact in kb.facts:
        if isinstance(fact.term, Compound):
            for arg in fact.term.args:
                if isinstance(arg, Atom):
                    atom_values.add(arg.value)

    if not atom_values:
        return

    atoms = sorted(atom_values)

    rule_preds = {}
    for rule in kb.rules:
        if isinstance(rule.head, Compound):
            key = (rule.head.functor, rule.head.arity)
            rule_preds[key] = True

    added = 0
    for (functor, arity) in rule_preds:
        if added >= max_queries:
            break
        if functor.startswith("_invented_") or functor.startswith("exception_"):
            continue
        if arity == 1:
            candidates = [Compound(functor, [Atom(a)]) for a in atoms[:10]]
        elif arity == 2:
            candidates = [Compound(functor, [Atom(a), Atom(b)])
                          for a in atoms[:8] for b in atoms[:8]]
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
                 shared_structure_threshold: float = 0.1):
        self.min_group_size = min_group_size
        self.shared_structure_threshold = shared_structure_threshold

    def dream(self, kb: KnowledgeBase, verify: bool = True) -> DreamSession:
        original_size = len(kb)
        if original_size == 0:
            return DreamSession(compressed=False, operations=[],
                                compression_ratio=1.0, verification=None)

        snapshot = kb.copy()
        suite = build_verification_suite(kb) if verify else None
        result = None
        ops: List[CompressionCandidate] = []

        ops.extend(self._eliminate_subsumed(kb))
        ops.extend(self._prune_redundant_facts(kb))
        ops.extend(self._generalize_facts(kb, suite))

        # Extend verification for rule-derived queries before Operation D
        if verify and suite:
            extend_verification_for_rules(suite, kb)
        ops.extend(self._invent_predicates(kb, suite))
        ops.extend(self._extract_body_patterns(kb, suite))

        if verify and suite:
            from .evaluator import PrologEvaluator
            result = suite.verify(kb, lambda k: PrologEvaluator(k))
            if not result.passed:
                kb.restore_from(snapshot)
                return DreamSession(compressed=False, operations=[],
                                    compression_ratio=1.0,
                                    verification=result)

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

    def _prune_redundant_facts(self, kb: KnowledgeBase) -> List[CompressionCandidate]:
        """Operation B: Remove facts derivable from remaining KB."""
        from .evaluator import PrologEvaluator

        ops = []
        facts = kb.facts

        # Phase 1: Find independently redundant facts
        redundant = []
        for fact in facts:
            kb.remove_fact_by_value(fact)
            ev = PrologEvaluator(kb)
            if ev.has_solution(fact.term):
                redundant.append(fact)
            kb.add_fact(fact)

        if not redundant:
            return ops

        # Phase 2: Batch remove
        for fact in redundant:
            kb.remove_fact_by_value(fact)

        # Phase 3: Verify all still derivable
        ev = PrologEvaluator(kb)
        still_ok = all(ev.has_solution(f.term) for f in redundant)

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
            ev = PrologEvaluator(kb)
            if ev.has_solution(fact.term):
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
        from .evaluator import PrologEvaluator

        ops = []

        # Group facts by functor/arity
        groups: dict = {}
        for fact in kb.facts:
            term = fact.term
            if isinstance(term, Compound):
                key = (term.functor, term.arity)
                groups.setdefault(key, []).append(fact)

        for (functor, arity), all_facts in groups.items():
            if functor.startswith("exception_"):
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
                        a.value if isinstance(a, Atom) else str(a)
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

                    for exc_val in sorted(exceptions):
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
                if f == functor or f.startswith("exception_"):
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
        from .evaluator import PrologEvaluator

        ops = []

        # Group rules by head functor
        pred_rules: Dict[str, List[Rule]] = {}
        for rule in kb.rules:
            if isinstance(rule.head, Compound):
                f = rule.head.functor
                if f.startswith("_invented_") or f.startswith("exception_"):
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

            invented_name = self._next_invented_name(kb)
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

    def _next_invented_name(self, kb: KnowledgeBase) -> str:
        """Find the next available _invented_N name."""
        max_n = -1
        for rule in kb.rules:
            if isinstance(rule.head, Compound) and rule.head.functor.startswith("_invented_"):
                try:
                    n = int(rule.head.functor.split("_invented_")[1])
                    max_n = max(max_n, n)
                except (ValueError, IndexError):
                    pass
        return f"_invented_{max_n + 1}"

    # ── Operation E: Rule body pattern extraction ──

    def _extract_body_patterns(self, kb: KnowledgeBase,
                               suite: Optional['VerificationSuite'] = None,
                               max_rounds: int = 10
                               ) -> List[CompressionCandidate]:
        """Operation E: Extract common contiguous sub-goal sequences from rule bodies."""
        from .evaluator import PrologEvaluator

        all_ops = []

        for _round in range(max_rounds):
            candidate = self._find_best_body_pattern(kb)
            if candidate is None:
                break

            subseq, occurrences = candidate

            # Compute interface variables for each occurrence
            extracted_name = self._next_extracted_name(kb)
            subseq_len = len(subseq)

            # Use first occurrence to determine interface vars
            # (all occurrences have same structure, so same interface)
            first_rule, first_start = occurrences[0]
            interface_vars = self._compute_interface_vars(
                first_rule, first_start, subseq_len)

            # Build extracted predicate
            # Use normalized variable names from the sub-sequence
            template_rule = first_rule
            template_body = list(template_rule.body)
            subseq_goals = template_body[first_start:first_start + subseq_len]

            # Map interface vars to positional names for the extracted predicate head
            extracted_head = Compound(extracted_name,
                                     [Variable(v) for v in interface_vars])
            extracted_rule = Rule(extracted_head, subseq_goals)

            # Verify before applying
            if suite is not None:
                test_kb = kb.copy()
                test_kb.add_rule(extracted_rule)
                for rule, start in occurrences:
                    body = list(rule.body)
                    # Map this occurrence's vars to the extracted predicate's interface
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
                    break

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

    def _find_best_body_pattern(self, kb: KnowledgeBase):
        """Find the best common contiguous sub-sequence across rule bodies."""
        # Collect rules to scan (skip generated predicates)
        rules = []
        for rule in kb.rules:
            if isinstance(rule.head, Compound):
                f = rule.head.functor
                if (f.startswith("_extracted_") or f.startswith("_invented_")
                        or f.startswith("exception_")):
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
            savings = (k - 1) * (n - 1) - 1
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
        return (subseq, occurrences)

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

    def _next_extracted_name(self, kb: KnowledgeBase) -> str:
        """Find the next available _extracted_N name."""
        max_n = -1
        for rule in kb.rules:
            if isinstance(rule.head, Compound) and rule.head.functor.startswith("_extracted_"):
                try:
                    n = int(rule.head.functor.split("_extracted_")[1])
                    max_n = max(max_n, n)
                except (ValueError, IndexError):
                    pass
        return f"_extracted_{max_n + 1}"
