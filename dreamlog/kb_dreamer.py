"""
Knowledge Base Dreamer - Sleep phase symbolic compression.

Implements the sleep/dream cycle via three operations:
A. Subsumption elimination
B. Redundant fact pruning
C. Fact generalization with exceptions (added in Task 7)

All operations are purely symbolic (no LLM). Compression is guided by
Minimum Description Length: compress only when the result is shorter.
"""

from typing import List, Optional, Union, Set
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
        """Operation C: Generalize fact groups into rules with exceptions."""
        from .anti_unification import anti_unify_many
        from .evaluator import PrologEvaluator

        ops = []
        groups = {}
        for fact in kb.facts:
            term = fact.term
            if isinstance(term, Compound):
                key = (term.functor, term.arity)
                groups.setdefault(key, []).append(fact)

        for (functor, arity), facts in groups.items():
            if functor.startswith("exception_"):
                continue
            if len(facts) < self.min_group_size:
                continue

            terms = [f.term for f in facts]
            au_result = anti_unify_many(terms)

            if au_result.shared_structure < self.shared_structure_threshold:
                continue
            if au_result.variables_introduced != 1:
                continue

            gen = au_result.generalized
            if not isinstance(gen, Compound):
                continue
            var_pos = None
            for i, arg in enumerate(gen.args):
                if isinstance(arg, Variable):
                    var_pos = i
                    break
            if var_pos is None:
                continue

            fact_values = set()
            for term in terms:
                arg = term.args[var_pos]
                if isinstance(arg, Atom):
                    fact_values.add(arg.value)

            guard = self._find_guard(kb, functor, fact_values)
            if guard is None:
                continue

            guard_functor, guard_values = guard
            exceptions = guard_values - fact_values
            cost_before = len(facts)
            cost_after = 1 + len(exceptions)
            if cost_after >= cost_before:
                continue

            exception_functor = f"exception_{functor}_{guard_functor}"
            new_clauses: List[Union[Fact, Rule]] = []

            rule_var = Variable("X")
            rule_args = list(gen.args)
            rule_args[var_pos] = rule_var
            rule_head = Compound(functor, rule_args)
            rule_body = [
                Compound(guard_functor, [rule_var]),
                Compound("not", [Compound(exception_functor, [rule_var])]),
            ]
            new_clauses.append(Rule(rule_head, rule_body))

            for exc_val in sorted(exceptions):
                new_clauses.append(
                    Fact(Compound(exception_functor, [Atom(exc_val)])))

            if suite is not None:
                test_kb = kb.copy()
                test_kb.replace_facts(facts, new_clauses)
                result = suite.verify(test_kb, lambda k: PrologEvaluator(k))
                if not result.passed:
                    continue

            kb.replace_facts(facts, new_clauses)
            ops.append(CompressionCandidate(
                operation="generalization",
                original_clauses=list(facts),
                new_clauses=list(new_clauses)))

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
