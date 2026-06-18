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

import re
from typing import Any, Dict, List, Optional, Union, Set
from dataclasses import dataclass, field
from .terms import Term, Atom, Compound
from .knowledge import KnowledgeBase, Fact, Rule
from .evaluator import PrologEvaluator

# Re-exported: tests import _filter_cyclic_rules here; experiments import
# _collect_user_functors here.  _is_system_predicate and _strip_llm_noise are
# used directly in this module.  _next_generated_name is no longer needed here
# (moved to generators) but kept so downstream experiment scripts don't break.
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
    rejections: List[tuple] = field(default_factory=list)   # (kind, reason) pairs

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
                 min_closure_coverage: float = 0.5,
                 disable_op_c: bool = False,
                 dl_mode: str = "clauses",
                 decision_recorder=None):
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
        # Operation I open-world partial-closure coverage threshold (tau). Only
        # consulted when open_world=True AND discover_recursion=True: a binary R
        # that is a strict subset of B's transitive closure is accepted as a
        # partial closure iff |r_ext|/|closure(b_ext)| >= this. Default 0.5
        # (spec 2026-06-18). Ignored in closed-world mode (zero drift).
        self.min_closure_coverage = min_closure_coverage
        # Skip Operation C (fact generalization). Off by default; used by the
        # EX28 within-predicate LLM-only ablation condition.
        self.disable_op_c = disable_op_c
        # P3: description-length mode ("clauses" = P1 behavior, default;
        # "bits" = the prefix code in compression/dl.py) and an optional
        # decision recorder for the dl_decision_diff tool. Both default to
        # no-ops: zero drift.
        self.dl_mode = dl_mode
        self.decision_recorder = decision_recorder
        self._rejections: list = []

    def dream(self, kb: KnowledgeBase, verify: bool = True) -> DreamSession:
        self._rejections = []
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
        from .compression import maintenance
        maintenance.prune_suite_for_dead(suite, dead_ops)

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
                                    verification=result,
                                    rejections=list(self._rejections))

        # Restore wake-phase usage data (discard usage from verification queries)
        kb._usage_counts = wake_usage
        kb._derivation_counts = wake_derivation_counts
        kb._derivation_terms = wake_derivation_terms

        new_size = len(kb)
        return DreamSession(
            compressed=new_size < original_size, operations=ops,
            compression_ratio=new_size / original_size if original_size > 0 else 1.0,
            verification=result,
            rejections=list(self._rejections))

    def _configure_policy(self, policy):
        policy.dl_mode = self.dl_mode
        policy.recorder = self.decision_recorder
        return policy

    def _eliminate_subsumed(self, kb: KnowledgeBase) -> List[CompressionCandidate]:
        """Operation A: Remove clauses subsumed by more general clauses."""
        from .compression import gate
        from .compression.generators import reduce as reduce_gen
        from .compression.policies import SubsumptionPolicy
        ops, policy = [], self._configure_policy(SubsumptionPolicy())
        for p in reduce_gen.propose_subsumed_rules(kb):
            res = gate.apply_proposal(kb, p, policy)
            if isinstance(res, gate.Accepted):
                ops.append(res.candidate)
            else:
                self._rejections.append((p.kind, res.reason))
        for p in reduce_gen.propose_subsumed_facts(kb):   # after A1 commits
            res = gate.apply_proposal(kb, p, policy)
            if isinstance(res, gate.Accepted):
                ops.append(res.candidate)
            else:
                self._rejections.append((p.kind, res.reason))
        return ops

    def _prune_redundant_facts(self, kb: KnowledgeBase,
                               max_calls: int = 0) -> List[CompressionCandidate]:
        """Operation B: Remove facts derivable from remaining KB."""
        from .compression import gate
        from .compression.generators import reduce as reduce_gen
        from .compression.policies import DerivabilityPolicy
        proposals = reduce_gen.propose_redundant_facts(kb, max_calls=max_calls)
        accepted, rejected = gate.apply_batch_with_fallback(
            kb, proposals, self._configure_policy(DerivabilityPolicy(max_calls=max_calls)))
        self._rejections.extend((r.kind, r.reason) for r in rejected)
        return [a.candidate for a in accepted]

    def _generalize_facts(self, kb: KnowledgeBase,
                          suite: Optional['VerificationSuite'] = None
                          ) -> List[CompressionCandidate]:
        """Operation C: Generalize fact subgroups into rules with exceptions."""
        from .compression import gate
        from .compression.generators import generalize
        from .compression.policies import SuiteVerifyPolicy
        return generalize.run(kb, suite, gate.apply_proposal,
                              self._configure_policy(SuiteVerifyPolicy(suite, "generalization")),
                              self.min_group_size, self._rejections)

    def _invent_predicates(self, kb: KnowledgeBase,
                           suite: Optional['VerificationSuite'] = None
                           ) -> List[CompressionCandidate]:
        """Operation D: thin orchestrator - delegates to factor.run_invention."""
        from .compression import gate
        from .compression.generators import factor
        from .compression.policies import SuiteVerifyPolicy
        return factor.run_invention(kb, suite, gate.apply_proposal,
                                    self._configure_policy(SuiteVerifyPolicy(suite, "invention")),
                                    self._rejections)

    def _extract_body_patterns(self, kb: KnowledgeBase,
                               suite: Optional['VerificationSuite'] = None,
                               max_rounds: int = 10
                               ) -> List[CompressionCandidate]:
        """Operation E: thin orchestrator - delegates to factor.run_extraction."""
        from .compression import gate
        from .compression.generators import factor
        from .compression.policies import ExtractionPolicy
        return factor.run_extraction(kb, suite, gate.apply_proposal,
                                     self._configure_policy(ExtractionPolicy(suite, "extraction")),
                                     self._rejections, max_rounds=max_rounds)

    def _discover_recursion(self, kb: KnowledgeBase,
                            suite: Optional['VerificationSuite'] = None,
                            max_calls: int = 5000
                            ) -> List[CompressionCandidate]:
        """Operation I: thin orchestrator - delegates to closure.run.

        ClosurePolicy carries open_world; in closed-world mode it behaves
        identically to BoundedSuitePolicy (verify delegates to super) and
        closure.run takes only the exact path."""
        from .compression import gate
        from .compression.generators import closure
        from .compression.policies import ClosurePolicy
        policy = self._configure_policy(
            ClosurePolicy(suite, "recursion", max_calls, self.open_world))
        return closure.run(kb, suite, gate.apply_proposal, policy,
                           self.min_base_facts, self._rejections,
                           open_world=self.open_world,
                           min_closure_coverage=self.min_closure_coverage)

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
        """Thin delegator to the llm generator's pure prompt builder.

        Kept for the experiments/tests that reference it by name. Reads
        ``self.max_prompt_facts`` and ``kb`` only; mutates nothing.
        """
        from .compression.generators import llm as llm_gen
        return llm_gen.build_op_g_prompt(kb, self.max_prompt_facts)

    def _llm_propose(self, kb: KnowledgeBase) -> List[Rule]:
        """Thin delegator to the llm generator's pure proposal stage.

        Builds the prompt, calls the LLM, parses, then applies Phase 1
        structural validation and Phase 2 cyclic filtering. Mutates nothing
        (reads ``kb`` only). Referenced by experiments/ex28_probe and tests.
        """
        from .compression.generators import llm as llm_gen
        return llm_gen.propose_rules(kb, self.llm_client, self.max_prompt_facts)

    def _llm_compress(self, kb: KnowledgeBase,
                      suite: Optional['VerificationSuite'] = None
                      ) -> List[CompressionCandidate]:
        """Operation G: Ask LLM to propose cross-functor rules.

        Thin orchestrator: the pure proposal stage runs in the llm generator;
        Phase 3 (helper/main split) runs here; Phase 4+5 acceptance runs
        through the staged-combined gate with LlmPolicy.
        """
        from .compression import gate
        from .compression.generators import llm as llm_gen
        from .compression.policies import LlmPolicy
        from .compression.proposal import Proposal
        if not self.llm_client:
            return []
        parsed_rules = llm_gen.propose_rules(kb, self.llm_client,
                                             self.max_prompt_facts)
        if not parsed_rules:
            return []

        # Scale call budget with KB size - small KBs need ~500, but
        # transitive predicates in large KBs need proportionally more.
        max_calls = max(500, len(kb) * 10)

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

        context = [Proposal(kind="llm_compression", add=(r,))
                   for r in helper_rules]
        items = [Proposal(kind="llm_compression", add=(r,))
                 for r in main_rules]
        policy = self._configure_policy(LlmPolicy(suite, max_calls, self.open_world, kb))
        accepted, rejected = gate.apply_batch_staged_combined(
            kb, context, items, policy)
        self._rejections.extend((r.kind, r.reason) for r in rejected)
        return [a.candidate for a in accepted]

    # ── Operation F: Dead clause pruning ──

    def _prune_dead_clauses(self, kb: KnowledgeBase,
                            min_query_threshold: int = 10,
                            seed_terms: Optional[Set] = None,
                            seed_rules: Optional[Set] = None,
                            ) -> List[CompressionCandidate]:
        from .compression import maintenance
        return maintenance.evict_dead_clauses(
            kb, min_query_threshold=min_query_threshold,
            seed_terms=seed_terms, seed_rules=seed_rules)

    # -- Operation H: Lemma caching --

    def _cache_lemmas(self, kb: KnowledgeBase,
                      min_derivation_count: int = 5
                      ) -> List[CompressionCandidate]:
        from .compression import maintenance
        return maintenance.cache_lemmas(kb, min_derivation_count=min_derivation_count)
