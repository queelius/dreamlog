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
        self._rejections: list = []

    def dream(self, kb: KnowledgeBase, verify: bool = True) -> DreamSession:
        original_size = len(kb)
        if original_size == 0:
            return DreamSession(compressed=False, operations=[],
                                compression_ratio=1.0, verification=None)
        self._rejections = []

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

    def _eliminate_subsumed(self, kb: KnowledgeBase) -> List[CompressionCandidate]:
        """Operation A: Remove clauses subsumed by more general clauses."""
        from .compression import gate
        from .compression.generators import reduce as reduce_gen
        from .compression.policies import SubsumptionPolicy
        ops, policy = [], SubsumptionPolicy()
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
            kb, proposals, DerivabilityPolicy(max_calls=max_calls))
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
                              SuiteVerifyPolicy(suite, "generalization"),
                              self.min_group_size, self._rejections)

    def _invent_predicates(self, kb: KnowledgeBase,
                           suite: Optional['VerificationSuite'] = None
                           ) -> List[CompressionCandidate]:
        """Operation D: thin orchestrator - delegates to factor.run_invention."""
        from .compression import gate
        from .compression.generators import factor
        from .compression.policies import SuiteVerifyPolicy
        return factor.run_invention(kb, suite, gate.apply_proposal,
                                    SuiteVerifyPolicy(suite, "invention"),
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
                                     ExtractionPolicy(suite, "extraction"),
                                     self._rejections, max_rounds=max_rounds)

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
        from .compression import maintenance
        return maintenance.evict_dead_clauses(
            kb, min_query_threshold=min_query_threshold,
            seed_terms=seed_terms, seed_rules=seed_rules)

    def _frequency_score(self, kb: KnowledgeBase,
                         clauses: List[Union[Fact, Rule]]) -> float:
        from .compression import maintenance
        return maintenance.frequency_score(kb, clauses)

    # -- Operation H: Lemma caching --

    def _cache_lemmas(self, kb: KnowledgeBase,
                      min_derivation_count: int = 5
                      ) -> List[CompressionCandidate]:
        from .compression import maintenance
        return maintenance.cache_lemmas(kb, min_derivation_count=min_derivation_count)
