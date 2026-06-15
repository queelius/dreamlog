"""Per-kind acceptance policies, lifted verbatim from the original ops."""
from typing import Optional

from ..knowledge import KnowledgeBase
from .proposal import Proposal


class Policy:
    operation = "generic"
    require_negative_delta = False
    dl_mode = "clauses"      # set per-dream by KnowledgeBaseDreamer
    recorder = None          # optional decision recorder (see gate._record)

    def pre_check(self, kb: KnowledgeBase, p: Proposal) -> Optional[str]:
        return None

    def verify(self, trial_kb: KnowledgeBase, p: Proposal) -> Optional[str]:
        return None

    def verify_batch(self, trial_kb, proposals) -> Optional[str]:
        """Return None to approve the full batch. WARNING: if a subclass does
        not override this, batch-level verification is skipped and every
        proposal commits WITHOUT per-item verify(); only use
        apply_batch_with_fallback with a policy that overrides this."""
        return None

    def verify_combined(self, trial_kb) -> Optional[str]:
        """Combined (whole-staged-set) verification for the staged-combined
        gate. Return None to approve. Defaults to no check; LlmPolicy
        overrides this with Operation G's Phase 5 verify."""
        return None


class SubsumptionPolicy(Policy):
    operation = "subsumption"
    require_negative_delta = True   # removal-only; the witness was the detection


class DerivabilityPolicy(Policy):
    """Operation B's acceptance: removed facts must remain derivable."""
    operation = "pruning"
    require_negative_delta = True

    def __init__(self, max_calls: int = 0):
        self.max_calls = max_calls

    def _ev(self, kb):
        from ..evaluator import PrologEvaluator
        return PrologEvaluator(kb, max_total_calls=self.max_calls)

    def verify(self, trial_kb, p):
        ev = self._ev(trial_kb)
        ok = all(ev.has_solution(c.term) for c in p.remove)
        return None if ok else "verify_failed"

    def verify_batch(self, trial_kb, proposals):
        ev = self._ev(trial_kb)
        ok = all(ev.has_solution(c.term)
                 for p in proposals for c in p.remove)
        return None if ok else "verify_failed"


class SuiteVerifyPolicy(Policy):
    """Verify a trial KB against the dream's suite with an UNBOUNDED evaluator
    (today's C/D/E behavior). suite=None means accept without verification."""
    require_negative_delta = True

    def __init__(self, suite, operation):
        from ..evaluator import PrologEvaluator
        self._PrologEvaluator = PrologEvaluator
        self.suite = suite
        self.operation = operation

    def verify(self, trial_kb, p):
        if self.suite is None:
            return None
        result = self.suite.verify(trial_kb, lambda k: self._PrologEvaluator(k))
        return None if result.passed else "verify_failed"


class ExtractionPolicy(SuiteVerifyPolicy):
    """Operation E: body-pattern extraction has delta = +1 BY DESIGN in
    clause-count terms (one extracted rule added, occurrences rewritten 1:1),
    so the strict-delta requirement is exempted in clauses mode. In bits mode
    the exemption ENDS: extraction must earn its definition overhead from the
    shared structure it removes (spec 2026-06-10 Section 4)."""

    @property
    def require_negative_delta(self):
        return self.dl_mode == "bits"


class BoundedSuitePolicy(SuiteVerifyPolicy):
    """Suite verification with a BOUNDED evaluator (Operation I: max_calls=5000)."""
    def __init__(self, suite, operation, max_calls):
        super().__init__(suite, operation)
        self.max_calls = max_calls

    def verify(self, trial_kb, p):
        if self.suite is None:
            return None
        result = self.suite.verify(
            trial_kb,
            lambda k: self._PrologEvaluator(k, max_total_calls=self.max_calls))
        return None if result.passed else "verify_failed"


class LlmPolicy(Policy):
    """Operation G's acceptance battery, lifted verbatim from _llm_compress
    Phase 4 (per-item) and Phase 5 (combined). Items are verified against
    kb + helpers + item; the kb fact snapshot for the derivability count and
    the false-positive check is taken at construction."""
    operation = "llm_compression"
    require_negative_delta = False   # add-only proposals; see spec 5.4

    def __init__(self, suite, max_calls, open_world, kb):
        from ..evaluator import PrologEvaluator
        self._PrologEvaluator = PrologEvaluator
        self.suite = suite
        self.max_calls = max_calls
        self.open_world = open_world
        self._kb_facts = list(kb.facts)   # snapshot, matches today's `for fact in kb.facts` against test_kb

    def verify(self, trial_kb, p):
        """Phase 4 acceptance battery for one main rule (lifted verbatim).

        ``trial_kb`` is kb + all helper rules + this rule. ``p.add[0]`` is the
        single main rule. Each failure mode rejects the item (scanning
        continues at the gate); the broad ``except Exception`` mirrors the
        original per-rule ``except Exception: continue``.
        """
        from ..terms import Compound
        try:
            rule = p.add[0]
            ev = self._PrologEvaluator(trial_kb, max_total_calls=self.max_calls)

            derivable_facts = []
            try:
                for fact in self._kb_facts:
                    if (isinstance(fact.term, Compound)
                            and fact.term.functor == rule.head.functor):
                        if ev.has_solution(fact.term):
                            derivable_facts.append(fact)
            except RecursionError:
                return "budget"

            if len(derivable_facts) < 2:
                return "policy"

            # Verify: adding rule must not break anything
            if self.suite is not None:
                def ev_factory(k, _mc=self.max_calls):
                    return self._PrologEvaluator(k, max_total_calls=_mc)
                try:
                    result = self.suite.verify(trial_kb, ev_factory)
                except RecursionError:
                    return "budget"
                if not result.passed:
                    return "verify_failed"

            # False-positive check: the rule must not derive ground
            # terms absent from the KB. Enumerate solutions for the
            # head pattern and reject if any is a novel (spurious) fact.
            #
            # Skipped in open-world mode (e.g., during holdout evaluation)
            # where the goal is precisely to recover absent facts.
            if not self.open_world:
                existing_terms = {f.term for f in self._kb_facts
                                  if isinstance(f.term, Compound)
                                  and f.term.functor == rule.head.functor}
                ev_fp = self._PrologEvaluator(trial_kb,
                                              max_total_calls=self.max_calls)
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
                    return "fp_check"

            return None
        except Exception:
            return "policy"

    def verify_combined(self, trial_kb):
        """Phase 5 combined verify. suite=None means no combined check ran
        today (the commit still happened), so return None. A RecursionError
        propagates to the gate's except clause -> 'budget' -> wipe, matching
        today's `except RecursionError: accepted_rules = []`."""
        if self.suite is None:
            return None
        def ev_factory(k, _mc=self.max_calls):
            return self._PrologEvaluator(k, max_total_calls=_mc)
        result = self.suite.verify(trial_kb, ev_factory)
        return None if result.passed else "verify_failed"
