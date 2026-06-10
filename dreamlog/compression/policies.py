"""Per-kind acceptance policies, lifted verbatim from the original ops."""
from typing import Optional

from ..knowledge import KnowledgeBase
from .proposal import Proposal


class Policy:
    operation = "generic"
    require_negative_delta = False

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
