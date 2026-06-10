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
