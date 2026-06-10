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
        return None
