"""
Knowledge Base Dreamer - Sleep phase knowledge optimization.

This module implements the "dreaming" component of DreamLog's wake-sleep architecture.
During sleep phases, the dreamer:
- Discovers patterns and redundancies in the knowledge base
- Compresses knowledge through abstraction
- Generalizes rules for broader applicability
- Verifies that behavior is preserved after optimization
"""

from typing import Protocol, List
from dataclasses import dataclass


class LLMProvider(Protocol):
    """Protocol for LLM providers used during dreaming."""
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class DreamVerification:
    """Results from verifying that behavior is preserved after optimization."""
    preserved: bool
    similarity_score: float
    improvements: List[str]


@dataclass
class DreamSession:
    """
    Results from a dream cycle.

    Tracks exploration paths, insights discovered, and verification results.
    """
    exploration_paths: int
    insights: List['DreamInsight']
    compression_ratio: float
    generalization_score: float
    verification: DreamVerification


@dataclass
class DreamInsight:
    """
    A single insight discovered during dreaming.

    Represents a discovered pattern, compression opportunity, or generalization.
    """
    type: str  # "abstraction", "compression", "generalization"
    description: str
    compression_ratio: float
    coverage_gain: float
    verified: bool


class KnowledgeBaseDreamer:
    """
    Implements knowledge optimization through "dreaming".

    The dreamer explores the knowledge base to discover:
    - Compression opportunities (redundant patterns)
    - Abstraction opportunities (common structures)
    - Generalization opportunities (broader applicability)
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the dreamer with an LLM provider.

        Args:
            llm_provider: An LLM provider for generating insights and optimizations
        """
        self.llm_provider = llm_provider

    def dream(self, kb, dream_cycles: int = 1, exploration_samples: int = 1,
              focus: str = "all", verify: bool = True) -> DreamSession:
        """
        Run a dream cycle on the knowledge base.

        Args:
            kb: The knowledge base to optimize
            dream_cycles: Number of dream cycles to run
            exploration_samples: Number of exploration samples per cycle
            focus: What to focus on ("all", "compression", "abstraction", "generalization")
            verify: Whether to verify behavior preservation

        Returns:
            A DreamSession with results from the dreaming process
        """
        insights = []

        # Detect patterns based on focus
        if focus in ["all", "compression"]:
            insights.extend(self._detect_compression_patterns(kb))

        if focus in ["all", "abstraction"]:
            insights.extend(self._detect_abstraction_patterns(kb))

        if focus in ["all", "generalization"]:
            insights.extend(self._detect_generalization_patterns(kb))

        # Calculate metrics
        compression_ratio = self._calculate_compression_ratio(kb, insights)
        generalization_score = self._calculate_generalization_score(insights)

        # Verify if requested
        verification = self._verify_optimizations(kb, insights) if verify else DreamVerification(
            preserved=True,
            similarity_score=1.0,
            improvements=[]
        )

        return DreamSession(
            exploration_paths=len(insights),
            insights=insights,
            compression_ratio=compression_ratio,
            generalization_score=generalization_score,
            verification=verification
        )

    def _detect_compression_patterns(self, kb) -> List[DreamInsight]:
        """Detect compression opportunities (redundant patterns)."""
        insights = []

        # Find rules with the same head but different bodies
        # This indicates potential compression through abstraction
        head_to_rules = {}
        for rule in kb._rules:
            head_str = str(rule.head)
            if head_str not in head_to_rules:
                head_to_rules[head_str] = []
            head_to_rules[head_str].append(rule)

        # Look for rules with the same head
        for head_str, rules in head_to_rules.items():
            if len(rules) > 1:
                # Found redundant pattern: multiple rules derive the same conclusion
                insight = DreamInsight(
                    type="compression",
                    description=f"Found {len(rules)} rules with head '{head_str}' - potential for abstraction",
                    compression_ratio=1.0 - (1.0 / len(rules)),
                    coverage_gain=len(rules) - 1.0,
                    verified=False
                )
                insights.append(insight)

        return insights

    def _detect_abstraction_patterns(self, kb) -> List[DreamInsight]:
        """Detect abstraction opportunities (common structures)."""
        # Placeholder for abstraction detection
        return []

    def _detect_generalization_patterns(self, kb) -> List[DreamInsight]:
        """Detect generalization opportunities (broader applicability)."""
        # Placeholder for generalization detection
        return []

    def _calculate_compression_ratio(self, kb, insights: List[DreamInsight]) -> float:
        """Calculate overall compression ratio."""
        if not insights:
            return 1.0

        # Average compression ratio from all insights
        total_compression = sum(i.compression_ratio for i in insights)
        return total_compression / len(insights)

    def _calculate_generalization_score(self, insights: List[DreamInsight]) -> float:
        """Calculate overall generalization score."""
        generalization_insights = [i for i in insights if i.type == "generalization"]
        if not generalization_insights:
            return 0.0

        return sum(i.coverage_gain for i in generalization_insights) / len(generalization_insights)

    def _verify_optimizations(self, kb, insights: List[DreamInsight]) -> DreamVerification:
        """Verify that optimizations preserve behavior."""
        # Placeholder for verification
        return DreamVerification(
            preserved=True,
            similarity_score=1.0,
            improvements=[]
        )
