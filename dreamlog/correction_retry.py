"""
Correction-based retry mechanism for rule generation.

When an LLM generates an incorrect rule, this system provides feedback
and asks the LLM to try again with worked examples showing what went wrong.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from .knowledge import Rule, KnowledgeBase
from .llm_providers import LLMProvider
from .llm_judge import LLMJudge, JudgementResult
from .llm_response_parser import DreamLogResponseParser


@dataclass
class GenerationAttempt:
    """Record of a generation attempt"""
    rule: Rule
    judgement: Optional[JudgementResult]
    attempt_number: int


class CorrectionBasedRetry:
    """
    Retry mechanism that provides feedback about incorrect rules.

    Instead of just retrying with the same prompt, this system:
    1. Explains what was wrong with the previous attempt
    2. Shows the incorrect rule and why it's wrong
    3. Provides worked examples if available
    4. Asks the LLM to correct its mistake
    """

    def __init__(self,
                 provider: LLMProvider,
                 judge: LLMJudge,
                 parser: DreamLogResponseParser,
                 max_attempts: int = 3,
                 debug: bool = False):
        self.provider = provider
        self.judge = judge
        self.parser = parser
        self.max_attempts = max_attempts
        self.debug = debug

    def generate_with_correction(self,
                                functor: str,
                                knowledge_base: KnowledgeBase,
                                initial_prompt: str) -> Tuple[Optional[Rule], List[GenerationAttempt]]:
        """
        Generate a rule with correction-based retry.

        Args:
            functor: The functor to generate a rule for
            knowledge_base: Current knowledge base
            initial_prompt: The initial prompt to use

        Returns:
            Tuple of (final_rule, attempts_history)
            final_rule is None if all attempts failed
        """
        attempts: List[GenerationAttempt] = []
        current_prompt = initial_prompt

        for attempt_num in range(1, self.max_attempts + 1):
            if self.debug:
                print(f"\n[Correction Retry] Attempt {attempt_num}/{self.max_attempts}")
                print(f"[Correction Retry] Prompt:\n{current_prompt}\n")

            # Generate rule
            try:
                response = self.provider.generate(current_prompt)

                if self.debug:
                    print(f"[Correction Retry] Response:\n{response}\n")

                # Parse the rule from response
                rule = self.parser.parse_rule_from_response(response)

                if not rule:
                    if self.debug:
                        print(f"[Correction Retry] Failed to parse rule from response")
                    continue

                if self.debug:
                    print(f"[Correction Retry] Parsed rule: {rule}")

                # Verify the rule using LLM judge
                judgement = self.judge.verify_rule(rule, functor, knowledge_base)

                attempts.append(GenerationAttempt(
                    rule=rule,
                    judgement=judgement,
                    attempt_number=attempt_num
                ))

                if judgement.is_correct and judgement.confidence >= 0.7:
                    # Success!
                    if self.debug:
                        print(f"[Correction Retry] ✓ Rule verified as correct")
                    return rule, attempts

                # Rule is incorrect, prepare correction prompt for next attempt
                if self.debug:
                    print(f"[Correction Retry] ✗ Rule incorrect: {judgement.explanation}")

                if attempt_num < self.max_attempts:
                    current_prompt = self._build_correction_prompt(
                        functor=functor,
                        knowledge_base=knowledge_base,
                        previous_attempts=attempts,
                        initial_prompt=initial_prompt
                    )

            except Exception as e:
                if self.debug:
                    print(f"[Correction Retry] Exception during attempt {attempt_num}: {e}")

                attempts.append(GenerationAttempt(
                    rule=None,
                    judgement=None,
                    attempt_number=attempt_num
                ))

        # All attempts failed
        if self.debug:
            print(f"[Correction Retry] Failed after {self.max_attempts} attempts")

        return None, attempts

    def _build_correction_prompt(self,
                                functor: str,
                                knowledge_base: KnowledgeBase,
                                previous_attempts: List[GenerationAttempt],
                                initial_prompt: str) -> str:
        """Build a correction prompt showing what went wrong"""

        # Format KB context
        kb_lines = []
        if knowledge_base.facts:
            kb_lines.append("Facts:")
            for fact in knowledge_base.facts:
                kb_lines.append(f"  {fact}")

        if knowledge_base.rules:
            kb_lines.append("\nRules:")
            for rule in knowledge_base.rules:
                kb_lines.append(f"  {rule}")

        kb_context = "\n".join(kb_lines) if kb_lines else "Empty knowledge base"

        # Format previous attempts
        attempt_lines = []
        for att in previous_attempts:
            if att.rule and att.judgement is not None:
                attempt_lines.append(f"\nAttempt {att.attempt_number}:")
                attempt_lines.append(f"  Generated: {att.rule}")
                attempt_lines.append(f"  Result: {'✓ CORRECT' if att.judgement.is_correct else '✗ INCORRECT'}")
                attempt_lines.append(f"  Explanation: {att.judgement.explanation}")

                if att.judgement.suggested_correction:
                    attempt_lines.append(f"  Suggested: {att.judgement.suggested_correction}")

        attempts_text = "\n".join(attempt_lines)

        prompt = f"""You previously tried to generate a rule for '{functor}' but made a mistake.

KNOWLEDGE BASE:
{kb_context}

PREVIOUS ATTEMPTS:
{attempts_text}

TASK: Generate a CORRECT rule for '{functor}'.

Learn from your previous mistakes:
- Pay attention to the order of arguments
- Make sure the logic flows correctly
- Verify that the relationship is defined properly
- Use the predicates in the knowledge base correctly

IMPORTANT: Think step by step about the logic before generating the rule.

Example of common mistakes:
- ancestor(X, Y) :- parent(Z, Y), grandparent(X, Z).  [WRONG - backwards logic]
- ancestor(X, Y) :- parent(X, Y).                      [CORRECT - base case]
- ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).      [CORRECT - recursive case]

Generate the CORRECTED rule in JSON format:
[["rule", ["head", ...], [["body1", ...], ["body2", ...]]]]

Your response (JSON array only):"""

        return prompt

    def format_attempt_history(self, attempts: List[GenerationAttempt]) -> str:
        """Format attempt history for display"""
        lines = []

        for att in attempts:
            lines.append(f"\n--- Attempt {att.attempt_number} ---")

            if att.rule:
                lines.append(f"Generated: {att.rule}")
            else:
                lines.append("Generated: [failed to parse]")

            if att.judgement is not None:
                status = "✓ CORRECT" if att.judgement.is_correct else "✗ INCORRECT"
                lines.append(f"Status: {status}")
                lines.append(f"Confidence: {att.judgement.confidence:.2f}")
                lines.append(f"Explanation: {att.judgement.explanation}")

                if att.judgement.suggested_correction:
                    lines.append(f"Suggested: {att.judgement.suggested_correction}")
            else:
                lines.append("Status: [no judgement]")

        return "\n".join(lines)
