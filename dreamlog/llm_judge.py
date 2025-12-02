"""
LLM-as-judge verification for generated rules.

Uses an LLM to verify the correctness of generated rules by checking them
against the knowledge base and the original query.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .knowledge import Rule, Fact, KnowledgeBase
from .llm_providers import LLMProvider
from .llm_response_parser import DreamLogResponseParser
import json


@dataclass
class JudgementResult:
    """Result from LLM judge verification"""
    is_correct: bool
    confidence: float  # 0.0 to 1.0
    explanation: str
    suggested_correction: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_correct


class LLMJudge:
    """
    LLM-based verification of generated rules.

    Uses an LLM to assess whether a generated rule is logically correct
    given the knowledge base context and the query.
    """

    def __init__(self, provider: LLMProvider, debug: bool = False):
        self.provider = provider
        self.parser = DreamLogResponseParser()
        self.debug = debug

    def verify_rule(self,
                    rule: Rule,
                    query_functor: str,
                    knowledge_base: KnowledgeBase,
                    max_retries: int = 3) -> JudgementResult:
        """
        Verify that a generated rule is logically correct.

        Args:
            rule: The generated rule to verify
            query_functor: The functor that triggered generation
            knowledge_base: Current knowledge base context
            max_retries: Maximum number of retry attempts

        Returns:
            JudgementResult with verification outcome
        """
        prompt = self._build_verification_prompt(rule, query_functor, knowledge_base)

        if self.debug:
            print(f"[LLM Judge] Verification prompt:\n{prompt}\n")

        for attempt in range(max_retries):
            try:
                response = self.provider.generate(prompt)

                if self.debug:
                    print(f"[LLM Judge] Response:\n{response}\n")

                # Parse the JSON response
                result = self._parse_judgement(response)
                return result

            except Exception as e:
                if self.debug:
                    print(f"[LLM Judge] Attempt {attempt + 1} failed: {e}")

                if attempt == max_retries - 1:
                    # Final attempt failed, return uncertain result
                    return JudgementResult(
                        is_correct=False,
                        confidence=0.0,
                        explanation=f"Verification failed: {e}"
                    )

        # Should not reach here, but just in case
        return JudgementResult(
            is_correct=False,
            confidence=0.0,
            explanation="Verification failed after all retries"
        )

    def verify_fact(self,
                    fact: Fact,
                    query_functor: str,
                    knowledge_base: KnowledgeBase,
                    max_retries: int = 3) -> JudgementResult:
        """
        Verify that a generated fact is reasonable and correct.

        Args:
            fact: The generated fact to verify
            query_functor: The functor that triggered generation
            knowledge_base: Current knowledge base context
            max_retries: Maximum number of retry attempts

        Returns:
            JudgementResult with verification outcome
        """
        prompt = self._build_fact_verification_prompt(fact, query_functor, knowledge_base)

        if self.debug:
            print(f"[LLM Judge] Fact verification prompt:\n{prompt}\n")

        for attempt in range(max_retries):
            try:
                response = self.provider.generate(prompt)

                if self.debug:
                    print(f"[LLM Judge] Response:\n{response}\n")

                result = self._parse_judgement(response)
                return result

            except Exception as e:
                if self.debug:
                    print(f"[LLM Judge] Attempt {attempt + 1} failed: {e}")

                if attempt == max_retries - 1:
                    return JudgementResult(
                        is_correct=False,
                        confidence=0.0,
                        explanation=f"Fact verification failed: {e}"
                    )

        return JudgementResult(
            is_correct=False,
            confidence=0.0,
            explanation="Fact verification failed after all retries"
        )

    def _build_fact_verification_prompt(self,
                                        fact: Fact,
                                        query_functor: str,
                                        knowledge_base: KnowledgeBase) -> str:
        """Build the prompt for fact verification"""
        kb_context = self._format_kb_context(knowledge_base, query_functor)

        prompt = f"""You are verifying the correctness of a generated Prolog fact.

KNOWLEDGE BASE CONTEXT:
{kb_context}

GENERATED FACT TO VERIFY:
{fact.term}

QUERY THAT TRIGGERED THIS: {query_functor}

TASK: Determine if this fact is correct and reasonable.

Consider:
1. Is this fact consistent with common knowledge/world knowledge?
2. Does it make sense given the context of the knowledge base?
3. Is the predicate being used correctly (e.g., male(john) not male(mary))?
4. Would this fact help answer the original query?

Common sense checks:
- male/female predicates should match typical gender associations with names
- parent/child relationships should be plausible
- Geographic facts should be accurate
- Category memberships should be correct

Respond in JSON format:
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "explanation": "detailed explanation of your reasoning",
  "suggested_correction": "corrected fact if incorrect, or null if correct"
}}

Your response (JSON only):"""

        return prompt

    def _build_verification_prompt(self,
                                   rule: Rule,
                                   query_functor: str,
                                   knowledge_base: KnowledgeBase) -> str:
        """Build the prompt for LLM judge verification"""

        # Get relevant facts and rules from KB
        kb_context = self._format_kb_context(knowledge_base, query_functor)

        prompt = f"""You are verifying the logical correctness of a generated Prolog rule.

KNOWLEDGE BASE CONTEXT:
{kb_context}

GENERATED RULE TO VERIFY:
{rule}

QUERY FUNCTOR: {query_functor}

TASK: Determine if this rule is logically correct given the knowledge base.

Consider:
1. Does the rule correctly define the predicate?
2. Are the variables used correctly (not backwards)?
3. Does the rule produce correct results when evaluated?
4. Are the body predicates used in the right order?

Respond in JSON format:
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "explanation": "detailed explanation of your reasoning",
  "suggested_correction": "corrected rule if incorrect, or null if correct"
}}

Example of what NOT to do:
- ancestor(X, Y) :- parent(Z, Y), grandparent(X, Z).
  This is WRONG because it reverses the relationship.

Example of what TO do:
- ancestor(X, Y) :- parent(X, Y).
- ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
  These correctly define the transitive ancestor relationship.

Your response (JSON only):"""

        return prompt

    def _format_kb_context(self, kb: KnowledgeBase, query_functor: str) -> str:
        """Format relevant parts of the knowledge base for the prompt"""
        lines = []

        # Include all facts (usually small)
        if kb.facts:
            lines.append("Facts:")
            for fact in kb.facts:
                lines.append(f"  {fact}")

        # Include all rules (for context)
        if kb.rules:
            lines.append("\nRules:")
            for rule in kb.rules:
                lines.append(f"  {rule}")

        return "\n".join(lines) if lines else "Empty knowledge base"

    def _parse_judgement(self, response: str) -> JudgementResult:
        """Parse the LLM's judgement response"""
        # Extract JSON from response
        json_str = self.parser.extract_json_from_text(response)

        if not json_str:
            raise ValueError("No JSON found in LLM response")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        # Validate required fields
        if "is_correct" not in data:
            raise ValueError("Missing 'is_correct' field in judgement")
        if "confidence" not in data:
            raise ValueError("Missing 'confidence' field in judgement")
        if "explanation" not in data:
            raise ValueError("Missing 'explanation' field in judgement")

        return JudgementResult(
            is_correct=bool(data["is_correct"]),
            confidence=float(data["confidence"]),
            explanation=str(data["explanation"]),
            suggested_correction=data.get("suggested_correction")
        )


class VerificationPipeline:
    """
    Complete verification pipeline combining structural, semantic, and LLM checks.
    """

    def __init__(self,
                 knowledge_base: KnowledgeBase,
                 llm_judge: Optional[LLMJudge] = None,
                 debug: bool = False):
        from .rule_validator import RuleValidator

        self.kb = knowledge_base
        self.validator = RuleValidator(knowledge_base)
        self.llm_judge = llm_judge
        self.debug = debug

    def verify_rule(self,
                    rule: Rule,
                    query_functor: str,
                    use_llm_judge: bool = True) -> Dict[str, Any]:
        """
        Run complete verification pipeline.

        Args:
            rule: Rule to verify
            query_functor: Functor that triggered generation
            use_llm_judge: Whether to use LLM judge (only if available)

        Returns:
            Dictionary with verification results:
            {
                "valid": bool,
                "structural_result": ValidationResult,
                "semantic_result": ValidationResult,
                "llm_judgement": JudgementResult or None,
                "errors": List[str],
                "warnings": List[str]
            }
        """
        errors = []
        warnings = []

        # 1. Structural validation (fast, cheap)
        if self.debug:
            print(f"[Verification] Structural validation...")

        structural_result = self.validator.structural_validator.validate_rule(rule)

        if not structural_result.is_valid:
            errors.append(f"Structural: {structural_result.error_message}")
            return {
                "valid": False,
                "structural_result": structural_result,
                "semantic_result": None,
                "llm_judgement": None,
                "errors": errors,
                "warnings": warnings
            }

        if structural_result.warning_message:
            warnings.append(f"Structural: {structural_result.warning_message}")

        # 2. Semantic validation (medium cost)
        if self.debug:
            print(f"[Verification] Semantic validation...")

        semantic_result = self.validator.semantic_validator.validate_rule(rule)

        if not semantic_result.is_valid:
            errors.append(f"Semantic: {semantic_result.error_message}")
            return {
                "valid": False,
                "structural_result": structural_result,
                "semantic_result": semantic_result,
                "llm_judgement": None,
                "errors": errors,
                "warnings": warnings
            }

        if semantic_result.warning_message:
            warnings.append(f"Semantic: {semantic_result.warning_message}")

        # 3. LLM judge verification (expensive, only if enabled and available)
        llm_judgement = None
        if use_llm_judge and self.llm_judge:
            if self.debug:
                print(f"[Verification] LLM judge verification...")

            try:
                llm_judgement = self.llm_judge.verify_rule(rule, query_functor, self.kb)

                if not llm_judgement.is_correct:
                    errors.append(f"LLM Judge: {llm_judgement.explanation}")

                    # Add suggested correction if available
                    if llm_judgement.suggested_correction:
                        warnings.append(f"Suggested: {llm_judgement.suggested_correction}")

            except Exception as e:
                warnings.append(f"LLM judge failed: {e}")

        # Overall result
        is_valid = len(errors) == 0

        return {
            "valid": is_valid,
            "structural_result": structural_result,
            "semantic_result": semantic_result,
            "llm_judgement": llm_judgement,
            "errors": errors,
            "warnings": warnings
        }
