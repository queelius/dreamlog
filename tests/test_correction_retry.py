"""
Comprehensive behavior-focused tests for correction-based retry mechanism.

Tests focus on the correction feedback loop behavior and retry logic.
"""

import pytest
import json
from dreamlog.correction_retry import CorrectionBasedRetry, GenerationAttempt
from dreamlog.llm_judge import LLMJudge, JudgementResult
from dreamlog.llm_response_parser import DreamLogResponseParser
from dreamlog.llm_providers import LLMProvider
from dreamlog.knowledge import KnowledgeBase, Rule, Fact
from dreamlog.terms import Atom, Variable, Compound
from dreamlog.prefix_parser import parse_s_expression


# ==================== Helper Functions ====================

def parse_rule_from_sexp(sexp_str: str) -> Rule:
    """
    Parse a rule from S-expression format.

    Example: "(test ?X) :- (foo ?X)"
    """
    # Split by :-
    if ":-" in sexp_str:
        parts = sexp_str.split(":-")
        head_str = parts[0].strip()
        body_str = parts[1].strip()

        # Parse head
        head = parse_s_expression(head_str)

        # Parse body - split by commas or spaces between parens
        body_terms = []
        current_term = ""
        paren_count = 0

        for char in body_str:
            if char == '(':
                paren_count += 1
                current_term += char
            elif char == ')':
                paren_count -= 1
                current_term += char
                if paren_count == 0 and current_term.strip():
                    body_terms.append(parse_s_expression(current_term.strip()))
                    current_term = ""
            elif paren_count > 0:
                current_term += char

        return Rule(head, body_terms)
    else:
        # Just a fact
        term = parse_s_expression(sexp_str)
        return Rule(term, [])


# ==================== Mock Classes ====================

class MockLLMProvider:
    """Mock LLM provider for generation testing"""

    def __init__(self, responses=None):
        if responses is None:
            responses = []
        elif isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.call_count = 0
        self.prompts = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.call_count += 1

        if not self.responses:
            raise ValueError("No more mocked responses available")

        return self.responses.pop(0)


class MockLLMJudge:
    """Mock LLM judge for deterministic verification"""

    def __init__(self, judgements=None):
        """
        Args:
            judgements: List of JudgementResults to return
        """
        if judgements is None:
            judgements = []
        self.judgements = list(judgements)
        self.call_count = 0

    def verify_rule(self, rule, query_functor, knowledge_base, max_retries=3):
        self.call_count += 1

        if not self.judgements:
            return JudgementResult(
                is_correct=True,
                confidence=1.0,
                explanation="Default acceptance"
            )

        return self.judgements.pop(0)


# ==================== Test GenerationAttempt ====================

class TestGenerationAttempt:
    """Test the GenerationAttempt record structure"""

    def test_stores_rule_and_judgement(self):
        """Should store all attempt information"""
        rule = parse_rule_from_sexp("(test ?X) :- (foo ?X)")
        judgement = JudgementResult(
            is_correct=True,
            confidence=0.9,
            explanation="Good"
        )

        attempt = GenerationAttempt(
            rule=rule,
            judgement=judgement,
            attempt_number=1
        )

        assert attempt.rule == rule
        assert attempt.judgement == judgement
        assert attempt.attempt_number == 1

    def test_allows_none_for_failed_attempts(self):
        """Should allow None for rule/judgement when parsing fails"""
        attempt = GenerationAttempt(
            rule=None,
            judgement=None,
            attempt_number=2
        )

        assert attempt.rule is None
        assert attempt.judgement is None
        assert attempt.attempt_number == 2


# ==================== Test Successful Generation ====================

class TestSuccessfulGeneration:
    """Test behavior when generation succeeds on first or later attempt"""

    def test_succeeds_on_first_attempt(self):
        """Should return rule immediately if first generation is correct"""
        # Given: Provider that generates correct rule
        provider = MockLLMProvider(
            '["rule", ["ancestor", "?X", "?Y"], [["parent", "?X", "?Y"]]]'
        )

        # Judge accepts it
        judge = MockLLMJudge([
            JudgementResult(is_correct=True, confidence=0.95, explanation="Perfect")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Generating rule
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        rule, attempts = retry.generate_with_correction(
            functor="ancestor",
            knowledge_base=kb,
            initial_prompt="Generate ancestor rule"
        )

        # Then: Should succeed on first attempt
        assert rule is not None
        assert len(attempts) == 1
        assert attempts[0].judgement.is_correct is True
        assert provider.call_count == 1
        assert judge.call_count == 1

    def test_succeeds_after_correction(self):
        """Should retry with correction feedback and eventually succeed"""
        # Given: Provider that fails then succeeds
        provider = MockLLMProvider([
            '["rule", ["ancestor", "?X", "?Y"], [["parent", "?Y", "?X"]]]',  # Wrong
            '["rule", ["ancestor", "?X", "?Y"], [["parent", "?X", "?Y"]]]'   # Fixed
        ])

        # Judge rejects first, accepts second
        judge = MockLLMJudge([
            JudgementResult(
                is_correct=False,
                confidence=0.8,
                explanation="Variables backwards",
                suggested_correction="Swap X and Y"
            ),
            JudgementResult(
                is_correct=True,
                confidence=0.95,
                explanation="Now correct"
            )
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Generating with correction
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="ancestor",
            knowledge_base=kb,
            initial_prompt="Generate ancestor rule"
        )

        # Then: Should succeed on second attempt
        assert rule is not None
        assert len(attempts) == 2
        assert attempts[0].judgement.is_correct is False
        assert attempts[1].judgement.is_correct is True
        assert provider.call_count == 2

    def test_respects_confidence_threshold(self):
        """Should only accept rules with confidence >= 0.7"""
        # Given: Provider generates rule with low confidence acceptance
        provider = MockLLMProvider([
            '["rule", ["test", "?X"], [["foo", "?X"]]]',  # Low confidence
            '["rule", ["test", "?X"], [["bar", "?X"]]]'   # High confidence
        ])

        # Judge gives low then high confidence
        judge = MockLLMJudge([
            JudgementResult(
                is_correct=True,
                confidence=0.6,  # Below threshold
                explanation="Maybe OK but uncertain"
            ),
            JudgementResult(
                is_correct=True,
                confidence=0.9,  # Above threshold
                explanation="Definitely correct"
            )
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Generating
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should retry until high confidence
        assert rule is not None
        assert len(attempts) == 2
        assert attempts[1].judgement.confidence >= 0.7


# ==================== Test Failure Scenarios ====================

class TestFailureScenarios:
    """Test behavior when all attempts fail"""

    def test_returns_none_after_max_attempts(self):
        """Should return None if all attempts fail"""
        # Given: Provider that always generates incorrect rules
        provider = MockLLMProvider([
            '["rule", ["test", "?X"], [["wrong", "?X"]]]',
            '["rule", ["test", "?X"], [["still_wrong", "?X"]]]',
            '["rule", ["test", "?X"], [["nope", "?X"]]]'
        ])

        # Judge rejects everything
        judge = MockLLMJudge([
            JudgementResult(is_correct=False, confidence=0.8, explanation="Wrong 1"),
            JudgementResult(is_correct=False, confidence=0.8, explanation="Wrong 2"),
            JudgementResult(is_correct=False, confidence=0.8, explanation="Wrong 3")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: All attempts fail
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should return None with full attempt history
        assert rule is None
        assert len(attempts) == 3
        assert all(not att.judgement.is_correct for att in attempts)

    def test_handles_parse_failures(self):
        """Should handle cases where LLM response cannot be parsed"""
        # Given: Provider returns unparseable responses
        provider = MockLLMProvider([
            "This is not valid JSON",
            "Still not JSON",
            '["rule", ["test", "?X"], [["foo", "?X"]]]'  # Finally valid
        ])

        # Judge accepts the valid one
        judge = MockLLMJudge([
            JudgementResult(is_correct=True, confidence=0.9, explanation="OK")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Generating (with parse failures)
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should eventually succeed, recording parse failures
        assert rule is not None
        assert len(attempts) == 1  # Only successful parse is recorded
        # Parse failures don't create attempts

    def test_handles_provider_exceptions(self):
        """Should handle exceptions from provider gracefully"""
        # Given: Provider that raises exception
        class FailingProvider:
            def generate(self, prompt, **kwargs):
                raise RuntimeError("API failed")

        provider = FailingProvider()
        judge = MockLLMJudge()
        parser = DreamLogResponseParser()

        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=2)

        # When: Provider fails
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should return None with failure recorded
        assert rule is None
        # Exceptions create attempts with None rule/judgement


# ==================== Test Correction Prompt Building ====================

class TestCorrectionPromptBuilding:
    """Test that correction prompts include proper feedback"""

    def test_correction_prompt_includes_previous_attempts(self):
        """Correction prompt should explain what was wrong before"""
        # Given: System that will generate correction prompt
        provider = MockLLMProvider([
            '["rule", ["test", "?X"], [["wrong", "?X"]]]',  # First attempt
            '["rule", ["test", "?X"], [["right", "?X"]]]'   # Second attempt
        ])

        judge = MockLLMJudge([
            JudgementResult(
                is_correct=False,
                confidence=0.9,
                explanation="The predicate 'wrong' doesn't exist",
                suggested_correction="Use 'foo' instead"
            ),
            JudgementResult(is_correct=True, confidence=0.95, explanation="OK")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Retrying with correction
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Second prompt should include feedback about first attempt
        second_prompt = provider.prompts[1]
        assert "previous" in second_prompt.lower() or "attempt" in second_prompt.lower()
        assert "wrong" in second_prompt.lower()  # Should mention the error

    def test_correction_prompt_includes_kb_context(self):
        """Correction prompt should include knowledge base facts/rules"""
        # Given: KB with content
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        provider = MockLLMProvider([
            '["rule", ["ancestor", "?X", "?Y"], [["wrong", "?X", "?Y"]]]',
            '["rule", ["ancestor", "?X", "?Y"], [["parent", "?X", "?Y"]]]'
        ])

        judge = MockLLMJudge([
            JudgementResult(is_correct=False, confidence=0.9, explanation="Bad"),
            JudgementResult(is_correct=True, confidence=0.95, explanation="OK")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Generating with correction
        rule, attempts = retry.generate_with_correction(
            functor="ancestor",
            knowledge_base=kb,
            initial_prompt="Generate ancestor rule"
        )

        # Then: Correction prompt should include KB context
        if len(provider.prompts) > 1:
            correction_prompt = provider.prompts[1]
            assert "parent" in correction_prompt.lower()
            assert "john" in correction_prompt.lower() or "mary" in correction_prompt.lower()

    def test_correction_prompt_includes_suggested_correction(self):
        """Should include judge's suggested correction in retry prompt"""
        provider = MockLLMProvider([
            '["rule", ["test", "?X"], [["bad", "?X"]]]',
            '["rule", ["test", "?X"], [["good", "?X"]]]'
        ])

        judge = MockLLMJudge([
            JudgementResult(
                is_correct=False,
                confidence=0.9,
                explanation="Use 'good' not 'bad'",
                suggested_correction="Replace bad with good"
            ),
            JudgementResult(is_correct=True, confidence=0.95, explanation="OK")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Retrying
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Correction prompt should include suggestion
        if len(provider.prompts) > 1:
            correction_prompt = provider.prompts[1]
            assert "suggested" in correction_prompt.lower() or "replace" in correction_prompt.lower()


# ==================== Test Attempt History ====================

class TestAttemptHistory:
    """Test tracking and formatting of attempt history"""

    def test_records_complete_attempt_history(self):
        """Should record all attempts with their outcomes"""
        provider = MockLLMProvider([
            '["rule", ["test", "?X"], [["v1", "?X"]]]',
            '["rule", ["test", "?X"], [["v2", "?X"]]]',
            '["rule", ["test", "?X"], [["v3", "?X"]]]'
        ])

        judge = MockLLMJudge([
            JudgementResult(is_correct=False, confidence=0.5, explanation="Bad v1"),
            JudgementResult(is_correct=False, confidence=0.6, explanation="Bad v2"),
            JudgementResult(is_correct=True, confidence=0.95, explanation="Good v3")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=3)

        # When: Multiple attempts
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should have complete history
        assert len(attempts) == 3
        assert attempts[0].attempt_number == 1
        assert attempts[1].attempt_number == 2
        assert attempts[2].attempt_number == 3
        assert all(att.rule is not None for att in attempts)
        assert all(att.judgement is not None for att in attempts)

    def test_format_attempt_history_is_readable(self):
        """Should produce human-readable attempt history"""
        # Given: Some attempts
        rule1 = parse_rule_from_sexp("(test ?X) :- (foo ?X)")
        rule2 = parse_rule_from_sexp("(test ?X) :- (bar ?X)")

        attempts = [
            GenerationAttempt(
                rule=rule1,
                judgement=JudgementResult(
                    is_correct=False,
                    confidence=0.7,
                    explanation="Wrong predicate",
                    suggested_correction="Use bar"
                ),
                attempt_number=1
            ),
            GenerationAttempt(
                rule=rule2,
                judgement=JudgementResult(
                    is_correct=True,
                    confidence=0.95,
                    explanation="Correct"
                ),
                attempt_number=2
            )
        ]

        provider = MockLLMProvider()
        judge = MockLLMJudge()
        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser)

        # When: Formatting history
        formatted = retry.format_attempt_history(attempts)

        # Then: Should contain key information
        assert "Attempt 1" in formatted
        assert "Attempt 2" in formatted
        assert "INCORRECT" in formatted or "✗" in formatted
        assert "CORRECT" in formatted or "✓" in formatted
        assert "0.95" in formatted  # Confidence

    def test_format_handles_failed_attempts(self):
        """Should format attempts that failed to parse"""
        attempts = [
            GenerationAttempt(
                rule=None,  # Failed to parse
                judgement=None,
                attempt_number=1
            )
        ]

        provider = MockLLMProvider()
        judge = MockLLMJudge()
        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser)

        # When: Formatting with failures
        formatted = retry.format_attempt_history(attempts)

        # Then: Should handle None gracefully
        assert "Attempt 1" in formatted
        assert "failed to parse" in formatted.lower() or "no judgement" in formatted.lower()


# ==================== Test Debug Mode ====================

class TestDebugMode:
    """Test debug output behavior"""

    def test_debug_mode_does_not_crash(self):
        """Debug mode should not cause failures"""
        provider = MockLLMProvider(
            '["rule", ["test", "?X"], [["foo", "?X"]]]'
        )

        judge = MockLLMJudge([
            JudgementResult(is_correct=True, confidence=0.9, explanation="OK")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, debug=True)

        # When: Running with debug enabled
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should complete successfully
        assert rule is not None


# ==================== Test Max Attempts Configuration ====================

class TestMaxAttemptsConfiguration:
    """Test that max_attempts is respected"""

    def test_respects_max_attempts_limit(self):
        """Should not exceed configured max attempts"""
        # Given: Provider that always fails
        provider = MockLLMProvider([
            '["rule", ["test", "?X"], [["bad", "?X"]]]',
            '["rule", ["test", "?X"], [["bad", "?X"]]]'
        ])

        judge = MockLLMJudge([
            JudgementResult(is_correct=False, confidence=0.9, explanation="Bad"),
            JudgementResult(is_correct=False, confidence=0.9, explanation="Bad")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=2)

        # When: Generating with limit of 2
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should stop after 2 attempts
        assert rule is None
        assert len(attempts) == 2
        assert provider.call_count == 2

    def test_single_attempt_mode(self):
        """Should work with max_attempts=1 (no retry)"""
        provider = MockLLMProvider(
            '["rule", ["test", "?X"], [["foo", "?X"]]]'
        )

        judge = MockLLMJudge([
            JudgementResult(is_correct=True, confidence=0.9, explanation="OK")
        ])

        parser = DreamLogResponseParser()
        retry = CorrectionBasedRetry(provider, judge, parser, max_attempts=1)

        # When: Single attempt only
        kb = KnowledgeBase()
        rule, attempts = retry.generate_with_correction(
            functor="test",
            knowledge_base=kb,
            initial_prompt="Generate test rule"
        )

        # Then: Should succeed or fail in one attempt
        assert len(attempts) == 1
        assert provider.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
