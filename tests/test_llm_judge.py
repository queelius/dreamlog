"""
Comprehensive behavior-focused tests for LLM Judge verification system.

Tests focus on observable behavior and contracts, not implementation details.
"""

import pytest
import json
from dreamlog.llm_judge import LLMJudge, JudgementResult, VerificationPipeline
from dreamlog.llm_providers import LLMProvider
from dreamlog.knowledge import KnowledgeBase, Rule, Fact
from dreamlog.terms import Atom, Variable, Compound


# ==================== Helper Functions ====================

def make_rule(head_functor, head_args, body_list):
    """
    Helper to create rules more easily.

    Example:
        make_rule("ancestor", ["X", "Y"], [
            ["parent", "X", "Z"],
            ["ancestor", "Z", "Y"]
        ])
    """
    def make_term(item):
        if isinstance(item, str):
            # Uppercase first letter = Variable
            if item[0].isupper() or item.startswith("?"):
                name = item.lstrip("?")
                return Variable(name)
            else:
                return Atom(item)
        elif isinstance(item, list):
            functor = item[0]
            args = [make_term(arg) for arg in item[1:]]
            return Compound(functor, args)
        else:
            return Atom(item)

    head = Compound(head_functor, [make_term(arg) for arg in head_args])
    body = [make_term(goal) for goal in body_list]
    return Rule(head, body)


# ==================== Mock Providers ====================

class MockLLMProvider:
    """Mock LLM provider for deterministic testing"""

    def __init__(self, responses=None):
        """
        Args:
            responses: List of responses or single response to return
        """
        if responses is None:
            responses = []
        elif isinstance(responses, str):
            responses = [responses]
        self.responses = list(responses)
        self.response_index = 0
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt: str, **kwargs) -> str:
        """Return next mocked response"""
        self.last_prompt = prompt
        self.call_count += 1

        if not self.responses:
            raise ValueError("No more mocked responses available")

        # Get response at current index, or last one if we've run out
        idx = min(self.response_index, len(self.responses) - 1)
        response = self.responses[idx]
        self.response_index += 1
        return response


# ==================== Test JudgementResult ====================

class TestJudgementResult:
    """Test the JudgementResult dataclass behavior"""

    def test_correct_judgement_is_truthy(self):
        """A correct judgement should evaluate to True in boolean context"""
        result = JudgementResult(
            is_correct=True,
            confidence=0.9,
            explanation="Looks good"
        )

        assert bool(result) is True
        assert result  # Should work in if statements

    def test_incorrect_judgement_is_falsy(self):
        """An incorrect judgement should evaluate to False in boolean context"""
        result = JudgementResult(
            is_correct=False,
            confidence=0.8,
            explanation="Wrong logic"
        )

        assert bool(result) is False
        assert not result

    def test_suggested_correction_is_optional(self):
        """suggested_correction field should be optional"""
        # Without correction
        result1 = JudgementResult(
            is_correct=True,
            confidence=1.0,
            explanation="Perfect"
        )
        assert result1.suggested_correction is None

        # With correction
        result2 = JudgementResult(
            is_correct=False,
            confidence=0.7,
            explanation="Wrong",
            suggested_correction="Use this instead"
        )
        assert result2.suggested_correction == "Use this instead"


# ==================== Test LLMJudge Core Behavior ====================

class TestLLMJudgeVerification:
    """Test LLMJudge's rule verification behavior"""

    def test_verify_rule_with_correct_judgement(self):
        """When LLM says rule is correct, should return positive judgement"""
        # Given: A mock provider that returns a correct judgement
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.95,
            "explanation": "The rule correctly defines the transitive relationship",
            "suggested_correction": None
        }))

        judge = LLMJudge(provider)

        # When: Verifying a rule
        rule = make_rule("ancestor", ["X", "Y"], [
            ["parent", "X", "Z"],
            ["ancestor", "Z", "Y"]
        ])
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        result = judge.verify_rule(rule, "ancestor", kb)

        # Then: Should return correct judgement
        assert result.is_correct is True
        assert result.confidence == 0.95
        assert "transitive" in result.explanation
        assert result.suggested_correction is None

    def test_verify_rule_with_incorrect_judgement(self):
        """When LLM says rule is incorrect, should return negative judgement with suggestion"""
        # Given: A mock provider that returns an incorrect judgement
        provider = MockLLMProvider(json.dumps({
            "is_correct": False,
            "confidence": 0.85,
            "explanation": "Variables are in wrong order",
            "suggested_correction": "(ancestor ?X ?Y) :- (parent ?X ?Z) (ancestor ?Z ?Y)"
        }))

        judge = LLMJudge(provider)

        # When: Verifying a potentially wrong rule
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "Z", "Y"], ["ancestor", "X", "Z"]])
        kb = KnowledgeBase()

        result = judge.verify_rule(rule, "ancestor", kb)

        # Then: Should return incorrect judgement with correction
        assert result.is_correct is False
        assert result.confidence == 0.85
        assert "wrong order" in result.explanation
        assert result.suggested_correction is not None

    def test_verify_rule_retries_on_parse_failure(self):
        """When response parsing fails, should retry up to max_retries times"""
        # Given: A provider that fails twice then succeeds
        provider = MockLLMProvider([
            "Invalid JSON {",  # First attempt - invalid
            "Not JSON at all",  # Second attempt - invalid
            json.dumps({  # Third attempt - valid
                "is_correct": True,
                "confidence": 0.9,
                "explanation": "Third time's the charm"
            })
        ])

        judge = LLMJudge(provider)

        # When: Verifying a rule
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])
        kb = KnowledgeBase()

        result = judge.verify_rule(rule, "ancestor", kb, max_retries=3)

        # Then: Should eventually succeed
        assert result.is_correct is True
        assert provider.call_count == 3  # Tried 3 times

    def test_verify_rule_returns_failure_after_max_retries(self):
        """When all retries fail, should return failure judgement"""
        # Given: A provider that always fails
        provider = MockLLMProvider([
            "Bad JSON",
            "Still bad",
            "Nope"
        ])

        judge = LLMJudge(provider)

        # When: Verifying with limited retries
        rule = make_rule("test", ["X"], [["foo", "X"]])
        kb = KnowledgeBase()

        result = judge.verify_rule(rule, "test", kb, max_retries=3)

        # Then: Should return failure result
        assert result.is_correct is False
        assert result.confidence == 0.0
        assert "Verification failed" in result.explanation


# ==================== Test Response Parsing ====================

class TestLLMJudgeResponseParsing:
    """Test how LLMJudge parses various response formats"""

    def test_parse_json_with_all_fields(self):
        """Should successfully parse valid JSON with all required fields"""
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.88,
            "explanation": "Well-formed rule",
            "suggested_correction": None
        }))

        judge = LLMJudge(provider)
        rule = make_rule("test", ["X"], [["foo", "X"]])
        kb = KnowledgeBase()

        result = judge.verify_rule(rule, "test", kb)

        assert result.is_correct is True
        assert result.confidence == 0.88
        assert result.explanation == "Well-formed rule"

    def test_parse_json_embedded_in_text(self):
        """Should extract JSON even when surrounded by text"""
        provider = MockLLMProvider("""
        Let me analyze this rule...

        ```json
        {
            "is_correct": false,
            "confidence": 0.6,
            "explanation": "Found an issue"
        }
        ```

        Hope this helps!
        """)

        judge = LLMJudge(provider)
        rule = make_rule("test", ["X"], [["foo", "X"]])
        kb = KnowledgeBase()

        result = judge.verify_rule(rule, "test", kb)

        assert result.is_correct is False
        assert result.confidence == 0.6

    def test_reject_json_missing_required_fields(self):
        """Should fail when JSON is missing required fields"""
        provider = MockLLMProvider(json.dumps({
            "is_correct": True
            # Missing confidence and explanation
        }))

        judge = LLMJudge(provider)
        rule = make_rule("test", ["X"], [["foo", "X"]])
        kb = KnowledgeBase()

        result = judge.verify_rule(rule, "test", kb, max_retries=1)

        # Should return failure result
        assert result.is_correct is False
        assert result.confidence == 0.0


# ==================== Test Knowledge Base Context ====================

class TestKnowledgeBaseContextFormatting:
    """Test that judge includes appropriate KB context in prompts"""

    def test_prompt_includes_facts_from_kb(self):
        """Verification prompt should include relevant facts"""
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.9,
            "explanation": "OK"
        }))

        judge = LLMJudge(provider)

        # Given: KB with facts
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))
        kb.add_fact(Fact(Compound("parent", [Atom("mary"), Atom("sue")])))

        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])

        # When: Verifying rule
        judge.verify_rule(rule, "ancestor", kb)

        # Then: Prompt should contain the facts
        prompt = provider.last_prompt
        assert "parent" in prompt.lower()
        assert "john" in prompt.lower() or "mary" in prompt.lower()

    def test_prompt_includes_existing_rules(self):
        """Verification prompt should include existing rules for context"""
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.9,
            "explanation": "OK"
        }))

        judge = LLMJudge(provider)

        # Given: KB with an existing rule
        kb = KnowledgeBase()
        existing_rule = make_rule("grandparent", ["X", "Z"], [["parent", "X", "Y"], ["parent", "Y", "Z"]])
        kb.add_rule(existing_rule)

        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])

        # When: Verifying new rule
        judge.verify_rule(rule, "ancestor", kb)

        # Then: Prompt should contain existing rules
        prompt = provider.last_prompt
        assert "grandparent" in prompt.lower()

    def test_prompt_includes_query_functor(self):
        """Verification prompt should specify what functor is being defined"""
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.9,
            "explanation": "OK"
        }))

        judge = LLMJudge(provider)

        kb = KnowledgeBase()
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])

        # When: Verifying for specific functor
        judge.verify_rule(rule, "ancestor", kb)

        # Then: Functor should be in prompt
        prompt = provider.last_prompt
        assert "ancestor" in prompt.lower()


# ==================== Test VerificationPipeline ====================

class TestVerificationPipeline:
    """Test the complete verification pipeline behavior"""

    def test_pipeline_runs_structural_validation_first(self):
        """Structural validation should run before semantic or LLM checks"""
        # Given: A structurally invalid rule (head is a variable)
        kb = KnowledgeBase()
        pipeline = VerificationPipeline(kb, debug=False)

        # Create a rule with variable as head (structurally invalid)
        invalid_rule = Rule(
            head=Variable("X"),  # Invalid - head cannot be a variable
            body=[Compound("foo", [Variable("X")])]
        )

        # When: Running verification
        result = pipeline.verify_rule(invalid_rule, "test", use_llm_judge=False)

        # Then: Should fail at structural level
        assert result["valid"] is False
        assert result["structural_result"] is not None
        assert not result["structural_result"].is_valid
        assert result["semantic_result"] is None  # Should not reach semantic
        assert result["llm_judgement"] is None  # Should not reach LLM

    def test_pipeline_skips_llm_when_not_available(self):
        """Should complete validation without LLM if judge not provided"""
        # Given: Pipeline without LLM judge
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        pipeline = VerificationPipeline(kb, llm_judge=None, debug=False)

        # When: Verifying a valid rule
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])
        result = pipeline.verify_rule(rule, "ancestor", use_llm_judge=True)

        # Then: Should succeed without LLM judgement
        assert result["valid"] is True
        assert result["llm_judgement"] is None

    def test_pipeline_includes_llm_judgement_when_available(self):
        """Should run LLM verification when judge is provided"""
        # Given: Pipeline with LLM judge
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.95,
            "explanation": "Rule is correct"
        }))

        llm_judge = LLMJudge(provider)
        pipeline = VerificationPipeline(kb, llm_judge=llm_judge, debug=False)

        # When: Verifying with LLM enabled
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])
        result = pipeline.verify_rule(rule, "ancestor", use_llm_judge=True)

        # Then: Should include LLM judgement
        assert result["valid"] is True
        assert result["llm_judgement"] is not None
        assert result["llm_judgement"].is_correct is True

    def test_pipeline_fails_when_llm_rejects_rule(self):
        """Should mark overall result as invalid if LLM judge rejects"""
        # Given: Pipeline with judge that rejects rule
        kb = KnowledgeBase()

        provider = MockLLMProvider(json.dumps({
            "is_correct": False,
            "confidence": 0.9,
            "explanation": "Logic is backwards",
            "suggested_correction": "Fix the order"
        }))

        llm_judge = LLMJudge(provider)
        pipeline = VerificationPipeline(kb, llm_judge=llm_judge, debug=False)

        # When: Verifying a rule that LLM rejects
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "Y", "X"]])  # Backwards
        result = pipeline.verify_rule(rule, "ancestor", use_llm_judge=True)

        # Then: Overall result should be invalid
        assert result["valid"] is False
        assert "Logic is backwards" in result["errors"][0]

    def test_pipeline_collects_warnings(self):
        """Should collect warnings from all validation stages"""
        # Given: A rule that might generate warnings
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.7,  # Lower confidence might generate warning
            "explanation": "Might be OK"
        }))

        llm_judge = LLMJudge(provider)
        pipeline = VerificationPipeline(kb, llm_judge=llm_judge, debug=False)

        # When: Verifying
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])
        result = pipeline.verify_rule(rule, "ancestor", use_llm_judge=True)

        # Then: Should have warnings array (even if empty)
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_pipeline_handles_llm_failure_gracefully(self):
        """Should record error when LLM judge fails to verify"""
        # Given: Pipeline with failing LLM
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        provider = MockLLMProvider()  # No responses - will fail
        llm_judge = LLMJudge(provider)
        pipeline = VerificationPipeline(kb, llm_judge=llm_judge, debug=False)

        # When: Verifying (LLM will fail)
        rule = make_rule("ancestor", ["X", "Y"], [["parent", "X", "Y"]])
        result = pipeline.verify_rule(rule, "ancestor", use_llm_judge=True)

        # Then: Should fail overall validation with error about LLM failure
        assert result["valid"] is False
        assert "errors" in result
        # LLM failure should be recorded in errors
        assert any("verification failed" in e.lower() for e in result["errors"])


# ==================== Test Debug Mode ====================

class TestDebugMode:
    """Test that debug mode provides useful output"""

    def test_debug_mode_does_not_crash(self):
        """Debug mode should not cause crashes"""
        # Given: Judge in debug mode
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.9,
            "explanation": "OK"
        }))

        judge = LLMJudge(provider, debug=True)

        # When: Verifying (with debug output)
        rule = make_rule("test", ["X"], [["foo", "X"]])
        kb = KnowledgeBase()

        # Then: Should complete successfully
        result = judge.verify_rule(rule, "test", kb)
        assert result.is_correct is True


# ==================== Test Fact Verification ====================

class TestFactVerification:
    """Test LLM judge verification of generated facts"""

    def test_verify_fact_with_correct_judgement(self):
        """When LLM approves a fact, should return positive judgement"""
        # Given: A mock provider that approves facts
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.95,
            "explanation": "The name 'john' is typically male, so male(john) is correct",
            "suggested_correction": None
        }))

        judge = LLMJudge(provider)

        # When: Verifying a reasonable fact
        fact = Fact(Compound("male", [Atom("john")]))
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("john"), Atom("mary")])))

        result = judge.verify_fact(fact, "male", kb)

        # Then: Should return correct judgement
        assert result.is_correct is True
        assert result.confidence == 0.95
        assert "john" in result.explanation.lower() or "male" in result.explanation.lower()

    def test_verify_fact_with_incorrect_judgement(self):
        """When LLM rejects a fact, should return negative judgement with suggestion"""
        # Given: A mock provider that rejects a fact
        provider = MockLLMProvider(json.dumps({
            "is_correct": False,
            "confidence": 0.9,
            "explanation": "The name 'mary' is typically female, not male",
            "suggested_correction": "female(mary)"
        }))

        judge = LLMJudge(provider)

        # When: Verifying an incorrect fact
        fact = Fact(Compound("male", [Atom("mary")]))
        kb = KnowledgeBase()

        result = judge.verify_fact(fact, "male", kb)

        # Then: Should return incorrect judgement with correction
        assert result.is_correct is False
        assert result.confidence == 0.9
        assert "female" in result.suggested_correction.lower()

    def test_verify_fact_prompt_includes_kb_context(self):
        """Fact verification prompt should include KB context"""
        provider = MockLLMProvider(json.dumps({
            "is_correct": True,
            "confidence": 0.9,
            "explanation": "OK"
        }))

        judge = LLMJudge(provider)

        # When: Verifying with KB context
        fact = Fact(Compound("female", [Atom("mary")]))
        kb = KnowledgeBase()
        kb.add_fact(Fact(Compound("parent", [Atom("mary"), Atom("alice")])))

        judge.verify_fact(fact, "female", kb)

        # Then: Prompt should contain KB context
        prompt = provider.last_prompt
        assert "parent" in prompt.lower()
        assert "mary" in prompt.lower()

    def test_verify_fact_retries_on_failure(self):
        """Should retry fact verification on parse failures"""
        # Given: A provider that fails twice then succeeds
        provider = MockLLMProvider([
            "Not JSON",
            json.dumps({
                "is_correct": True,
                "confidence": 0.85,
                "explanation": "Eventually succeeded"
            })
        ])

        judge = LLMJudge(provider)

        # When: Verifying a fact
        fact = Fact(Compound("male", [Atom("bob")]))
        kb = KnowledgeBase()

        result = judge.verify_fact(fact, "male", kb, max_retries=3)

        # Then: Should eventually succeed
        assert result.is_correct is True
        assert provider.call_count == 2  # Tried 2 times

    def test_verify_fact_returns_failure_after_max_retries(self):
        """When all retries fail, should return failure"""
        # Given: A provider that always fails
        provider = MockLLMProvider(["Bad", "Bad", "Bad"])

        judge = LLMJudge(provider)

        # When: Verifying with limited retries
        fact = Fact(Compound("test", [Atom("x")]))
        kb = KnowledgeBase()

        result = judge.verify_fact(fact, "test", kb, max_retries=3)

        # Then: Should return failure
        assert result.is_correct is False
        assert result.confidence == 0.0
        assert "failed" in result.explanation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
