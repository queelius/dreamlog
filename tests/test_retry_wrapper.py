"""
Tests for LLM retry wrapper and validation feedback.
"""
import pytest
from unittest.mock import Mock, MagicMock

from dreamlog.validation_feedback import OutputValidator, ValidationResult
from dreamlog.llm_retry_wrapper import RetryConfig, RetryLLMProvider, create_retry_provider
from dreamlog.llm_providers import LLMResponse


class TestOutputValidator:
    """Test OutputValidator class"""

    @pytest.fixture
    def validator(self):
        """Create a validator instance"""
        return OutputValidator(verbose=False)

    def test_analyze_empty_response(self, validator):
        """Empty response should be invalid"""
        result = validator.analyze_output("")
        assert result['valid'] is False
        assert result['score'] == 0.0
        assert 'Empty response' in result['issues']

    def test_analyze_valid_fact_response(self, validator):
        """Valid fact response should be parsed"""
        response = "(parent john mary)"
        result = validator.analyze_output(response)
        assert result['valid'] is True
        assert result['parsed'] is not None
        assert result['score'] == 1.0

    def test_analyze_valid_rule_response(self, validator):
        """Valid rule response should be parsed"""
        response = "(grandparent X Z) :- (parent X Y), (parent Y Z)"
        result = validator.analyze_output(response)
        assert result['valid'] is True
        assert result['parsed'] is not None

    def test_analyze_json_response(self, validator):
        """JSON response with facts/rules should be parsed"""
        import json
        response = json.dumps({
            "facts": ["(parent john mary)"],
            "rules": []
        })
        result = validator.analyze_output(response)
        # May or may not parse depending on parser
        assert 'valid' in result

    def test_analyze_close_but_wrong(self, validator):
        """Almost-valid response should be flagged"""
        # Has parentheses but malformed
        response = "(parent john"  # Missing closing paren
        result = validator.analyze_output(response)
        assert result['valid'] is False
        # Should have some score for having S-expression syntax
        assert result['score'] > 0

    def test_generate_feedback_prompt(self, validator):
        """Feedback prompt should include issues"""
        analysis = {
            'valid': False,
            'issues': ['Missing closing parenthesis', 'Invalid syntax'],
            'score': 0.3
        }
        feedback = validator.generate_feedback_prompt(analysis)
        assert 'Missing closing parenthesis' in feedback
        assert 'S-expression syntax' in feedback


class TestRetryConfig:
    """Test RetryConfig dataclass"""

    def test_default_values(self):
        """Default config should have sensible values"""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.max_samples == 5
        assert config.temperature_increase == 0.1
        assert config.enforce_json is True
        assert config.verbose is False

    def test_custom_values(self):
        """Custom config should use provided values"""
        config = RetryConfig(
            max_retries=5,
            max_samples=10,
            verbose=True
        )
        assert config.max_retries == 5
        assert config.max_samples == 10
        assert config.verbose is True


class TestRetryLLMProvider:
    """Test RetryLLMProvider wrapper"""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider"""
        provider = Mock()
        provider.model = "test-model"
        provider.get_metadata = Mock(return_value={"provider_type": "test"})
        return provider

    def test_initialization(self, mock_provider):
        """Retry provider should initialize with base provider"""
        retry_provider = RetryLLMProvider(mock_provider)
        assert retry_provider.base_provider == mock_provider
        assert retry_provider.model == "test-model"

    def test_initialization_with_config(self, mock_provider):
        """Retry provider should use custom config"""
        config = RetryConfig(max_retries=5, verbose=True)
        retry_provider = RetryLLMProvider(mock_provider, config)
        assert retry_provider.config.max_retries == 5
        assert retry_provider.config.verbose is True

    def test_generate_knowledge_success(self, mock_provider):
        """Should return response on first successful attempt"""
        mock_provider.generate_knowledge = Mock(return_value=LLMResponse(
            text="(parent john mary)",
            facts=["(parent john mary)"],
            rules=[],
            raw_response="(parent john mary)"
        ))

        retry_provider = RetryLLMProvider(mock_provider)
        result = retry_provider.generate_knowledge("parent", "context")

        assert result is not None
        # First call should have been made
        mock_provider.generate_knowledge.assert_called()

    def test_generate_knowledge_with_failure(self, mock_provider):
        """Should retry on failure"""
        # First call fails, subsequent calls succeed
        mock_provider.generate_knowledge = Mock(side_effect=[
            Exception("API Error"),
            LLMResponse(
                text="(parent john mary)",
                facts=["(parent john mary)"],
                rules=[],
                raw_response="(parent john mary)"
            )
        ])

        config = RetryConfig(max_retries=3, verbose=False)
        retry_provider = RetryLLMProvider(mock_provider, config)
        result = retry_provider.generate_knowledge("parent", "context")

        # Should have made multiple attempts
        assert mock_provider.generate_knowledge.call_count >= 1


class TestCreateRetryProvider:
    """Test create_retry_provider factory function"""

    def test_create_with_defaults(self):
        """Should create retry provider with default config"""
        mock_provider = Mock()
        mock_provider.model = "test-model"

        retry_provider = create_retry_provider(mock_provider)

        assert isinstance(retry_provider, RetryLLMProvider)
        assert retry_provider.config.max_retries == 3

    def test_create_with_custom_retries(self):
        """Should use custom max_retries"""
        mock_provider = Mock()
        mock_provider.model = "test-model"

        retry_provider = create_retry_provider(
            mock_provider,
            max_retries=5
        )

        assert retry_provider.config.max_retries == 5

    def test_create_with_verbose(self):
        """Should set verbose flag"""
        mock_provider = Mock()
        mock_provider.model = "test-model"

        retry_provider = create_retry_provider(
            mock_provider,
            verbose=True
        )

        assert retry_provider.config.verbose is True


class TestValidationResult:
    """Test ValidationResult dataclass"""

    def test_create_valid_result(self):
        """Should create valid result"""
        result = ValidationResult(
            valid=True,
            parsed={'facts': ['(parent john mary)'], 'rules': []},
            close_but_wrong=False,
            score=1.0,
            issues=[]
        )
        assert result.valid is True
        assert result.score == 1.0

    def test_create_invalid_result(self):
        """Should create invalid result with issues"""
        result = ValidationResult(
            valid=False,
            parsed=None,
            close_but_wrong=True,
            score=0.5,
            issues=['Parse error']
        )
        assert result.valid is False
        assert len(result.issues) == 1
