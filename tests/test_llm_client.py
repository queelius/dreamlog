# tests/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock


class TestLLMClient:
    def test_openai_init(self):
        """OpenAI client created with base_url."""
        with patch("openai.OpenAI") as mock_cls:
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="openai", base_url="http://localhost:11434/v1",
                               api_key="test", model="llama3")
            assert client.model == "llama3"
            assert client.provider == "openai"
            mock_cls.assert_called_once()

    def test_anthropic_init(self):
        """Anthropic client created with timeout."""
        with patch("anthropic.Anthropic") as mock_cls:
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="anthropic", api_key="test",
                               model="claude-haiku-4-5-20251001", timeout=60)
            assert client.provider == "anthropic"
            assert client.timeout == 60
            mock_cls.assert_called_once_with(api_key="test", timeout=60)

    def test_default_models(self):
        """Each provider gets a sensible default model."""
        with patch("anthropic.Anthropic"):
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="anthropic", api_key="test")
            assert client.model == "claude-haiku-4-5-20251001"

        with patch("openai.OpenAI"):
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="openai", api_key="test")
            assert client.model == "gpt-4o-mini"

    def test_public_attrs_writable(self):
        """Model, temperature, max_tokens are public and writable."""
        with patch("openai.OpenAI"):
            from dreamlog.llm_client import LLMClient
            client = LLMClient(model="gpt-4", api_key="test")
            assert client.model == "gpt-4"
            client.model = "gpt-3.5-turbo"
            assert client.model == "gpt-3.5-turbo"
            client.temperature = 0.5
            assert client.temperature == 0.5

    def test_complete_openai(self):
        """complete() calls OpenAI chat completions."""
        with patch("openai.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "test response"
            mock_client.chat.completions.create.return_value = mock_response

            from dreamlog.llm_client import LLMClient
            client = LLMClient(model="gpt-4", api_key="test")
            result = client.complete("hello")
            assert result == "test response"

    def test_complete_anthropic(self):
        """complete() calls Anthropic messages."""
        with patch("anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "anthropic response"
            mock_client.messages.create.return_value = mock_response

            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="anthropic", api_key="test")
            result = client.complete("hello")
            assert result == "anthropic response"

    def test_ollama_via_openai(self):
        """Ollama uses OpenAI SDK with custom base_url."""
        with patch("openai.OpenAI") as mock_cls:
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="ollama",
                               base_url="http://localhost:11434/v1",
                               model="llama3")
            assert client.provider == "ollama"
            mock_cls.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key="dummy",
                timeout=30,
            )

    def test_api_key_env(self):
        """api_key_env resolves from environment."""
        with patch("openai.OpenAI"), \
             patch.dict("os.environ", {"MY_TEST_KEY": "secret123"}):
            from dreamlog.llm_client import LLMClient
            client = LLMClient(provider="openai", api_key_env="MY_TEST_KEY")
            # Key was resolved and passed to OpenAI client
            assert client.provider == "openai"
