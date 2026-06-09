"""Connectivity probe for the remote Ollama qwen2.5:3b model used by EX28.

Marked `integration`: it makes a real network call. It SKIPS (not fails) when
the host is unreachable, but lets genuine errors (a missing dependency, or a
bug in LLMClient) surface as real failures rather than masking them as a skip.
"""
import pytest
from experiments.ollama_helper import make_ollama_client


def _connection_errors():
    """Exception types meaning 'host unreachable' (skip), not 'code is broken'.

    Built lazily so a missing openai dependency raises ImportError from
    complete() and fails the test (a real setup problem) instead of skipping.
    """
    types = [OSError, TimeoutError]
    try:
        import openai
        types += [openai.APIConnectionError, openai.APITimeoutError]
    except ImportError:
        pass
    return tuple(types)


@pytest.mark.integration
def test_ollama_reachable_and_responds():
    client = make_ollama_client(timeout=60)
    try:
        resp = client.complete("Reply with exactly one word: pong")
    except _connection_errors() as e:
        pytest.skip(f"Ollama host not reachable: {e}")
    assert len(resp.strip()) > 0          # connectivity: any non-empty reply
    assert client.usage.calls == 1
