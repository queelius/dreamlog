import pytest
from experiments.ollama_helper import make_ollama_client


@pytest.mark.integration
def test_ollama_reachable_and_responds():
    client = make_ollama_client(timeout=60)
    try:
        resp = client.complete("Reply with exactly one word: pong")
    except Exception as e:
        pytest.skip(f"Ollama host not reachable: {e}")
    assert "pong" in resp.lower()
    assert client.usage.calls == 1
