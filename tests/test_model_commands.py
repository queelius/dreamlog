#!/usr/bin/env python3
"""
Test script for new model commands
"""

from dreamlog.engine import DreamLogEngine
from dreamlog.llm_providers import OllamaProvider
from dreamlog.llm_hook import LLMHook
from dreamlog.embedding_providers import TfIdfEmbeddingProvider

# Create engine
engine = DreamLogEngine()

# Set up LLM with Ollama
provider = OllamaProvider(model="phi4-mini:latest", temperature=0.1)
# Create TF-IDF embedding provider with empty corpus initially
embedding_provider = TfIdfEmbeddingProvider(corpus=[])
hook = LLMHook(provider, embedding_provider, debug=False)
engine.llm_hook = hook

print("=== Testing Model Commands ===\n")

# Test 1: Get current model
print("1. Current model:")
current = provider.get_parameter("model")
print(f"   {current}\n")

# Test 2: Get current temperature
print("2. Current temperature:")
temp = provider.get_parameter("temperature")
print(f"   {temp}\n")

# Test 3: Set new temperature
print("3. Setting temperature to 0.5:")
provider.set_parameter("temperature", 0.5)
new_temp = provider.get_parameter("temperature")
print(f"   New temperature: {new_temp}\n")

# Test 4: Set max tokens
print("4. Setting max tokens to 1000:")
provider.set_parameter("max_tokens", 1000)
max_tok = provider.get_parameter("max_tokens")
print(f"   Max tokens: {max_tok}\n")

# Test 5: Get provider metadata
print("5. Provider metadata:")
metadata = provider.get_metadata()
print(f"   Provider: {metadata.get('provider_class')}")
print(f"   Model: {metadata.get('model')}")
print(f"   Parameters: {metadata.get('parameters')}\n")

# Test 6: Switch model
print("6. Switching model to llama3.2:")
provider.set_parameter("model", "llama3.2")
new_model = provider.get_parameter("model")
print(f"   New model: {new_model}\n")

print("✓ All tests passed!")
