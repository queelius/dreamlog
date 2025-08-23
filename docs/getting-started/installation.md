# Installation

## Requirements

- Python 3.8 or higher
- No external dependencies for core functionality
- Optional: API keys for LLM providers

## Install from Source

```bash
# Clone the repository
git clone https://github.com/queelius/dreamlog.git
cd dreamlog

# Install in development mode
pip install -e .
```

## Install with pip (coming soon)

```bash
pip install dreamlog
```

## Install with Extra Dependencies

```bash
# For REST API server
pip install dreamlog[api]

# For Jupyter integration
pip install dreamlog[jupyter]

# For VS Code Language Server
pip install dreamlog[lsp]

# All integrations
pip install dreamlog[all]
```

## Verify Installation

```python
# Test basic functionality
from dreamlog.pythonic import dreamlog

jl = dreamlog()
jl.fact("parent", "john", "mary")

for result in jl.query("parent", "john", "X"):
    print(f"Success! Found: {result['X']}")
```

## Configure LLM Providers

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
export DreamLog_LLM_PROVIDER="openai"
export OPENAI_MODEL="gpt-4"  # Optional, defaults to gpt-3.5-turbo
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export DreamLog_LLM_PROVIDER="anthropic"
export ANTHROPIC_MODEL="claude-3-opus-20240229"  # Optional
```

### Ollama (Local)

```bash
# Start Ollama server first
ollama serve

# Configure DreamLog
export DreamLog_LLM_PROVIDER="ollama"
export OLLAMA_MODEL="llama2"  # Or any model you have
export OLLAMA_BASE_URL="http://localhost:11434"  # Default
```

### Configuration File

Create `~/.dreamlog/llm_config.json`:

```json
{
    "provider": "openai",
    "api_key": "sk-...",
    "model": "gpt-4",
    "temperature": 0.1
}
```

## Test LLM Integration

```python
from dreamlog.pythonic import dreamlog

# Will auto-detect from environment or config file
jl = dreamlog(llm_provider="openai")

# Query for undefined knowledge
# LLM will generate relevant facts/rules
for result in jl.query("capital", "france", "X"):
    print(f"The capital of France is {result['X']}")
```

## Troubleshooting

### Import Error

If you get import errors, ensure DreamLog is in your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/dreamlog"
```

### LLM Provider Not Found

If LLM auto-detection fails:

```python
from dreamlog.llm_config import LLMConfig

# List available providers
print(LLMConfig.list_providers())

# Check current configuration
provider = LLMConfig.auto_detect()
print(f"Detected: {type(provider).__name__}")
```

### Network Issues with LLM

For environments behind proxies:

```bash
export HTTP_PROXY="http://proxy.example.com:8080"
export HTTPS_PROXY="http://proxy.example.com:8080"
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get running in 5 minutes
- [Tutorial](tutorial.md) - Learn DreamLog step by step
- [LLM Configuration](../guide/llm.md) - Advanced LLM setup