# DreamLog MCP Server

Model Context Protocol server for DreamLog, enabling LLMs to interact with DreamLog as a tool.

## Features

- **Tools**: Query, add facts/rules, explain reasoning
- **Resources**: Access knowledge base and statistics
- **LLM Integration**: Automatic knowledge generation from config
- **Config Support**: Uses DreamLog unified configuration

## Usage

### Basic Usage

```bash
# Start server with default config (dreamlog_config.yaml)
python integrations/mcp/dreamlog_mcp_server.py

# Start with custom config
python integrations/mcp/dreamlog_mcp_server.py --config /path/to/config.yaml

# Load existing knowledge base
python integrations/mcp/dreamlog_mcp_server.py --kb facts.json

# Custom port
python integrations/mcp/dreamlog_mcp_server.py --port 9000
```

### Available Tools

1. **dreamlog_query** - Query the knowledge base
   ```json
   {
     "query": "(parent john X)",
     "limit": 10
   }
   ```

2. **dreamlog_add_fact** - Add a fact
   ```json
   {
     "fact": "(parent john mary)"
   }
   ```

3. **dreamlog_add_rule** - Add a rule
   ```json
   {
     "head": "(grandparent X Z)",
     "body": ["(parent X Y)", "(parent Y Z)"]
   }
   ```

4. **dreamlog_explain** - Explain query resolution
   ```json
   {
     "query": "(parent john X)"
   }
   ```

### Resources

- `dreamlog://kb/current` - Current knowledge base (JSON)
- `dreamlog://kb/stats` - Statistics (facts, rules, functors)

## Configuration

The server uses DreamLog's unified config system. Set `llm_enabled: true` in your config to enable automatic knowledge generation.

Example `dreamlog_config.yaml`:
```yaml
provider:
  provider: ollama
  base_url: http://localhost:11434
  model: qwen3:4b
  temperature: 0.3

llm_enabled: true
```

## MCP Protocol

The server implements the Model Context Protocol v1.0, compatible with:
- Claude Desktop
- Other MCP-enabled LLM clients

## Testing

```python
import asyncio
from integrations.mcp.dreamlog_mcp_server import DreamLogMCPServer

async def test():
    server = DreamLogMCPServer()
    result = await server.handle_tool_call('dreamlog_query', {
        'query': '(parent john X)'
    })
    print(result)

asyncio.run(test())
```
