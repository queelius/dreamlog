# MCP (Model Context Protocol) Integration

The MCP integration allows AI assistants like Claude Desktop to interact with DreamLog through a standardized tool interface.

## Installation

```bash
# Install with MCP support
pip install dreamlog[mcp]

# Or install MCP dependencies separately
pip install mcp websockets
```

## Configuration

### Claude Desktop Configuration

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "dreamlog": {
      "command": "python",
      "args": ["-m", "dreamlog.integrations.mcp"],
      "env": {
        "DREAMLOG_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Environment Variables

```bash
# LLM Configuration
export DREAMLOG_LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key

# MCP Server Settings
export MCP_PORT=8765
export MCP_HOST=localhost
export MCP_MAX_CONNECTIONS=10
```

## Available Tools

### `dreamlog_query`

Execute a query against the knowledge base.

```json
{
  "tool": "dreamlog_query",
  "parameters": {
    "query": "(parent john X)",
    "limit": 10
  }
}
```

**Parameters:**
- `query` (required): S-expression query
- `limit` (optional): Maximum results (default: 100)

**Returns:**
```json
{
  "results": [
    {"X": "mary"},
    {"X": "bob"}
  ],
  "count": 2
}
```

### `dreamlog_add_fact`

Add a fact to the knowledge base.

```json
{
  "tool": "dreamlog_add_fact",
  "parameters": {
    "fact": "(parent john mary)"
  }
}
```

**Parameters:**
- `fact` (required): S-expression fact

**Returns:**
```json
{
  "success": true,
  "message": "Fact added: (parent john mary)"
}
```

### `dreamlog_add_rule`

Add a rule to the knowledge base.

```json
{
  "tool": "dreamlog_add_rule",
  "parameters": {
    "rule": "(grandparent X Z) :- (parent X Y), (parent Y Z)"
  }
}
```

**Parameters:**
- `rule` (required): S-expression rule with :- operator

**Returns:**
```json
{
  "success": true,
  "message": "Rule added: (grandparent X Z) :- ..."
}
```

### `dreamlog_dream`

Run a dream cycle to optimize the knowledge base.

```json
{
  "tool": "dreamlog_dream",
  "parameters": {
    "cycles": 3,
    "focus": "compression",
    "verify": true
  }
}
```

**Parameters:**
- `cycles` (optional): Number of dream cycles (default: 1)
- `focus` (optional): "compression", "abstraction", or "all"
- `verify` (optional): Verify behavior preservation (default: true)

**Returns:**
```json
{
  "session": {
    "insights": 5,
    "compression_ratio": 0.25,
    "verified": true,
    "improvements": ["Merged similar rules", "Found abstraction"]
  }
}
```

### `dreamlog_load`

Load a knowledge base from file.

```json
{
  "tool": "dreamlog_load",
  "parameters": {
    "path": "/path/to/knowledge.dl"
  }
}
```

**Parameters:**
- `path` (required): File path to load

**Returns:**
```json
{
  "success": true,
  "facts_loaded": 50,
  "rules_loaded": 10
}
```

### `dreamlog_save`

Save the current knowledge base.

```json
{
  "tool": "dreamlog_save",
  "parameters": {
    "path": "/path/to/knowledge.dl",
    "format": "sexp"
  }
}
```

**Parameters:**
- `path` (required): File path to save
- `format` (optional): "sexp" or "json" (default: "sexp")

**Returns:**
```json
{
  "success": true,
  "facts_saved": 50,
  "rules_saved": 10
}
```

### `dreamlog_stats`

Get knowledge base statistics.

```json
{
  "tool": "dreamlog_stats"
}
```

**Returns:**
```json
{
  "facts": 50,
  "rules": 10,
  "functors": ["parent", "grandparent", "sibling"],
  "total_items": 60
}
```

### `dreamlog_clear`

Clear the knowledge base.

```json
{
  "tool": "dreamlog_clear",
  "parameters": {
    "confirm": true
  }
}
```

**Parameters:**
- `confirm` (required): Must be true to clear

**Returns:**
```json
{
  "success": true,
  "message": "Knowledge base cleared"
}
```

## Starting the MCP Server

### Command Line

```bash
# Start with defaults
python -m dreamlog.integrations.mcp

# Specify port and host
python -m dreamlog.integrations.mcp --port 9000 --host 0.0.0.0

# Enable debug logging
python -m dreamlog.integrations.mcp --debug

# Use specific LLM provider
python -m dreamlog.integrations.mcp --llm-provider anthropic
```

### Programmatic

```python
from dreamlog.integrations.mcp import DreamLogMCPServer

server = DreamLogMCPServer(
    port=8765,
    host="localhost",
    llm_provider="openai"
)

# Run server
server.run()
```

## WebSocket Protocol

The MCP server uses WebSocket for communication:

```python
import websockets
import json

async def query_dreamlog():
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        # Send tool request
        request = {
            "id": "1",
            "method": "tools/call",
            "params": {
                "name": "dreamlog_query",
                "arguments": {
                    "query": "(parent john X)"
                }
            }
        }
        
        await websocket.send(json.dumps(request))
        
        # Receive response
        response = await websocket.recv()
        result = json.loads(response)
        print(result)
```

## Error Handling

The MCP server returns structured errors:

```json
{
  "error": {
    "code": -32602,
    "message": "Invalid query syntax",
    "data": {
      "query": "(invalid",
      "position": 8
    }
  }
}
```

Error codes:
- `-32700`: Parse error
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

## Security

### Authentication

Add authentication to the MCP server:

```python
# In your configuration
{
  "mcpServers": {
    "dreamlog": {
      "command": "python",
      "args": ["-m", "dreamlog.integrations.mcp"],
      "env": {
        "MCP_AUTH_TOKEN": "secret-token"
      }
    }
  }
}
```

### Rate Limiting

Configure rate limits:

```python
# Environment variables
export MCP_RATE_LIMIT=100  # Requests per minute
export MCP_BURST_LIMIT=10  # Burst allowance
```

## Monitoring

### Metrics

The MCP server exposes metrics:

```json
{
  "tool": "dreamlog_metrics"
}
```

Returns:
```json
{
  "requests_total": 1000,
  "requests_per_minute": 50,
  "average_response_time": 0.05,
  "active_connections": 3,
  "errors_total": 2
}
```

### Logging

Enable detailed logging:

```bash
# Set log level
export MCP_LOG_LEVEL=DEBUG

# Log to file
export MCP_LOG_FILE=/var/log/dreamlog-mcp.log
```

## Advanced Usage

### Custom Tools

Add custom tools to the MCP server:

```python
from dreamlog.integrations.mcp import MCPTool, register_tool

@register_tool
class CustomAnalysisTool(MCPTool):
    name = "dreamlog_analyze"
    description = "Analyze knowledge base patterns"
    
    def execute(self, params):
        # Custom analysis logic
        patterns = analyze_patterns(self.kb)
        return {"patterns": patterns}
```

### Middleware

Add middleware for request processing:

```python
from dreamlog.integrations.mcp import MCPMiddleware

class LoggingMiddleware(MCPMiddleware):
    async def process_request(self, request):
        print(f"Request: {request}")
        return await self.next(request)

server.add_middleware(LoggingMiddleware())
```

## Troubleshooting

### Connection Issues

```bash
# Test WebSocket connection
wscat -c ws://localhost:8765

# Check if port is in use
lsof -i :8765

# Test with curl
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: test" \
     http://localhost:8765
```

### Debug Mode

```bash
# Enable verbose logging
export MCP_DEBUG=true
python -m dreamlog.integrations.mcp --debug
```

### Common Errors

**"Connection refused"**
- Check if server is running
- Verify port and host settings
- Check firewall rules

**"Invalid tool name"**
- Verify tool name spelling
- Check available tools with `dreamlog_list_tools`

**"Query timeout"**
- Increase timeout settings
- Optimize complex queries
- Check LLM provider status