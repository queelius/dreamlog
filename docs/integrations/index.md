# DreamLog Integrations

DreamLog provides multiple integration options to work with various tools and platforms.

## Available Integrations

### [MCP Server](mcp.md)
Model Context Protocol integration for AI assistants like Claude Desktop.

- Tool-based interaction with DreamLog
- Knowledge base management
- Query execution
- Dream cycle operations

### [REST API](rest_api.md)
HTTP API server for web applications and services.

- RESTful endpoints for all operations
- WebSocket support for real-time REPL
- JSON request/response format
- Authentication and rate limiting

### Jupyter Integration (Coming Soon)
Magic commands for Jupyter notebooks.

- `%%dreamlog` cell magic
- Interactive knowledge exploration
- Visualization of query results
- Dream cycle analysis

### TUI (Terminal User Interface)
Interactive terminal interface.

- `dreamlog` or `python -m dreamlog.tui` - Main TUI
- Rich formatting and syntax highlighting
- LLM and dreaming commands
- Knowledge base management

## Quick Start

### MCP Server

```bash
# Install MCP server
pip install dreamlog[mcp]

# Start server
python -m dreamlog.integrations.mcp
```

### REST API

```bash
# Start API server
python -m dreamlog.integrations.api --port 8000

# Make requests
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "(parent john X)"}'
```

### Jupyter

```python
# Load extension
%load_ext dreamlog.integrations.jupyter

# Use magic command
%%dreamlog
(parent john mary)
(parent mary alice)
(grandparent X Z) :- (parent X Y), (parent Y Z)
query: (grandparent john X)
```

## Integration Architecture

```
┌──────────────────────────────────────┐
│         Client Applications          │
├────────┬────────┬────────┬──────────┤
│  MCP   │  REST  │Jupyter │  VS Code │
│ Server │  API   │ Magic  │Extension │
├────────┴────────┴────────┴──────────┤
│       DreamLog Core Engine          │
├──────────────────────────────────────┤
│    Knowledge Base & LLM Providers    │
└──────────────────────────────────────┘
```

## Common Integration Patterns

### 1. Embedded Integration

Embed DreamLog directly in your Python application:

```python
from dreamlog import DreamLog

dl = DreamLog()
dl.add_fact("(parent john mary)")
results = dl.query("(parent john X)")
```

### 2. Service Integration

Run DreamLog as a service and communicate via API:

```python
import requests

response = requests.post(
    "http://dreamlog-server:8000/query",
    json={"query": "(parent john X)"}
)
results = response.json()
```

### 3. Tool Integration

Use DreamLog through MCP tools in AI assistants:

```json
{
  "tool": "dreamlog_query",
  "parameters": {
    "query": "(parent john X)",
    "knowledge_base": "family.dl"
  }
}
```

## Configuration

Each integration can be configured via:

1. **Environment variables**
2. **Configuration files** (`dreamlog.yaml`)
3. **Command-line arguments**

Example configuration:

```yaml
# dreamlog.yaml
integrations:
  api:
    port: 8000
    host: "0.0.0.0"
    cors: true
  
  mcp:
    max_query_results: 100
    enable_dreaming: true
  
  jupyter:
    auto_display: true
    syntax_highlight: true
```

## Security Considerations

When using integrations:

1. **Authentication**: Use API keys for REST endpoints
2. **Rate limiting**: Prevent abuse of LLM providers
3. **Input validation**: Sanitize all user inputs
4. **Sandboxing**: Run untrusted queries in isolation
5. **Audit logging**: Track all operations

## Performance Tips

1. **Connection pooling**: Reuse connections for API calls
2. **Caching**: Cache query results and LLM responses
3. **Async operations**: Use async/await for I/O operations
4. **Batch processing**: Group multiple operations
5. **Resource limits**: Set memory and time limits

## Troubleshooting

### Common Issues

**MCP Server not connecting**
- Check firewall settings
- Verify port availability
- Review MCP configuration

**REST API timeout**
- Increase timeout settings
- Optimize complex queries
- Check LLM provider response times

**Jupyter kernel crashes**
- Limit query result size
- Check memory usage
- Update Jupyter and dependencies

## Contributing

To add a new integration:

1. Create module in `dreamlog/integrations/`
2. Implement integration protocol
3. Add documentation
4. Write tests
5. Submit pull request

See the [GitHub repository](https://github.com/queelius/dreamlog) for details.