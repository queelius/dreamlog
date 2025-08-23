# REST API Integration

The DreamLog REST API provides HTTP endpoints for interacting with the knowledge base and includes WebSocket support for real-time REPL sessions.

## Installation

```bash
# Install with API support
pip install dreamlog[api]

# Or install dependencies separately
pip install fastapi uvicorn websockets
```

## Starting the Server

### Command Line

```bash
# Start with defaults
python -m dreamlog.integrations.api

# Specify port and host
python -m dreamlog.integrations.api --port 8000 --host 0.0.0.0

# Enable CORS for web applications
python -m dreamlog.integrations.api --cors

# With authentication
python -m dreamlog.integrations.api --auth-token your-secret-token
```

### Programmatic

```python
from dreamlog.integrations.api import DreamLogAPI

app = DreamLogAPI(
    port=8000,
    host="0.0.0.0",
    cors=True,
    auth_token="secret"
)

app.run()
```

## API Endpoints

### Query Execution

**POST** `/query`

Execute a DreamLog query.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "(parent john X)",
    "limit": 10
  }'
```

**Request:**
```json
{
  "query": "(parent john X)",
  "limit": 10
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {"X": "mary"},
    {"X": "bob"}
  ],
  "count": 2,
  "execution_time": 0.023
}
```

### Add Fact

**POST** `/fact`

Add a fact to the knowledge base.

```bash
curl -X POST http://localhost:8000/fact \
  -H "Content-Type: application/json" \
  -d '{
    "fact": "(parent john mary)"
  }'
```

**Request:**
```json
{
  "fact": "(parent john mary)"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Fact added successfully",
  "fact": "(parent john mary)"
}
```

### Add Rule

**POST** `/rule`

Add a rule to the knowledge base.

```bash
curl -X POST http://localhost:8000/rule \
  -H "Content-Type: application/json" \
  -d '{
    "rule": "(grandparent X Z) :- (parent X Y), (parent Y Z)"
  }'
```

**Request:**
```json
{
  "rule": "(grandparent X Z) :- (parent X Y), (parent Y Z)"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Rule added successfully",
  "head": "(grandparent X Z)",
  "body": ["(parent X Y)", "(parent Y Z)"]
}
```

### Knowledge Base Operations

**GET** `/kb`

Get the entire knowledge base.

```bash
curl http://localhost:8000/kb
```

**Response:**
```json
{
  "facts": [
    "(parent john mary)",
    "(parent mary alice)"
  ],
  "rules": [
    {
      "head": "(grandparent X Z)",
      "body": ["(parent X Y)", "(parent Y Z)"]
    }
  ],
  "stats": {
    "num_facts": 2,
    "num_rules": 1,
    "functors": ["parent", "grandparent"]
  }
}
```

**DELETE** `/kb`

Clear the knowledge base.

```bash
curl -X DELETE http://localhost:8000/kb \
  -H "X-Confirm: true"
```

**Response:**
```json
{
  "success": true,
  "message": "Knowledge base cleared"
}
```

### Load Knowledge Base

**POST** `/kb/load`

Load knowledge base from file.

```bash
curl -X POST http://localhost:8000/kb/load \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/path/to/knowledge.dl",
    "format": "sexp"
  }'
```

**Request:**
```json
{
  "path": "/path/to/knowledge.dl",
  "format": "sexp"
}
```

**Response:**
```json
{
  "success": true,
  "facts_loaded": 50,
  "rules_loaded": 10
}
```

### Save Knowledge Base

**POST** `/kb/save`

Save knowledge base to file.

```bash
curl -X POST http://localhost:8000/kb/save \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/path/to/knowledge.dl",
    "format": "json"
  }'
```

**Request:**
```json
{
  "path": "/path/to/knowledge.dl",
  "format": "json"
}
```

**Response:**
```json
{
  "success": true,
  "facts_saved": 50,
  "rules_saved": 10,
  "file_size": 2048
}
```

### Dream Cycle

**POST** `/dream`

Run optimization cycle.

```bash
curl -X POST http://localhost:8000/dream \
  -H "Content-Type: application/json" \
  -d '{
    "cycles": 3,
    "focus": "compression",
    "verify": true
  }'
```

**Request:**
```json
{
  "cycles": 3,
  "focus": "compression",
  "verify": true
}
```

**Response:**
```json
{
  "success": true,
  "session": {
    "insights_found": 5,
    "compression_ratio": 0.25,
    "verification_passed": true,
    "improvements": [
      "Merged redundant rules",
      "Found general abstraction"
    ]
  }
}
```

### Statistics

**GET** `/stats`

Get API and knowledge base statistics.

```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "kb": {
    "facts": 50,
    "rules": 10,
    "functors": ["parent", "grandparent"]
  },
  "api": {
    "requests_total": 1000,
    "requests_per_minute": 50,
    "average_response_time": 0.05,
    "uptime_seconds": 3600
  }
}
```

## WebSocket REPL

Connect to interactive REPL via WebSocket.

### JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8000/repl');

ws.onopen = () => {
  console.log('Connected to DreamLog REPL');
  
  // Send command
  ws.send(JSON.stringify({
    type: 'command',
    command: '(parent john X)'
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  console.log('Response:', response);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

### Python Client

```python
import asyncio
import websockets
import json

async def repl_session():
    uri = "ws://localhost:8000/repl"
    
    async with websockets.connect(uri) as websocket:
        # Send query
        await websocket.send(json.dumps({
            "type": "command",
            "command": "(parent john X)"
        }))
        
        # Receive response
        response = await websocket.recv()
        result = json.loads(response)
        print(f"Results: {result}")
        
        # Add fact
        await websocket.send(json.dumps({
            "type": "command",
            "command": "fact (parent bob alice)"
        }))
        
        response = await websocket.recv()
        print(f"Response: {json.loads(response)}")

asyncio.run(repl_session())
```

### REPL Commands

Available commands in WebSocket REPL:

- `(query)` - Execute query
- `fact (fact)` - Add fact
- `rule (head) :- (body)` - Add rule
- `load path/to/file` - Load knowledge base
- `save path/to/file` - Save knowledge base
- `clear` - Clear knowledge base
- `stats` - Show statistics
- `dream` - Run dream cycle
- `help` - Show help

## Authentication

### API Key Authentication

```bash
# Start server with API key
python -m dreamlog.integrations.api --api-key secret-key

# Make authenticated request
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "(parent john X)"}'
```

### JWT Authentication

```python
from dreamlog.integrations.api import DreamLogAPI
from dreamlog.integrations.api.auth import JWTAuth

app = DreamLogAPI(
    auth=JWTAuth(secret="jwt-secret")
)
```

## Rate Limiting

Configure rate limiting:

```python
from dreamlog.integrations.api import DreamLogAPI
from dreamlog.integrations.api.ratelimit import RateLimiter

app = DreamLogAPI(
    rate_limiter=RateLimiter(
        requests_per_minute=100,
        burst=10
    )
)
```

## CORS Configuration

Enable CORS for web applications:

```python
app = DreamLogAPI(
    cors=True,
    cors_origins=["http://localhost:3000"],
    cors_methods=["GET", "POST"],
    cors_headers=["Content-Type", "X-API-Key"]
)
```

## Error Responses

The API returns structured error responses:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_QUERY",
    "message": "Query syntax error",
    "details": {
      "query": "(invalid",
      "position": 8,
      "expected": "closing parenthesis"
    }
  }
}
```

Error codes:
- `INVALID_QUERY` - Query syntax error
- `INVALID_FACT` - Fact syntax error
- `INVALID_RULE` - Rule syntax error
- `NOT_FOUND` - Resource not found
- `UNAUTHORIZED` - Authentication required
- `RATE_LIMITED` - Too many requests
- `INTERNAL_ERROR` - Server error

## Monitoring

### Health Check

**GET** `/health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "llm_provider": "openai"
}
```

### Metrics

**GET** `/metrics`

Returns Prometheus-compatible metrics:

```
# HELP dreamlog_requests_total Total number of requests
# TYPE dreamlog_requests_total counter
dreamlog_requests_total 1000

# HELP dreamlog_query_duration_seconds Query execution time
# TYPE dreamlog_query_duration_seconds histogram
dreamlog_query_duration_seconds_bucket{le="0.01"} 500
dreamlog_query_duration_seconds_bucket{le="0.1"} 900
dreamlog_query_duration_seconds_bucket{le="1"} 990
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "dreamlog.integrations.api", "--host", "0.0.0.0"]
```

```bash
# Build image
docker build -t dreamlog-api .

# Run container
docker run -p 8000:8000 \
  -e DREAMLOG_LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=your-key \
  dreamlog-api
```

## Client Libraries

### Python Client

```python
from dreamlog.client import DreamLogClient

client = DreamLogClient("http://localhost:8000")

# Add facts
client.add_fact("(parent john mary)")

# Query
results = client.query("(parent john X)")
print(results)

# Dream cycle
session = client.dream(cycles=3)
print(f"Compression: {session.compression_ratio}")
```

### JavaScript Client

```javascript
import { DreamLogClient } from 'dreamlog-js';

const client = new DreamLogClient('http://localhost:8000');

// Add facts
await client.addFact('(parent john mary)');

// Query
const results = await client.query('(parent john X)');
console.log(results);

// Dream cycle
const session = await client.dream({ cycles: 3 });
console.log(`Compression: ${session.compressionRatio}`);
```