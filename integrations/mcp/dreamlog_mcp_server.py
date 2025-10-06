"""
DreamLog MCP (Model Context Protocol) Server

This allows DreamLog to be used as a tool/resource by LLMs that support MCP.
MCP enables LLMs to interact with external systems in a standardized way.

Usage:
    python dreamlog_mcp_server.py --kb knowledge.json --port 8765
"""

import json
import asyncio
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dreamlog.engine import DreamLogEngine
from dreamlog.prefix_parser import parse_s_expression, parse_prefix_notation
from dreamlog.llm_hook import LLMHook
from dreamlog.llm_providers import create_provider
from dreamlog.embedding_providers import TfIdfEmbeddingProvider
from dreamlog.config import DreamLogConfig, get_config
from dreamlog.prompt_template_system import RULE_EXAMPLES


@dataclass
class MCPTool:
    """Represents a tool in the MCP protocol"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    

@dataclass
class MCPResource:
    """Represents a resource in the MCP protocol"""
    uri: str
    name: str
    description: str
    mime_type: str


class DreamLogMCPServer:
    """
    MCP Server for DreamLog
    
    Provides tools and resources for:
    - Querying the knowledge base
    - Adding facts and rules
    - Loading/saving knowledge bases
    - Generating knowledge with LLM
    """
    
    def __init__(self, kb_path: Optional[str] = None, config: Optional[DreamLogConfig] = None):
        # Load config or use provided one
        self.config = config or get_config()
        self.engine = DreamLogEngine()

        # Setup LLM if enabled in config
        if self.config.llm_enabled:
            try:
                # Create provider from config
                provider_kwargs = {
                    'model': self.config.provider.model,
                    'temperature': self.config.provider.temperature,
                    'max_tokens': self.config.provider.max_tokens,
                }

                if self.config.provider.base_url:
                    provider_kwargs['base_url'] = self.config.provider.base_url

                api_key = self.config.provider.get_api_key()
                if api_key:
                    provider_kwargs['api_key'] = api_key

                provider = create_provider(
                    provider_type=self.config.provider.provider,
                    **provider_kwargs
                )

                # Create embedding provider
                embedding_provider = TfIdfEmbeddingProvider(RULE_EXAMPLES)

                # Create LLM hook
                self.engine.llm_hook = LLMHook(provider, embedding_provider, debug=False)
                print(f"✓ LLM enabled with {self.config.provider.provider} ({self.config.provider.model})")
            except Exception as e:
                print(f"⚠ LLM setup failed: {e}")

        if kb_path and os.path.exists(kb_path):
            self.load_knowledge_base(kb_path)
        
        self.tools = self._define_tools()
        self.resources = self._define_resources()
    
    def _define_tools(self) -> List[MCPTool]:
        """Define available MCP tools"""
        return [
            MCPTool(
                name="dreamlog_query",
                description="Query the DreamLog knowledge base using Prolog-like syntax",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query in S-expression format, e.g., (parent john X)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of solutions",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="dreamlog_add_fact",
                description="Add a fact to the DreamLog knowledge base",
                input_schema={
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "Fact in S-expression format, e.g., (parent john mary)"
                        }
                    },
                    "required": ["fact"]
                }
            ),
            MCPTool(
                name="dreamlog_add_rule",
                description="Add a rule to the DreamLog knowledge base",
                input_schema={
                    "type": "object",
                    "properties": {
                        "head": {
                            "type": "string",
                            "description": "Rule head in S-expression format"
                        },
                        "body": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Rule body as list of conditions"
                        }
                    },
                    "required": ["head", "body"]
                }
            ),
            MCPTool(
                name="dreamlog_explain",
                description="Explain how a query would be resolved",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to explain"
                        }
                    },
                    "required": ["query"]
                }
            )
        ]
    
    def _define_resources(self) -> List[MCPResource]:
        """Define available MCP resources"""
        return [
            MCPResource(
                uri="dreamlog://kb/current",
                name="Current Knowledge Base",
                description="The current DreamLog knowledge base in JSON format",
                mime_type="application/json"
            ),
            MCPResource(
                uri="dreamlog://kb/stats",
                name="Knowledge Base Statistics",
                description="Statistics about the current knowledge base",
                mime_type="application/json"
            )
        ]
    
    def load_knowledge_base(self, path: str) -> None:
        """Load a knowledge base from file"""
        with open(path, 'r') as f:
            data = f.read()
            self.engine.load_from_prefix(data)
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        
        if tool_name == "dreamlog_query":
            query_str = arguments["query"]
            limit = arguments.get("limit", 10)
            
            # Parse query
            if query_str.startswith("("):
                query_term = parse_s_expression(query_str)
            else:
                query_term = parse_prefix_notation(json.loads(query_str))
            
            # Execute query
            solutions = []
            for i, solution in enumerate(self.engine.query([query_term])):
                if i >= limit:
                    break
                solutions.append({
                    "bindings": {k: str(v) for k, v in solution.bindings.items()},
                    "ground_bindings": {k: str(v) for k, v in solution.get_ground_bindings().items()}
                })
            
            return {
                "success": True,
                "solutions": solutions,
                "count": len(solutions)
            }
        
        elif tool_name == "dreamlog_add_fact":
            fact_str = arguments["fact"]
            
            # Parse fact
            if fact_str.startswith("("):
                fact_term = parse_s_expression(fact_str)
            else:
                fact_term = parse_prefix_notation(json.loads(fact_str))
            
            # Add to KB
            self.engine.add_fact(fact_term)
            
            return {
                "success": True,
                "message": f"Added fact: {fact_term}"
            }
        
        elif tool_name == "dreamlog_add_rule":
            head_str = arguments["head"]
            body_strs = arguments["body"]
            
            # Parse head and body
            if head_str.startswith("("):
                head = parse_s_expression(head_str)
                body = [parse_s_expression(b) for b in body_strs]
            else:
                head = parse_prefix_notation(json.loads(head_str))
                body = [parse_prefix_notation(json.loads(b)) for b in body_strs]

            # Add to KB
            from dreamlog.knowledge import Rule
            rule = Rule(head, body)
            self.engine.kb.add_rule(rule)
            
            return {
                "success": True,
                "message": f"Added rule: {head} :- {body}"
            }
        
        elif tool_name == "dreamlog_explain":
            query_str = arguments["query"]
            
            # Parse query
            if query_str.startswith("("):
                query_term = parse_s_expression(query_str)
            else:
                query_term = parse_prefix_notation(json.loads(query_str))
            
            # Get explanation (trace unification)
            from dreamlog.unification import unify
            
            explanation = {
                "query": str(query_term),
                "facts_checked": [],
                "rules_checked": [],
                "unification_trace": []
            }
            
            # Check against facts
            for fact in self.engine.kb.facts:
                result = unify(query_term, fact.term)
                if result:
                    explanation["facts_checked"].append({
                        "fact": str(fact.term),
                        "unified": True,
                        "bindings": {k: str(v) for k, v in result.items()}
                    })

            # Check against rules
            for rule in self.engine.kb.rules:
                result = unify(query_term, rule.head)
                if result:
                    explanation["rules_checked"].append({
                        "rule": f"{rule.head} :- {rule.body}",
                        "head_unified": True,
                        "bindings": {k: str(v) for k, v in result.items()}
                    })
            
            return explanation
        
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def handle_resource_read(self, uri: str) -> Dict[str, Any]:
        """Handle MCP resource reads"""
        
        if uri == "dreamlog://kb/current":
            return {
                "content": self.engine.save_to_prefix(),
                "mime_type": "application/json"
            }
        
        elif uri == "dreamlog://kb/stats":
            return {
                "content": json.dumps({
                    "num_facts": len(self.engine.kb.facts),
                    "num_rules": len(self.engine.kb.rules),
                    "functors": list(self.engine.kb._fact_index.keys() | self.engine.kb._rule_index.keys())
                }),
                "mime_type": "application/json"
            }
        
        else:
            return {"error": f"Unknown resource: {uri}"}
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get MCP server information"""
        return {
            "name": "DreamLog MCP Server",
            "version": "1.0.0",
            "protocol_version": "1.0",
            "capabilities": {
                "tools": [asdict(tool) for tool in self.tools],
                "resources": {
                    "list": [asdict(resource) for resource in self.resources],
                    "subscribe": False  # Could add real-time updates later
                },
                "prompts": [],  # Could add prompt templates
                "logging": {}
            }
        }


async def run_mcp_server(host: str = "localhost", port: int = 8765, kb_path: Optional[str] = None, config: Optional[DreamLogConfig] = None):
    """Run the MCP server"""
    mcp_server = DreamLogMCPServer(kb_path=kb_path, config=config)
    
    async def handle_client(reader, writer):
        """Handle MCP client connections"""
        try:
            while True:
                # Read request
                data = await reader.read(4096)
                if not data:
                    break
                
                request = json.loads(data.decode())
                
                # Handle different request types
                if request.get("method") == "initialize":
                    response = mcp_server.get_server_info()
                elif request.get("method") == "tools/call":
                    response = await mcp_server.handle_tool_call(
                        request["params"]["name"],
                        request["params"]["arguments"]
                    )
                elif request.get("method") == "resources/read":
                    response = await mcp_server.handle_resource_read(
                        request["params"]["uri"]
                    )
                else:
                    response = {"error": "Unknown method"}
                
                # Send response
                writer.write(json.dumps(response).encode())
                await writer.drain()
        
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    # Start server
    server = await asyncio.start_server(handle_client, host, port)
    print(f"DreamLog MCP Server running on {host}:{port}")
    
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamLog MCP Server")
    parser.add_argument("--kb", help="Path to knowledge base file")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--config", help="Path to DreamLog config file")

    args = parser.parse_args()

    # Load config if specified
    config = None
    if args.config:
        config = DreamLogConfig.load(args.config)

    asyncio.run(run_mcp_server(
        host=args.host,
        port=args.port,
        kb_path=args.kb,
        config=config
    ))