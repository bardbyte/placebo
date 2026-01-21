#!/usr/bin/env python3
"""
Mock MCP Server for Testing

A simple MCP-compatible server that implements the Streamable HTTP transport.
Supports both application/json and text/event-stream responses.

This allows testing the MCP client without needing actual Looker credentials.

Usage:
    python examples/mock_mcp_server.py

Then test with:
    python examples/test_mcp_client.py
"""

import json
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any


# Mock tools that mimic Looker tools
MOCK_TOOLS = [
    {
        "name": "get-models",
        "description": "Retrieves all LookML models available in the Looker instance.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get-explores",
        "description": "Retrieves all explores within a specified LookML model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "The name of the model"
                }
            },
            "required": ["model"]
        }
    },
    {
        "name": "multiply",
        "description": "Multiply two numbers together.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    }
]

# Mock data for tool calls
MOCK_MODELS = ["ecommerce", "marketing", "finance"]
MOCK_EXPLORES = {
    "ecommerce": ["order_items", "users", "products"],
    "marketing": ["campaigns", "conversions"],
    "finance": ["transactions", "accounts"]
}


class MCPHandler(BaseHTTPRequestHandler):
    """Handler for MCP Streamable HTTP requests."""

    def _send_json_response(self, data: dict, status: int = 200):
        """Send a JSON response."""
        response = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)

    def _handle_jsonrpc(self, request: dict) -> dict:
        """Handle a JSON-RPC request and return a response."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        print(f"  [MCP] Method: {method}")
        print(f"  [MCP] Params: {params}")

        result = {}

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True}
                },
                "serverInfo": {
                    "name": "mock-mcp-server",
                    "version": "1.0.0"
                }
            }
        elif method == "notifications/initialized":
            # No response needed for notifications
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        elif method == "tools/list":
            result = {"tools": MOCK_TOOLS}
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self._execute_tool(tool_name, arguments)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    def _execute_tool(self, name: str, arguments: dict) -> dict:
        """Execute a mock tool."""
        print(f"  [MCP] Executing tool: {name} with args: {arguments}")

        if name == "get-models":
            content = json.dumps(MOCK_MODELS, indent=2)
        elif name == "get-explores":
            model = arguments.get("model", "")
            explores = MOCK_EXPLORES.get(model, [])
            content = json.dumps(explores, indent=2)
        elif name == "multiply":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            content = str(a * b)
        else:
            content = f"Unknown tool: {name}"

        return {
            "content": [{"type": "text", "text": content}],
            "isError": False
        }

    def do_POST(self):
        """Handle POST requests (JSON-RPC)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            request = json.loads(body)
            print(f"\n[MCP] Received request: {request.get('method', 'unknown')}")

            response = self._handle_jsonrpc(request)
            self._send_json_response(response)

        except json.JSONDecodeError as e:
            self._send_json_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"}
            }, status=400)

    def do_GET(self):
        """Handle GET requests (health check)."""
        if self.path == "/health":
            self._send_json_response({"status": "ok"})
        else:
            self._send_json_response({
                "message": "Mock MCP Server",
                "endpoints": {
                    "POST /mcp": "JSON-RPC endpoint",
                    "GET /health": "Health check"
                }
            })

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[HTTP] {args[0]}")


def run_server(host: str = "127.0.0.1", port: int = 5001):
    """Run the mock MCP server."""
    server = HTTPServer((host, port), MCPHandler)
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              MOCK MCP SERVER                                   ║
║                                                                 ║
║  Listening on: http://{host}:{port}/mcp                         ║
║                                                                 ║
║  Available tools:                                               ║
║  - get-models: List mock LookML models                         ║
║  - get-explores: List explores for a model                     ║
║  - multiply: Multiply two numbers                              ║
║                                                                 ║
║  Press Ctrl+C to stop                                          ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    run_server()
