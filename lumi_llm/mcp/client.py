"""
MCP (Model Context Protocol) client for connecting to tool servers.
Supports Streamable HTTP transport per MCP specification.
Handles both application/json and text/event-stream responses.
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from lumi_llm.config.settings import MCPServerConfig


@dataclass
class MCPTool:
    """Represents an MCP tool definition."""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""
    content: Any
    is_error: bool = False


class MCPClient:
    """
    Client for MCP (Model Context Protocol) servers.
    Supports Streamable HTTP transport with both JSON and SSE responses.
    """

    def __init__(self, config: "MCPServerConfig"):
        """
        Initialize MCP client.

        Args:
            config: MCP server configuration.
        """
        self.config = config
        self._tools: list[MCPTool] = []
        self._session_id: str | None = None
        self._message_endpoint: str | None = None

    def _get_verify_ssl(self) -> bool:
        """Get SSL verification setting from config."""
        return getattr(self.config, 'verify_ssl', True)

    async def connect(self) -> None:
        """
        Connect to the MCP server and initialize the session.
        Discovers available tools.
        """
        # For streamable-http, the message endpoint is the same as the config URL
        self._message_endpoint = self.config.url
        self._session_id = str(uuid.uuid4())

        # Initialize and list tools
        await self._initialize()
        await self._list_tools()

    def connect_sync(self) -> None:
        """Connect synchronously."""
        # For streamable-http, the message endpoint is the same as the config URL
        self._message_endpoint = self.config.url
        self._session_id = str(uuid.uuid4())

        # Initialize and list tools
        self._initialize_sync()
        self._list_tools_sync()

    async def _send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Send a JSON-RPC request to the MCP server.
        Handles both application/json and text/event-stream responses.
        """
        if not self._message_endpoint:
            raise RuntimeError("Not connected to MCP server")

        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            payload["params"] = params

        async with httpx.AsyncClient(
            timeout=60.0,
            verify=self._get_verify_ssl()
        ) as client:
            response = await client.post(
                self._message_endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            if "text/event-stream" in content_type:
                # Handle SSE response - parse events and find the result
                return await self._parse_sse_response_async(client, response)
            else:
                # Handle JSON response (application/json or default)
                result = response.json()

                if "error" in result:
                    raise RuntimeError(f"MCP error: {result['error']}")

                return result.get("result", {})

    async def _parse_sse_response_async(
        self, client: httpx.AsyncClient, initial_response: httpx.Response
    ) -> dict[str, Any]:
        """Parse SSE response stream and extract the JSON-RPC result."""
        # For SSE, we need to iterate through events
        text = initial_response.text
        result = {}

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('data:'):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        data = json.loads(data_str)
                        if "result" in data:
                            result = data.get("result", {})
                        elif "error" in data:
                            raise RuntimeError(f"MCP error: {data['error']}")
                    except json.JSONDecodeError:
                        continue

        return result

    def _send_request_sync(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Send a JSON-RPC request synchronously.
        Handles both application/json and text/event-stream responses.
        """
        if not self._message_endpoint:
            raise RuntimeError("Not connected to MCP server")

        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            payload["params"] = params

        with httpx.Client(
            timeout=60.0,
            verify=self._get_verify_ssl()
        ) as client:
            response = client.post(
                self._message_endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            if "text/event-stream" in content_type:
                # Handle SSE response
                return self._parse_sse_response_sync(response)
            else:
                # Handle JSON response
                result = response.json()

                if "error" in result:
                    raise RuntimeError(f"MCP error: {result['error']}")

                return result.get("result", {})

    def _parse_sse_response_sync(self, response: httpx.Response) -> dict[str, Any]:
        """Parse SSE response and extract the JSON-RPC result."""
        text = response.text
        result = {}

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('data:'):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        data = json.loads(data_str)
                        if "result" in data:
                            result = data.get("result", {})
                        elif "error" in data:
                            raise RuntimeError(f"MCP error: {data['error']}")
                    except json.JSONDecodeError:
                        continue

        return result

    async def _initialize(self) -> None:
        """Initialize the MCP session."""
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "lumi-llm",
                "version": "0.1.0"
            }
        })
        await self._send_request("notifications/initialized")

    def _initialize_sync(self) -> None:
        """Initialize the MCP session synchronously."""
        self._send_request_sync("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "lumi-llm",
                "version": "0.1.0"
            }
        })
        self._send_request_sync("notifications/initialized")

    async def _list_tools(self) -> None:
        """Fetch available tools from the server."""
        result = await self._send_request("tools/list")
        self._tools = [
            MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {})
            )
            for tool in result.get("tools", [])
        ]

    def _list_tools_sync(self) -> None:
        """Fetch available tools synchronously."""
        result = self._send_request_sync("tools/list")
        self._tools = [
            MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {})
            )
            for tool in result.get("tools", [])
        ]

    @property
    def tools(self) -> list[MCPTool]:
        """Get the list of available tools."""
        return self._tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        """
        Call an MCP tool.

        Args:
            name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            MCPToolResult with the tool output.
        """
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

        content = result.get("content", [])
        is_error = result.get("isError", False)

        # Extract text content
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return MCPToolResult(
            content="\n".join(text_parts) if text_parts else content,
            is_error=is_error
        )

    def call_tool_sync(
        self, name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        """Call an MCP tool synchronously."""
        result = self._send_request_sync("tools/call", {
            "name": name,
            "arguments": arguments
        })

        content = result.get("content", [])
        is_error = result.get("isError", False)

        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return MCPToolResult(
            content="\n".join(text_parts) if text_parts else content,
            is_error=is_error
        )

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        self._session_id = None
        self._message_endpoint = None
        self._tools = []
