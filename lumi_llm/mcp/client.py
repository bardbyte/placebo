"""
MCP (Model Context Protocol) client for connecting to tool servers.
Uses SSE transport to communicate with MCP servers like Looker Toolbox.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import httpx
from httpx_sse import aconnect_sse, connect_sse

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
    Supports SSE transport for real-time communication.
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

    async def connect(self) -> None:
        """
        Connect to the MCP server and initialize the session.
        Discovers available tools.
        """
        # For SSE transport, we establish connection and get session info
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Initial SSE connection to get endpoints
            async with aconnect_sse(
                client, "GET", self.config.url
            ) as event_source:
                async for event in event_source.aiter_sse():
                    if event.event == "endpoint":
                        # Parse endpoint info
                        data = json.loads(event.data)
                        self._message_endpoint = data.get("url")
                        self._session_id = data.get("sessionId")
                        break

            # If we got an endpoint, initialize and list tools
            if self._message_endpoint:
                await self._initialize()
                await self._list_tools()

    def connect_sync(self) -> None:
        """Connect synchronously."""
        with httpx.Client(timeout=30.0) as client:
            with connect_sse(client, "GET", self.config.url) as event_source:
                for event in event_source.iter_sse():
                    if event.event == "endpoint":
                        data = json.loads(event.data)
                        self._message_endpoint = data.get("url")
                        self._session_id = data.get("sessionId")
                        break

            if self._message_endpoint:
                self._initialize_sync()
                self._list_tools_sync()

    async def _send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON-RPC request to the MCP server."""
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

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self._message_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise RuntimeError(f"MCP error: {result['error']}")

            return result.get("result", {})

    def _send_request_sync(
        self, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a JSON-RPC request synchronously."""
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

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                self._message_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                raise RuntimeError(f"MCP error: {result['error']}")

            return result.get("result", {})

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
