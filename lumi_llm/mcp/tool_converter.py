"""
Convert MCP tool definitions to LLM provider-specific formats.
"""

from typing import Any

from lumi_llm.mcp.client import MCPTool


def convert_mcp_tools_to_openai(tools: list[MCPTool]) -> list[dict[str, Any]]:
    """
    Convert MCP tools to OpenAI function calling format.

    Args:
        tools: List of MCP tool definitions.

    Returns:
        List of OpenAI-compatible tool definitions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema or {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        for tool in tools
    ]


def convert_mcp_tools_to_gemini(tools: list[MCPTool]) -> list[dict[str, Any]]:
    """
    Convert MCP tools to Gemini function declaration format.

    Args:
        tools: List of MCP tool definitions.

    Returns:
        List of Gemini-compatible function declarations.
    """
    function_declarations = []

    for tool in tools:
        declaration = {
            "name": tool.name,
            "description": tool.description,
        }

        # Gemini expects 'parameters' directly
        if tool.input_schema:
            # Clean up schema for Gemini compatibility
            schema = tool.input_schema.copy()
            # Remove unsupported fields if present
            schema.pop("$schema", None)
            schema.pop("additionalProperties", None)
            declaration["parameters"] = schema

        function_declarations.append(declaration)

    return function_declarations


def create_tool_executor(mcp_client: "MCPClient") -> callable:
    """
    Create a tool executor function that routes calls to MCP.

    Args:
        mcp_client: Connected MCP client.

    Returns:
        Async function that executes tools by name.
    """
    from lumi_llm.mcp.client import MCPClient

    async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        result = await mcp_client.call_tool(name, arguments)
        if result.is_error:
            return f"Error: {result.content}"
        return str(result.content)

    return execute_tool
