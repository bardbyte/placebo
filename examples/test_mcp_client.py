#!/usr/bin/env python3
"""
Test Script: MCP Client

Tests that our MCP client can:
1. Connect to an MCP server (Streamable HTTP)
2. List available tools
3. Call tools and get results

Prerequisites:
    Start the mock MCP server first:
    python examples/mock_mcp_server.py

Then run this test:
    python examples/test_mcp_client.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import using importlib to avoid package __init__.py which requires langgraph
import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = Path(__file__).parent.parent / "lumi_llm"
client_module = import_module_from_path("client", base_path / "mcp" / "client.py")
settings_module = import_module_from_path("settings", base_path / "config" / "settings.py")

MCPClient = client_module.MCPClient
MCPServerConfig = settings_module.MCPServerConfig


async def test_mcp_connection():
    """Test 1: Connect to MCP server."""
    print("\n" + "=" * 60)
    print("TEST 1: MCP Connection")
    print("=" * 60)

    config = MCPServerConfig(
        url="http://localhost:5000/mcp",
        transport="streamable-http",
        verify_ssl=False
    )

    print(f"  Connecting to: {config.url}")
    print(f"  Transport: {config.transport}")

    client = MCPClient(config)

    try:
        await client.connect()
        print(f"  [OK] Connected successfully!")
        print(f"  Session ID: {client._session_id}")
        print(f"  Message endpoint: {client._message_endpoint}")
        return client
    except Exception as e:
        print(f"  [FAIL] Connection failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_list_tools(client: MCPClient):
    """Test 2: List available tools."""
    print("\n" + "=" * 60)
    print("TEST 2: List Tools")
    print("=" * 60)

    tools = client.tools

    print(f"  Found {len(tools)} tools:")
    for tool in tools:
        print(f"    - {tool.name}: {tool.description[:50]}...")

    assert len(tools) > 0, "Should have at least one tool"
    print(f"  [OK] Tools listed successfully")

    return tools


async def test_call_tool_get_models(client: MCPClient):
    """Test 3: Call get-models tool."""
    print("\n" + "=" * 60)
    print("TEST 3: Call Tool - get-models")
    print("=" * 60)

    print(f"  Calling get-models...")

    result = await client.call_tool("get-models", {})

    print(f"  [OK] Tool executed")
    print(f"  Is error: {result.is_error}")
    print(f"  Content: {result.content}")

    assert not result.is_error, "Tool call should not return error"
    print(f"  [OK] get-models works correctly")

    return result


async def test_call_tool_with_args(client: MCPClient):
    """Test 4: Call tool with arguments."""
    print("\n" + "=" * 60)
    print("TEST 4: Call Tool with Arguments - get-explores")
    print("=" * 60)

    print(f"  Calling get-explores with model='ecommerce'...")

    result = await client.call_tool("get-explores", {"model": "ecommerce"})

    print(f"  [OK] Tool executed")
    print(f"  Is error: {result.is_error}")
    print(f"  Content: {result.content}")

    assert not result.is_error, "Tool call should not return error"
    print(f"  [OK] get-explores works correctly")

    return result


async def test_call_tool_multiply(client: MCPClient):
    """Test 5: Call multiply tool."""
    print("\n" + "=" * 60)
    print("TEST 5: Call Tool - multiply")
    print("=" * 60)

    print(f"  Calling multiply(a=7, b=8)...")

    result = await client.call_tool("multiply", {"a": 7, "b": 8})

    print(f"  [OK] Tool executed")
    print(f"  Is error: {result.is_error}")
    print(f"  Content: {result.content}")

    assert not result.is_error, "Tool call should not return error"
    assert "56" in str(result.content), f"Expected 56, got {result.content}"
    print(f"  [OK] multiply works correctly (7 * 8 = 56)")

    return result


async def test_sync_operations():
    """Test 6: Synchronous operations."""
    print("\n" + "=" * 60)
    print("TEST 6: Synchronous Operations")
    print("=" * 60)

    config = MCPServerConfig(
        url="http://localhost:5000/mcp",
        transport="streamable-http",
        verify_ssl=False
    )

    client = MCPClient(config)

    print(f"  Connecting synchronously...")
    client.connect_sync()
    print(f"  [OK] Connected")

    print(f"  Found {len(client.tools)} tools")

    print(f"  Calling multiply(a=3, b=4) synchronously...")
    result = client.call_tool_sync("multiply", {"a": 3, "b": 4})
    print(f"  Result: {result.content}")

    assert "12" in str(result.content), f"Expected 12, got {result.content}"
    print(f"  [OK] Sync operations work correctly")

    return True


async def main():
    """Run all MCP client tests."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              TEST: MCP CLIENT                                  ║
║                                                                 ║
║  Testing MCP Streamable HTTP client                            ║
║                                                                 ║
║  Make sure mock server is running:                             ║
║  python examples/mock_mcp_server.py                            ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        # Test 1: Connection
        client = await test_mcp_connection()

        # Test 2: List tools
        await test_list_tools(client)

        # Test 3: Call get-models
        await test_call_tool_get_models(client)

        # Test 4: Call get-explores with args
        await test_call_tool_with_args(client)

        # Test 5: Call multiply
        await test_call_tool_multiply(client)

        # Test 6: Sync operations
        await test_sync_operations()

        # Disconnect
        await client.disconnect()

        print("\n" + "=" * 60)
        print("ALL MCP CLIENT TESTS PASSED!")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"MCP CLIENT TESTS FAILED: {e}")
        print("=" * 60)
        print("\nMake sure the mock MCP server is running:")
        print("  python examples/mock_mcp_server.py")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
