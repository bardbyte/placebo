#!/usr/bin/env python3
"""
Test script to verify the lumi-llm package setup.
Validates configuration loading and module imports without requiring
external services (IdaaS, MCP server).

Usage:
    python examples/test_setup.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("[1/4] Testing imports...")

    try:
        from lumi_llm.config import Settings, load_settings
        from lumi_llm.auth import IdaaSClient
        from lumi_llm.providers import GeminiProvider, BaseLLMProvider
        from lumi_llm.mcp import MCPClient
        from lumi_llm.agents import create_tool_agent, ConsoleThinkingCallback
        print("      All imports successful!")
        return True
    except ImportError as e:
        print(f"      Import error: {e}")
        print("      Run: pip install -r requirements.txt")
        return False


def test_config_structure():
    """Test configuration structure without loading files."""
    print("\n[2/4] Testing config structure...")

    from lumi_llm.config.settings import (
        IdaaSConfig,
        LLMProviderConfig,
        LLMConfig,
        MCPServerConfig,
        MCPConfig,
        Settings,
    )

    # Test with minimal config
    try:
        idaas = IdaaSConfig(
            id="test_id",
            url="https://example.com/token",
            secret="test_secret",
        )

        provider = LLMProviderConfig(
            url="https://example.com/llm",
            scope=["test_scope"],
        )

        llm = LLMConfig(
            default_provider="gemini",
            providers={"gemini": provider},
        )

        mcp_server = MCPServerConfig(
            url="http://localhost:5000/mcp",
            transport="sse",
        )

        mcp = MCPConfig(servers={"looker": mcp_server})

        settings = Settings(idaas=idaas, llm=llm, mcp=mcp)

        print("      Config validation successful!")
        print(f"      IdaaS URL: {settings.idaas.url}")
        print(f"      LLM Provider: {settings.llm.default_provider}")
        print(f"      MCP Servers: {list(settings.mcp.servers.keys())}")
        return True
    except Exception as e:
        print(f"      Config error: {e}")
        return False


def test_config_file():
    """Test loading actual config file."""
    print("\n[3/4] Testing config file loading...")

    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        print(f"      Config file not found: {config_path}")
        return False

    try:
        # Just test YAML parsing, not env substitution
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print("      Config file parsed successfully!")
        print(f"      Sections: {list(config.keys())}")
        return True
    except Exception as e:
        print(f"      Error parsing config: {e}")
        return False


def test_thinking_callback():
    """Test the thinking callback system."""
    print("\n[4/4] Testing thinking callback...")

    from lumi_llm.agents.tool_agent import (
        ThinkingEvent,
        ThinkingType,
        ConsoleThinkingCallback,
    )

    callback = ConsoleThinkingCallback(use_rich=False)

    events = [
        ThinkingEvent(
            type=ThinkingType.REASONING,
            content="I need to analyze the user's question...",
        ),
        ThinkingEvent(
            type=ThinkingType.TOOL_CALL,
            content='{"query": "SELECT * FROM sales"}',
            metadata={"tool_name": "run_query"},
        ),
        ThinkingEvent(
            type=ThinkingType.TOOL_RESULT,
            content="Found 100 rows",
            metadata={"tool_name": "run_query"},
        ),
        ThinkingEvent(
            type=ThinkingType.FINAL_ANSWER,
            content="Based on the data, here are the results...",
        ),
    ]

    print("      Simulating thinking events:\n")
    for event in events:
        callback.on_thinking(event)

    print("\n      Thinking callback test complete!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Lumi LLM - Setup Verification")
    print("=" * 60)

    results = [
        ("Imports", test_imports()),
        ("Config Structure", test_config_structure()),
        ("Config File", test_config_file()),
        ("Thinking Callback", test_thinking_callback()),
    ]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and fill in your credentials")
        print("  2. Start the Looker MCP server")
        print("  3. Run: python examples/looker_nl_to_sql.py")
    else:
        print("\nSome tests failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
