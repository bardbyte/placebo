#!/usr/bin/env python3
"""
Demo: Natural Language to SQL using Looker MCP and Gemini.

This example demonstrates:
1. Loading configuration from config.yaml and .env
2. Authenticating via IdaaS
3. Connecting to Looker MCP server
4. Using Gemini to translate natural language to SQL
5. Displaying the agent's thinking process

Usage:
    python examples/looker_nl_to_sql.py "Show me total sales by region for last month"
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumi_llm.config import load_settings
from lumi_llm.auth import IdaaSClient
from lumi_llm.providers import GeminiProvider
from lumi_llm.mcp import MCPClient
from lumi_llm.agents import (
    create_tool_agent,
    ConsoleThinkingCallback,
    run_agent,
)


SYSTEM_PROMPT = """You are a data analyst assistant that helps users query data using Looker.

## Your Workflow

When a user asks a question about data, follow this systematic approach:

### Step 1: Discover the Data Model
- First, call `get_models` to see what Looker models are available
- Identify which model is most relevant to the user's question

### Step 2: Explore the Model Structure
- Call `get_explores` on the relevant model to see available explores
- Choose the explore that best matches the user's data needs

### Step 3: Understand Available Fields
- Call `get_dimensions` on the chosen explore to see what dimensions exist
- Call `get_measures` to see what metrics/aggregations are available
- Identify which dimensions and measures map to the user's question

### Step 4: Build and Execute the Query
- Based on your understanding, construct the query
- Use `query` or `query_sql` to execute it
- If the user wants SQL specifically, use `query_sql` and show them the generated SQL

### Step 5: Present Results
- Analyze the query results
- Present findings clearly with context
- If relevant, suggest follow-up analyses

## Important Guidelines
- Always explain your reasoning at each step so the user can follow your thinking
- If you're unsure which model or explore to use, briefly explain your choices
- When showing SQL, explain what each part does
- If a query returns no results or errors, explain why and suggest alternatives

Be methodical and transparent in your exploration of the data model."""


async def main(query: str | None = None):
    """Run the NL to SQL demo."""
    print("=" * 60)
    print("Lumi LLM - Natural Language to SQL Demo")
    print("=" * 60)

    # Load configuration
    print("\n[1/5] Loading configuration...")
    try:
        config_path = Path(__file__).parent.parent / "config.yaml"
        env_path = Path(__file__).parent.parent / ".env"
        settings = load_settings(config_path=config_path, env_path=env_path)
        print("      Configuration loaded successfully")
    except FileNotFoundError as e:
        print(f"      Error: {e}")
        print("      Make sure config.yaml and .env files exist")
        return

    # Initialize IdaaS client
    print("\n[2/5] Initializing authentication...")
    auth_client = IdaaSClient(settings.idaas)
    print("      IdaaS client initialized")

    # Initialize Gemini provider
    print("\n[3/5] Initializing Gemini provider...")
    gemini_config = settings.llm.providers.get("gemini")
    if not gemini_config:
        print("      Error: Gemini provider not configured")
        return
    llm_provider = GeminiProvider(gemini_config, auth_client)
    print("      Gemini provider ready")

    # Connect to Looker MCP
    print("\n[4/5] Connecting to Looker MCP server...")
    looker_config = settings.mcp.servers.get("looker")
    if not looker_config:
        print("      Error: Looker MCP server not configured")
        return

    mcp_client = MCPClient(looker_config)
    try:
        await mcp_client.connect()
        print(f"      Connected! Found {len(mcp_client.tools)} tools:")
        for tool in mcp_client.tools:
            print(f"        - {tool.name}: {tool.description[:50]}...")
    except Exception as e:
        print(f"      Error connecting to MCP server: {e}")
        print("      Make sure the Looker MCP server is running at", looker_config.url)
        return

    # Create agent with thinking callback
    print("\n[5/5] Creating agent...")
    thinking_callback = ConsoleThinkingCallback(use_rich=True)
    agent = create_tool_agent(
        llm_provider=llm_provider,
        mcp_client=mcp_client,
        system_prompt=SYSTEM_PROMPT,
        max_tool_calls=15,  # Allow enough calls for: models → explores → dimensions → measures → query
        thinking_callback=thinking_callback,
    )
    print("      Agent ready!")

    # Get query
    if not query:
        print("\n" + "=" * 60)
        query = input("Enter your question: ").strip()
        if not query:
            query = "What tables are available in the data model?"

    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print("\nAgent Processing...\n")

    # Run the agent
    try:
        result = await run_agent(agent, query, thinking_callback)

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(result.get("final_answer", "No answer generated"))

    except Exception as e:
        print(f"\nError running agent: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    await mcp_client.disconnect()
    print("\n[Done] Disconnected from MCP server")


def run_sync(query: str | None = None):
    """Synchronous entry point."""
    asyncio.run(main(query))


if __name__ == "__main__":
    # Get query from command line if provided
    user_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    run_sync(user_query)
