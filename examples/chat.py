#!/usr/bin/env python3
"""
Interactive Chat Demo with Looker MCP and Gemini.

An interactive CLI chatbot that:
1. Shows available MCP tools on startup
2. Allows continuous conversation with the agent
3. Displays the agent's thinking process in real-time

Usage:
    python examples/chat.py

Commands:
    /tools    - List available tools
    /clear    - Clear conversation history
    /help     - Show help
    /quit     - Exit the chat
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
from lumi_llm.agents.tool_agent import (
    create_tool_agent,
    ConsoleThinkingCallback,
    ThinkingEvent,
    ThinkingType,
    AgentState,
)


SYSTEM_PROMPT = """You are a data analyst assistant that helps users explore and query data using Looker.

## Your Capabilities
You have access to Looker tools that let you:
- Explore LookML models, projects, and files
- Discover dimensions, measures, and explores
- Run queries and generate SQL
- Work with saved Looks and Dashboards

## Your Workflow

When a user asks about data:

1. **Discover**: Use `get_models` or `get_projects` to understand what's available
2. **Explore**: Use `get_explores`, `get_dimensions`, `get_measures` to understand the data model
3. **Query**: Use `query` or `query_sql` to fetch data
4. **Explain**: Present results clearly with context

When a user asks about LookML files:
- Use `get_projects` to list projects
- Use `get_project_files` to see files in a project
- Use `get_project_file` to read a specific file

## Guidelines
- Always explain your reasoning as you work
- Show SQL when you generate it
- If unsure, ask clarifying questions
- Be concise but thorough"""


class ChatSession:
    """Manages an interactive chat session with the agent."""

    def __init__(
        self,
        llm_provider: GeminiProvider,
        mcp_client: MCPClient,
        thinking_callback: ConsoleThinkingCallback,
    ):
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client
        self.thinking_callback = thinking_callback
        self.conversation_history: list[dict] = []
        self.agent = None
        self._setup_agent()

    def _setup_agent(self):
        """Create the LangGraph agent."""
        self.agent = create_tool_agent(
            llm_provider=self.llm_provider,
            mcp_client=self.mcp_client,
            system_prompt=SYSTEM_PROMPT,
            max_tool_calls=20,
            thinking_callback=self.thinking_callback,
        )

    def show_tools(self):
        """Display available MCP tools."""
        print("\n" + "=" * 60)
        print("AVAILABLE TOOLS")
        print("=" * 60)

        if not self.mcp_client.tools:
            print("No tools available. Is the MCP server running?")
            return

        # Group tools by category
        categories = {
            "Models & Explores": [],
            "Fields": [],
            "Queries": [],
            "Looks & Dashboards": [],
            "Projects & Files": [],
            "Other": [],
        }

        for tool in self.mcp_client.tools:
            name = tool.name.lower()
            if "model" in name or "explore" in name:
                categories["Models & Explores"].append(tool)
            elif "dimension" in name or "measure" in name or "filter" in name or "parameter" in name:
                categories["Fields"].append(tool)
            elif "query" in name:
                categories["Queries"].append(tool)
            elif "look" in name or "dashboard" in name:
                categories["Looks & Dashboards"].append(tool)
            elif "project" in name or "file" in name:
                categories["Projects & Files"].append(tool)
            else:
                categories["Other"].append(tool)

        for category, tools in categories.items():
            if tools:
                print(f"\n{category}:")
                for tool in tools:
                    desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
                    print(f"  - {tool.name}")
                    print(f"    {desc}")

        print("\n" + "=" * 60)
        print(f"Total: {len(self.mcp_client.tools)} tools available")
        print("=" * 60 + "\n")

    def show_help(self):
        """Display help information."""
        print("""
╭─────────────────────────────────────────────────────────────╮
│                     CHAT COMMANDS                           │
├─────────────────────────────────────────────────────────────┤
│  /tools   - List all available MCP tools                    │
│  /clear   - Clear conversation history                      │
│  /help    - Show this help message                          │
│  /quit    - Exit the chat                                   │
├─────────────────────────────────────────────────────────────┤
│                   EXAMPLE QUERIES                           │
├─────────────────────────────────────────────────────────────┤
│  "What tools do you have access to?"                        │
│  "What LookML projects are available?"                      │
│  "Show me the files in the ecommerce project"               │
│  "What models are available?"                               │
│  "Explore the order_items model"                            │
│  "Show me total sales by region for last month"             │
╰─────────────────────────────────────────────────────────────╯
""")

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("\n[Conversation history cleared]\n")

    async def chat(self, user_input: str) -> str:
        """
        Send a message and get a response.

        Args:
            user_input: The user's message.

        Returns:
            The agent's response.
        """
        # Build state with conversation history
        messages = self.conversation_history + [{"role": "user", "content": user_input}]

        initial_state: AgentState = {
            "messages": messages,
            "thinking_events": [],
            "tool_calls_made": 0,
            "final_answer": None,
        }

        print("\n")  # Space before thinking output

        # Run the agent
        result = await self.agent.ainvoke(initial_state)

        # Extract the final answer
        final_answer = result.get("final_answer", "I couldn't generate a response.")

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": final_answer})

        # Keep history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return final_answer


async def main():
    """Run the interactive chat."""
    print("""
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║              LUMI LLM - Interactive Chat                    ║
║                                                             ║
║         Natural Language to SQL with Looker MCP             ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
""")

    # Load configuration
    print("[1/4] Loading configuration...")
    try:
        config_path = Path(__file__).parent.parent / "config.yaml"
        env_path = Path(__file__).parent.parent / ".env"
        settings = load_settings(config_path=config_path, env_path=env_path)
        print("      Configuration loaded")
    except FileNotFoundError as e:
        print(f"      Error: {e}")
        print("      Make sure config.yaml and .env files exist")
        return

    # Initialize IdaaS client
    print("[2/4] Initializing authentication...")
    auth_client = IdaaSClient(settings.idaas)
    print("      IdaaS client ready")

    # Initialize Gemini provider
    print("[3/4] Initializing Gemini provider...")
    gemini_config = settings.llm.providers.get("gemini")
    if not gemini_config:
        print("      Error: Gemini provider not configured")
        return
    llm_provider = GeminiProvider(gemini_config, auth_client)
    print("      Gemini provider ready")

    # Connect to Looker MCP
    print("[4/4] Connecting to MCP server...")
    looker_config = settings.mcp.servers.get("looker")
    if not looker_config:
        print("      Error: Looker MCP server not configured")
        return

    mcp_client = MCPClient(looker_config)
    try:
        await mcp_client.connect()
        print(f"      Connected! Found {len(mcp_client.tools)} tools")
    except Exception as e:
        print(f"      Error connecting to MCP server: {e}")
        print(f"      Make sure the MCP Toolbox is running at {looker_config.url}")
        print("\n      Start it with: ./toolbox --tools-file tools.yaml")
        return

    # Create chat session
    thinking_callback = ConsoleThinkingCallback(use_rich=True)
    session = ChatSession(llm_provider, mcp_client, thinking_callback)

    # Show available tools on startup
    session.show_tools()
    session.show_help()

    # Main chat loop
    print("Type your question or command. Use /quit to exit.\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break
            elif user_input.lower() == "/tools":
                session.show_tools()
                continue
            elif user_input.lower() == "/clear":
                session.clear_history()
                continue
            elif user_input.lower() == "/help":
                session.show_help()
                continue

            # Chat with the agent
            try:
                response = await session.chat(user_input)
                print(f"\n{'─' * 60}")
                print(f"Assistant: {response}")
                print(f"{'─' * 60}\n")
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break

    # Cleanup
    await mcp_client.disconnect()
    print("[Disconnected from MCP server]")


def run():
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
