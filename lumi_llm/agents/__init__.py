from lumi_llm.agents.tool_agent import (
    create_tool_agent,
    AgentState,
    ThinkingCallback,
    ConsoleThinkingCallback,
    run_agent,
)

try:
    from lumi_llm.agents.mcp_tool_agent import MCPToolAgent
except ImportError:
    MCPToolAgent = None  # Optional — requires langchain

__all__ = [
    "create_tool_agent",
    "AgentState",
    "ThinkingCallback",
    "ConsoleThinkingCallback",
    "run_agent",
    "MCPToolAgent",
]
