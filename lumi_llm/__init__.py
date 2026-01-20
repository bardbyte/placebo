"""
Lumi LLM - A reusable package for LLM access with MCP tool binding.
"""

from lumi_llm.config.settings import Settings, load_settings
from lumi_llm.auth.idaas import IdaaSClient
from lumi_llm.providers.gemini import GeminiProvider
from lumi_llm.mcp.client import MCPClient
from lumi_llm.agents.tool_agent import create_tool_agent, ThinkingCallback

__version__ = "0.1.0"

__all__ = [
    "Settings",
    "load_settings",
    "IdaaSClient",
    "GeminiProvider",
    "MCPClient",
    "create_tool_agent",
    "ThinkingCallback",
]
