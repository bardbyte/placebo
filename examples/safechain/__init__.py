"""
SafeChain MCP Agent Examples.

This package contains examples for using SafeChain's MCPToolAgent
with a ReAct-style orchestration layer for multi-turn reasoning.
"""

from .chat import (
    SafeChainOrchestrator,
    ChatSession,
    ThinkingCallback,
    ConsoleThinkingCallback,
    ThinkingEvent,
    ThinkingType,
)

__all__ = [
    "SafeChainOrchestrator",
    "ChatSession",
    "ThinkingCallback",
    "ConsoleThinkingCallback",
    "ThinkingEvent",
    "ThinkingType",
]
