"""
Base LLM provider interface.
All providers (Gemini, OpenAI, Azure) implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """
    Standardized response from LLM providers.
    Provides common fields while preserving the raw response.
    """
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass
class LLMChunk:
    """A chunk of streamed LLM response."""
    content: str | None = None
    tool_call_delta: dict[str, Any] | None = None
    finish_reason: str | None = None


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Implementations must provide sync, async, and streaming methods.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response asynchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            **kwargs: Provider-specific parameters.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    @abstractmethod
    def generate_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response synchronously.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            **kwargs: Provider-specific parameters.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:
        """
        Generate a streaming response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            **kwargs: Provider-specific parameters.

        Yields:
            LLMChunk objects as they arrive.
        """
        pass
