"""
LangGraph-based tool agent with thinking display.
Implements a ReAct-style agent that shows its reasoning process.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Annotated, TypedDict, Literal
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from lumi_llm.providers.base import BaseLLMProvider, LLMResponse, ToolCall
from lumi_llm.mcp.client import MCPClient
from lumi_llm.mcp.tool_converter import convert_mcp_tools_to_openai


class ThinkingType(str, Enum):
    """Types of thinking events."""
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class ThinkingEvent:
    """Represents a thinking event from the agent."""
    type: ThinkingType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ThinkingCallback(ABC):
    """Abstract callback for receiving thinking events."""

    @abstractmethod
    def on_thinking(self, event: ThinkingEvent) -> None:
        """Called when a thinking event occurs."""
        pass


class ConsoleThinkingCallback(ThinkingCallback):
    """Callback that prints thinking events to console with formatting."""

    def __init__(self, use_rich: bool = True):
        """
        Initialize console callback.

        Args:
            use_rich: Whether to use rich library for formatting.
        """
        self.use_rich = use_rich
        self._console = None

        if use_rich:
            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.markdown import Markdown
                self._console = Console()
            except ImportError:
                self.use_rich = False

    def on_thinking(self, event: ThinkingEvent) -> None:
        """Print thinking event to console."""
        if self.use_rich and self._console:
            self._print_rich(event)
        else:
            self._print_plain(event)

    def _print_rich(self, event: ThinkingEvent) -> None:
        """Print with rich formatting."""
        from rich.panel import Panel
        from rich.markdown import Markdown

        styles = {
            ThinkingType.REASONING: ("bold blue", "Thinking"),
            ThinkingType.TOOL_CALL: ("bold yellow", "Tool Call"),
            ThinkingType.TOOL_RESULT: ("bold green", "Tool Result"),
            ThinkingType.FINAL_ANSWER: ("bold cyan", "Answer"),
            ThinkingType.ERROR: ("bold red", "Error"),
        }

        style, title = styles.get(event.type, ("white", "Event"))

        if event.type == ThinkingType.TOOL_CALL:
            tool_name = event.metadata.get("tool_name", "unknown")
            title = f"Tool Call: {tool_name}"

        self._console.print(Panel(
            Markdown(event.content) if event.type in [ThinkingType.REASONING, ThinkingType.FINAL_ANSWER] else event.content,
            title=title,
            style=style,
            expand=False,
        ))

    def _print_plain(self, event: ThinkingEvent) -> None:
        """Print without rich formatting."""
        prefixes = {
            ThinkingType.REASONING: "[THINKING]",
            ThinkingType.TOOL_CALL: "[TOOL CALL]",
            ThinkingType.TOOL_RESULT: "[TOOL RESULT]",
            ThinkingType.FINAL_ANSWER: "[ANSWER]",
            ThinkingType.ERROR: "[ERROR]",
        }

        prefix = prefixes.get(event.type, "[EVENT]")

        if event.type == ThinkingType.TOOL_CALL:
            tool_name = event.metadata.get("tool_name", "unknown")
            prefix = f"[TOOL CALL: {tool_name}]"

        print(f"\n{prefix}")
        print("-" * 40)
        print(event.content)
        print("-" * 40)


class AgentState(TypedDict):
    """State for the tool agent."""
    messages: Annotated[list[dict[str, Any]], add_messages]
    thinking_events: list[ThinkingEvent]
    tool_calls_made: int
    final_answer: str | None


def create_tool_agent(
    llm_provider: BaseLLMProvider,
    mcp_client: MCPClient,
    system_prompt: str | None = None,
    max_tool_calls: int = 10,
    thinking_callback: ThinkingCallback | None = None,
):
    """
    Create a LangGraph tool agent.

    Args:
        llm_provider: The LLM provider to use.
        mcp_client: Connected MCP client with tools.
        system_prompt: Optional system prompt for the agent.
        max_tool_calls: Maximum number of tool calls before stopping.
        thinking_callback: Optional callback for thinking events.

    Returns:
        Compiled LangGraph that can be invoked.
    """
    # Get tools in OpenAI format (works for Gemini too)
    tools = convert_mcp_tools_to_openai(mcp_client.tools)

    default_system_prompt = """You are a helpful assistant that can query data using available tools.

When answering questions:
1. Think about what information you need
2. Use the available tools to get that information
3. Analyze the results
4. Provide a clear, helpful answer

Always explain your reasoning process."""

    final_system_prompt = system_prompt or default_system_prompt

    def emit_thinking(state: AgentState, event: ThinkingEvent) -> None:
        """Emit a thinking event."""
        state["thinking_events"].append(event)
        if thinking_callback:
            thinking_callback.on_thinking(event)

    async def call_llm(state: AgentState) -> dict:
        """Call the LLM with current messages."""
        messages = state["messages"]

        # Add system message if not present
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": final_system_prompt}] + messages

        response = await llm_provider.generate(messages, tools=tools)

        # Build new message
        new_message = {"role": "assistant", "content": response.content}

        if response.has_tool_calls:
            new_message["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]

            # Emit thinking about tool calls
            for tc in response.tool_calls:
                emit_thinking(state, ThinkingEvent(
                    type=ThinkingType.TOOL_CALL,
                    content=f"Calling tool with arguments:\n```json\n{tc.arguments}\n```",
                    metadata={"tool_name": tc.name, "arguments": tc.arguments}
                ))
        elif response.content:
            # Emit reasoning/final answer
            emit_thinking(state, ThinkingEvent(
                type=ThinkingType.REASONING,
                content=response.content,
            ))

        return {
            "messages": [new_message],
            "thinking_events": state["thinking_events"],
        }

    async def execute_tools(state: AgentState) -> dict:
        """Execute tool calls from the last message."""
        last_message = state["messages"][-1]
        tool_calls = last_message.get("tool_calls", [])

        new_messages = []
        tool_count = state["tool_calls_made"]

        for tc in tool_calls:
            tool_name = tc["name"]
            arguments = tc["arguments"]

            result = await mcp_client.call_tool(tool_name, arguments)

            result_content = str(result.content)
            if result.is_error:
                result_content = f"Error: {result_content}"
                emit_thinking(state, ThinkingEvent(
                    type=ThinkingType.ERROR,
                    content=result_content,
                    metadata={"tool_name": tool_name}
                ))
            else:
                emit_thinking(state, ThinkingEvent(
                    type=ThinkingType.TOOL_RESULT,
                    content=result_content[:500] + "..." if len(result_content) > 500 else result_content,
                    metadata={"tool_name": tool_name}
                ))

            new_messages.append({
                "role": "tool",
                "name": tool_name,
                "content": result_content,
                "tool_call_id": tc["id"],
            })

            tool_count += 1

        return {
            "messages": new_messages,
            "tool_calls_made": tool_count,
            "thinking_events": state["thinking_events"],
        }

    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Decide whether to continue with tools or end."""
        last_message = state["messages"][-1]

        # Check if we've exceeded max tool calls
        if state["tool_calls_made"] >= max_tool_calls:
            return "end"

        # Check if the last message has tool calls
        if last_message.get("tool_calls"):
            return "tools"

        return "end"

    def finalize(state: AgentState) -> dict:
        """Finalize the agent response."""
        last_message = state["messages"][-1]
        final_answer = last_message.get("content", "")

        if final_answer:
            emit_thinking(state, ThinkingEvent(
                type=ThinkingType.FINAL_ANSWER,
                content=final_answer,
            ))

        return {
            "final_answer": final_answer,
            "thinking_events": state["thinking_events"],
        }

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tools", execute_tools)
    graph.add_node("finalize", finalize)

    # Add edges
    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": "finalize",
        }
    )
    graph.add_edge("tools", "llm")
    graph.add_edge("finalize", END)

    return graph.compile()


async def run_agent(
    agent,
    query: str,
    thinking_callback: ThinkingCallback | None = None,
) -> dict:
    """
    Run the agent with a user query.

    Args:
        agent: Compiled LangGraph agent.
        query: User's question.
        thinking_callback: Optional callback for thinking events.

    Returns:
        Final state with answer and thinking events.
    """
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": query}],
        "thinking_events": [],
        "tool_calls_made": 0,
        "final_answer": None,
    }

    result = await agent.ainvoke(initial_state)
    return result
