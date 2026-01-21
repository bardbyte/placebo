"""
MCPToolAgent: LCEL-compatible agent for executing MCP tools based on LLM decisions.

This module provides the MCPToolAgent class, which integrates with LCEL and executes
MCP tools as requested by language model outputs. It supports both synchronous and
asynchronous invocation, error handling, and tool result aggregation.
"""

import logging
import asyncio
from typing import Any, List, Optional, Dict

from langchain.schema import AIMessage
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.tools import BaseTool


class MCPToolAgent(Runnable):
    """
    LCEL-compatible agent that can execute MCP tools based on LLM decisions.

    Attributes:
        model_id (str): The model identifier.
        tools (Dict[str, BaseTool]): Mapping of tool names to tool instances.
    """

    def __init__(self, model_id: str, tools: List[BaseTool], amodel_func=None):
        """
        Initialize the MCPToolAgent.

        Args:
            model_id: The identifier for the LLM model to use.
            tools: List of BaseTool instances available for the agent.
            amodel_func: Optional async function to get the LLM model.
                         Should accept model_id and return an LLM instance.
        """
        self.model_id = model_id
        self.tools = {tool.name: tool for tool in tools}
        self._amodel_func = amodel_func

    async def _get_llm(self):
        """Get the LLM instance, optionally binding tools."""
        if self._amodel_func is None:
            raise ValueError(
                "No amodel_func provided. Pass an async function that returns "
                "an LLM instance when initializing MCPToolAgent."
            )
        llm = await self._amodel_func(self.model_id)
        if hasattr(llm, "bind_tools"):
            llm = llm.bind_tools(list(self.tools.values()))
        return llm

    async def ainvoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> dict:  # pylint: disable=redefined-builtin
        """
        Process input through LLM and execute any requested tools.

        Args:
            input: The input to process (typically a message or prompt).
            config: Optional runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary containing:
                - content: The LLM's response content
                - tool_results: List of tool execution results
        """
        input_ = input
        llm = await self._get_llm()

        response = await llm.ainvoke(input_, config)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return response

        results = []
        for tool_call in response.tool_calls:
            tool_name = (
                tool_call.get("name")
                if isinstance(tool_call, dict)
                else tool_call.name
            )
            tool_args = (
                tool_call.get("args", {})
                if isinstance(tool_call, dict)
                else tool_call.args
            )

            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name].ainvoke(tool_args)
                    results.append({"tool": tool_name, "result": result})
                except Exception as e:  # pylint: disable=broad-except
                    logging.getLogger(__name__).warning(
                        f"Tool '{tool_name}' execution failed: {e}"
                    )
                    results.append({"tool": tool_name, "error": str(e)})
            else:
                logging.getLogger(__name__).warning(
                    f"Unknown tool requested: {tool_name}"
                )

        return {
            "content": (
                response.content
                if isinstance(response, AIMessage)
                else str(response)
            ),
            "tool_results": results,
        }

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> dict:  # pylint: disable=redefined-builtin
        """
        Synchronous wrapper for the asynchronous ainvoke function.

        Args:
            input: The input to process.
            config: Optional runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            The result from ainvoke.
        """
        input_ = input
        return asyncio.run(self.ainvoke(input_, config, **kwargs))

    def _call(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> dict:  # pylint: disable=redefined-builtin
        """Not supported - raises NotImplementedError."""
        raise NotImplementedError(
            "MCPToolAgent does not support hidden `_call`. "
            "Please use a supported method (`invoke`, etc.)."
        )

    def batch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> List[dict]:  # pylint: disable=redefined-builtin
        """Not supported - raises NotImplementedError."""
        raise NotImplementedError(
            "MCPToolAgent does not support `batch`. "
            "Please use a supported method (`invoke`, etc.)."
        )

    async def abatch(
        self,
        inputs: List[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> List[dict]:  # pylint: disable=redefined-builtin
        """Not supported - raises NotImplementedError."""
        raise NotImplementedError(
            "MCPToolAgent does not support `abatch`. "
            "Please use a supported method (`invoke`, etc.)."
        )
