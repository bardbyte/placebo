"""
Gemini LLM provider implementation.
Communicates with Gemini API through internal proxy with IdaaS auth.
"""

import json
import uuid
from typing import Any, AsyncIterator, TYPE_CHECKING

import httpx

from lumi_llm.providers.base import BaseLLMProvider, LLMResponse, LLMChunk, ToolCall

if TYPE_CHECKING:
    from lumi_llm.auth.idaas import IdaaSClient
    from lumi_llm.config.settings import LLMConfig


class GeminiProvider(BaseLLMProvider):
    """
    Gemini provider using internal proxy with IdaaS authentication.
    Handles Gemini-native request/response format.
    """

    def __init__(
        self,
        config: "LLMConfig",
        auth_client: "IdaaSClient",
    ):
        """
        Initialize Gemini provider.

        Args:
            config: Provider configuration with URL, scope, etc.
            auth_client: IdaaS client for token acquisition.
        """
        self.config = config
        self.auth_client = auth_client

    def _convert_messages_to_gemini(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict], str | None]:
        """
        Convert standard messages format to Gemini's content format.

        Returns:
            Tuple of (contents list, system instruction or None)
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}] if isinstance(content, str) else content
                })
            elif role == "assistant":
                parts = []
                if content:
                    parts.append({"text": content})
                # Handle tool calls in assistant message
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        parts.append({
                            "functionCall": {
                                "name": tc["name"],
                                "args": tc["arguments"]
                            }
                        })
                contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                # Tool response
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.get("name", ""),
                            "response": {"result": content}
                        }
                    }]
                })

        return contents, system_instruction

    def _convert_tools_to_gemini(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict] | None:
        """Convert standard tool format to Gemini function declarations."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            # Handle both OpenAI-style and raw format
            if "function" in tool:
                func = tool["function"]
            else:
                func = tool

            declaration = {
                "name": func["name"],
                "description": func.get("description", ""),
            }

            # Convert parameters
            if "parameters" in func:
                declaration["parameters"] = func["parameters"]

            function_declarations.append(declaration)

        return [{"functionDeclarations": function_declarations}]

    def _parse_gemini_response(self, response_data: dict) -> LLMResponse:
        """Parse Gemini API response into LLMResponse."""
        candidates = response_data.get("candidates", [])
        if not candidates:
            return LLMResponse(
                content=None,
                finish_reason="error",
                raw_response=response_data
            )

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        finish_reason = candidate.get("finishReason", "STOP")

        text_content = None
        tool_calls = []

        for part in content_parts:
            if "text" in part:
                text_content = (text_content or "") + part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=str(uuid.uuid4()),  # Gemini doesn't provide IDs
                    name=fc["name"],
                    arguments=fc.get("args", {})
                ))

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=response_data
        )

    async def _get_headers(self) -> dict[str, str]:
        """Get request headers with auth token."""
        token = await self.auth_client.get_token(scope=self.config.scope)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }

    def _get_headers_sync(self) -> dict[str, str]:
        """Get request headers with auth token (sync)."""
        token = self.auth_client.get_token_sync(scope=self.config.scope)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        }

    def _build_request_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict:
        """Build the Gemini API request body."""
        contents, system_instruction = self._convert_messages_to_gemini(messages)

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }

        if self.config.top_k:
            body["generationConfig"]["topK"] = kwargs.get("top_k", self.config.top_k)
        if self.config.top_p:
            body["generationConfig"]["topP"] = kwargs.get("top_p", self.config.top_p)

        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        gemini_tools = self._convert_tools_to_gemini(tools)
        if gemini_tools:
            body["tools"] = gemini_tools

        return body

    def _get_verify_ssl(self) -> bool:
        """Get SSL verification setting from config."""
        return getattr(self.config, 'verify_ssl', True)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response asynchronously."""
        headers = await self._get_headers()
        body = self._build_request_body(messages, tools, **kwargs)

        async with httpx.AsyncClient(
            timeout=60.0,
            verify=self._get_verify_ssl()
        ) as client:
            response = await client.post(
                self.config.url,
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            return self._parse_gemini_response(response.json())

    def generate_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response synchronously."""
        headers = self._get_headers_sync()
        body = self._build_request_body(messages, tools, **kwargs)

        with httpx.Client(
            timeout=60.0,
            verify=self._get_verify_ssl()
        ) as client:
            response = client.post(
                self.config.url,
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            return self._parse_gemini_response(response.json())

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming response."""
        headers = await self._get_headers()
        body = self._build_request_body(messages, tools, **kwargs)

        # Use streamGenerateContent endpoint
        stream_url = self.config.url.replace(
            "generateContent", "streamGenerateContent"
        )

        async with httpx.AsyncClient(
            timeout=60.0,
            verify=self._get_verify_ssl()
        ) as client:
            async with client.stream(
                "POST",
                stream_url,
                headers=headers,
                json=body,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # Gemini streams JSON objects
                    try:
                        data = json.loads(line)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    yield LLMChunk(content=part["text"])
                                elif "functionCall" in part:
                                    yield LLMChunk(tool_call_delta=part["functionCall"])

                            finish_reason = candidates[0].get("finishReason")
                            if finish_reason:
                                yield LLMChunk(finish_reason=finish_reason)
                    except json.JSONDecodeError:
                        continue
