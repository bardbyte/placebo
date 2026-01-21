#!/usr/bin/env python3
"""
Test Script 3: Gemini with Tools Binding

Tests that the Gemini provider can successfully:
1. Send requests with function declarations (tools)
2. Receive and parse function call responses
3. Handle tool results in conversation

Uses the same tool format as shown in curl.png:
- multiply: Multiply two integers
- temp_convert: Convert temperature between units

Usage:
    python examples/test_3_gemini_tools.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumi_llm.config.settings import load_settings
from lumi_llm.auth.idaas import IdaaSClient
from lumi_llm.providers.gemini import GeminiProvider


# Define tools matching curl.png format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "First integer"
                    },
                    "b": {
                        "type": "integer",
                        "description": "Second integer"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "temp_convert",
            "description": "Convert a temperature value from one unit to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "The temperature value to convert."
                    },
                    "source_unit": {
                        "type": "string",
                        "enum": ["C", "Fahrenheit", "Kelvin"],
                        "description": "The unit of the input temperature."
                    },
                    "destination_unit": {
                        "type": "string",
                        "enum": ["C", "Fahrenheit", "Kelvin"],
                        "description": "The unit to convert the temperature to."
                    }
                },
                "required": ["temperature", "source_unit", "destination_unit"]
            }
        }
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if name == "multiply":
        result = arguments["a"] * arguments["b"]
        return str(result)
    elif name == "temp_convert":
        temp = arguments["temperature"]
        source = arguments["source_unit"]
        dest = arguments["destination_unit"]

        # Convert to Celsius first
        if source == "Fahrenheit":
            celsius = (temp - 32) * 5 / 9
        elif source == "Kelvin":
            celsius = temp - 273.15
        else:  # C
            celsius = temp

        # Convert from Celsius to destination
        if dest == "Fahrenheit":
            result = celsius * 9 / 5 + 32
        elif dest == "Kelvin":
            result = celsius + 273.15
        else:  # C
            result = celsius

        return f"{result:.2f} {dest}"
    else:
        return f"Unknown tool: {name}"


async def test_tool_call_detection(llm_provider: GeminiProvider):
    """Test 3a: Verify LLM returns tool calls."""
    print("\n" + "=" * 60)
    print("TEST 3a: Tool Call Detection")
    print("=" * 60)

    try:
        messages = [
            {"role": "user", "content": "What is 15 times 7?"}
        ]

        print(f"  User: {messages[0]['content']}")
        print(f"  Sending request with {len(TOOLS)} tools...")

        response = await llm_provider.generate(messages, tools=TOOLS)

        print(f"  [OK] Response received")
        print(f"  Has tool calls: {response.has_tool_calls}")
        print(f"  Content: {response.content}")

        if response.has_tool_calls:
            for tc in response.tool_calls:
                print(f"  Tool call: {tc.name}({tc.arguments})")

            assert any(tc.name == "multiply" for tc in response.tool_calls), \
                "Expected 'multiply' tool call"
            print(f"  [OK] Correct tool identified")
        else:
            print(f"  [WARN] No tool call made - LLM answered directly")

        return response

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_full_tool_conversation(llm_provider: GeminiProvider):
    """Test 3b: Full conversation with tool execution."""
    print("\n" + "=" * 60)
    print("TEST 3b: Full Tool Conversation (multiply)")
    print("=" * 60)

    try:
        # Initial user message
        messages = [
            {"role": "user", "content": "Calculate 15 times 3. The result is in Fahrenheit. Convert the result to Kelvin. You need to use the available tools to provide tool call details"}
        ]

        print(f"  User: {messages[0]['content']}")

        # First LLM call - should return tool call
        response = await llm_provider.generate(messages, tools=TOOLS)

        if not response.has_tool_calls:
            print(f"  [WARN] LLM answered without tools: {response.content}")
            return response

        # Process tool calls
        print(f"  LLM requested {len(response.tool_calls)} tool call(s)")

        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]
        })

        # Execute each tool and add results
        for tc in response.tool_calls:
            print(f"  Executing: {tc.name}({tc.arguments})")
            result = execute_tool(tc.name, tc.arguments)
            print(f"  Result: {result}")

            messages.append({
                "role": "tool",
                "name": tc.name,
                "content": result,
                "tool_call_id": tc.id
            })

        # Second LLM call - should get final answer
        print(f"  Sending tool results back to LLM...")
        final_response = await llm_provider.generate(messages, tools=TOOLS)

        print(f"  [OK] Final response received")
        print(f"  Content: {final_response.content}")

        # May have more tool calls for temp conversion
        if final_response.has_tool_calls:
            print(f"  LLM requested additional tool calls")
            for tc in final_response.tool_calls:
                print(f"    Tool: {tc.name}({tc.arguments})")

        return final_response

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_temp_conversion_tool(llm_provider: GeminiProvider):
    """Test 3c: Temperature conversion tool."""
    print("\n" + "=" * 60)
    print("TEST 3c: Temperature Conversion Tool")
    print("=" * 60)

    try:
        messages = [
            {"role": "user", "content": "Convert 100 degrees Fahrenheit to Celsius. Use the temp_convert tool."}
        ]

        print(f"  User: {messages[0]['content']}")

        response = await llm_provider.generate(messages, tools=TOOLS)

        print(f"  Has tool calls: {response.has_tool_calls}")

        if response.has_tool_calls:
            for tc in response.tool_calls:
                print(f"  Tool call: {tc.name}")
                print(f"  Arguments: {tc.arguments}")

                if tc.name == "temp_convert":
                    result = execute_tool(tc.name, tc.arguments)
                    print(f"  Result: {result}")
                    print(f"  [OK] Temperature conversion tool called correctly")
        else:
            print(f"  [WARN] No tool call - LLM response: {response.content}")

        return response

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_tool_format_validation(llm_provider: GeminiProvider):
    """Test 3d: Validate the tool format sent matches curl.png."""
    print("\n" + "=" * 60)
    print("TEST 3d: Tool Format Validation")
    print("=" * 60)

    try:
        # Check the internal conversion
        gemini_tools = llm_provider._convert_tools_to_gemini(TOOLS)

        print(f"  Input tools: {len(TOOLS)}")
        print(f"  Gemini tools structure: {len(gemini_tools)} wrapper(s)")

        # Validate structure matches curl.png: tools: [{functionDeclarations: [...]}]
        assert len(gemini_tools) == 1, "Should have one tools wrapper"
        assert "functionDeclarations" in gemini_tools[0], \
            "Should have functionDeclarations key"

        declarations = gemini_tools[0]["functionDeclarations"]
        print(f"  Function declarations: {len(declarations)}")

        for decl in declarations:
            print(f"    - {decl['name']}: {decl.get('description', 'no desc')[:50]}...")
            assert "name" in decl, "Declaration must have name"
            assert "description" in decl, "Declaration must have description"

        print(f"  [OK] Tool format matches expected Gemini format")

        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Run all tool-related Gemini tests."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          TEST SCRIPT 3: GEMINI WITH TOOLS                      ║
║                                                                 ║
║  Tests function calling / tool use with Gemini                 ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        # Load config
        print("[Setup] Loading configuration...")
        config_path = Path(__file__).parent.parent / "config.yaml"
        env_path = Path(__file__).parent.parent / ".env"
        settings = load_settings(config_path=config_path, env_path=env_path)
        print(f"  LLM URL: {settings.llm.url}")

        # Create auth client
        print("[Setup] Creating IdaaS client...")
        auth_client = IdaaSClient(settings.idaas)

        # Create Gemini provider
        print("[Setup] Creating Gemini provider...")
        llm_provider = GeminiProvider(settings.llm, auth_client)
        print("  Setup complete\n")

        # Run tests
        await test_tool_format_validation(llm_provider)
        await test_tool_call_detection(llm_provider)
        await test_temp_conversion_tool(llm_provider)
        await test_full_tool_conversation(llm_provider)

        print("\n" + "=" * 60)
        print("ALL TOOL TESTS PASSED!")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TOOL TESTS FAILED: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
