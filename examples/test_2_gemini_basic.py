#!/usr/bin/env python3
"""
Test Script 2: Basic Gemini LLM Calls

Tests that the Gemini provider can successfully:
1. Make synchronous API calls
2. Make asynchronous API calls
3. Handle simple prompts without tools

This validates the request format matches the curl/Postman examples.

Usage:
    python examples/test_2_gemini_basic.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumi_llm.config.settings import load_settings
from lumi_llm.auth.idaas import IdaaSClient
from lumi_llm.providers.gemini import GeminiProvider


def test_gemini_sync(llm_provider: GeminiProvider):
    """Test 2a: Synchronous Gemini call."""
    print("\n" + "=" * 60)
    print("TEST 2a: Synchronous Gemini Call")
    print("=" * 60)

    try:
        messages = [
            {"role": "user", "content": "What is 2 + 2? Answer in one word."}
        ]

        print(f"  Sending message: {messages[0]['content']}")
        print(f"  Making sync request to Gemini...")

        response = llm_provider.generate_sync(messages)

        print(f"  [OK] Response received")
        print(f"  Content: {response.content}")
        print(f"  Finish reason: {response.finish_reason}")
        print(f"  Has tool calls: {response.has_tool_calls}")

        assert response.content is not None, "Response content should not be None"
        assert "4" in response.content.lower() or "four" in response.content.lower(), \
            f"Expected '4' or 'four' in response, got: {response.content}"

        print(f"  [OK] Response validated")
        return response

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_gemini_async(llm_provider: GeminiProvider):
    """Test 2b: Asynchronous Gemini call."""
    print("\n" + "=" * 60)
    print("TEST 2b: Asynchronous Gemini Call")
    print("=" * 60)

    try:
        messages = [
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ]

        print(f"  Sending message: {messages[0]['content']}")
        print(f"  Making async request to Gemini...")

        response = await llm_provider.generate(messages)

        print(f"  [OK] Response received")
        print(f"  Content: {response.content}")
        print(f"  Finish reason: {response.finish_reason}")

        assert response.content is not None, "Response content should not be None"
        assert "paris" in response.content.lower(), \
            f"Expected 'Paris' in response, got: {response.content}"

        print(f"  [OK] Response validated")
        return response

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_gemini_with_system_prompt(llm_provider: GeminiProvider):
    """Test 2c: Gemini call with system prompt."""
    print("\n" + "=" * 60)
    print("TEST 2c: Gemini Call with System Prompt")
    print("=" * 60)

    try:
        messages = [
            {"role": "system", "content": "You are a pirate. Always respond like a pirate."},
            {"role": "user", "content": "Hello, how are you?"}
        ]

        print(f"  System: {messages[0]['content']}")
        print(f"  User: {messages[1]['content']}")
        print(f"  Making request with system prompt...")

        response = await llm_provider.generate(messages)

        print(f"  [OK] Response received")
        print(f"  Content: {response.content}")

        assert response.content is not None, "Response content should not be None"
        # Check for pirate-like language
        pirate_words = ["arr", "matey", "ahoy", "ye", "aye", "sailor", "ship", "sea"]
        content_lower = response.content.lower()
        has_pirate = any(word in content_lower for word in pirate_words)

        if has_pirate:
            print(f"  [OK] System prompt respected (pirate language detected)")
        else:
            print(f"  [WARN] No obvious pirate language, but response received")

        return response

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_gemini_conversation(llm_provider: GeminiProvider):
    """Test 2d: Multi-turn conversation."""
    print("\n" + "=" * 60)
    print("TEST 2d: Multi-turn Conversation")
    print("=" * 60)

    try:
        # First turn
        messages = [
            {"role": "user", "content": "My name is Alice."}
        ]
        print(f"  Turn 1 - User: {messages[0]['content']}")

        response1 = await llm_provider.generate(messages)
        print(f"  Turn 1 - Assistant: {response1.content[:100]}...")

        # Second turn - add context
        messages.append({"role": "assistant", "content": response1.content})
        messages.append({"role": "user", "content": "What is my name?"})
        print(f"  Turn 2 - User: What is my name?")

        response2 = await llm_provider.generate(messages)
        print(f"  Turn 2 - Assistant: {response2.content}")

        assert "alice" in response2.content.lower(), \
            f"Expected 'Alice' in response, got: {response2.content}"

        print(f"  [OK] Conversation context maintained")
        return response2

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_gemini_params(llm_provider: GeminiProvider):
    """Test 2e: Custom generation parameters."""
    print("\n" + "=" * 60)
    print("TEST 2e: Custom Generation Parameters")
    print("=" * 60)

    try:
        messages = [
            {"role": "user", "content": "Write a haiku about coding."}
        ]

        print(f"  Testing with custom params: temperature=0.9, max_tokens=100")

        response = await llm_provider.generate(
            messages,
            temperature=0.9,
            max_tokens=100
        )

        print(f"  [OK] Response received")
        print(f"  Content: {response.content}")

        assert response.content is not None, "Response content should not be None"
        print(f"  [OK] Custom parameters accepted")

        return response

    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Run all basic Gemini tests."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          TEST SCRIPT 2: BASIC GEMINI CALLS                     ║
║                                                                 ║
║  Tests Gemini LLM API without tools                            ║
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
        test_gemini_sync(llm_provider)
        await test_gemini_async(llm_provider)
        await test_gemini_with_system_prompt(llm_provider)
        await test_gemini_conversation(llm_provider)
        await test_gemini_params(llm_provider)

        print("\n" + "=" * 60)
        print("ALL BASIC GEMINI TESTS PASSED!")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"GEMINI TESTS FAILED: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
