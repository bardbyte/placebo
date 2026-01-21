#!/usr/bin/env python3
"""
Master Test Runner

Runs all test scripts in sequence:
1. test_1_token.py - Token generation
2. test_2_gemini_basic.py - Basic Gemini calls
3. test_3_gemini_tools.py - Gemini with tools

Usage:
    python examples/run_all_tests.py

    # Or run individual tests:
    python examples/test_1_token.py
    python examples/test_2_gemini_basic.py
    python examples/test_3_gemini_tools.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                                 ║
║           LUMI LLM - COMPREHENSIVE TEST SUITE                   ║
║                                                                 ║
║  Running all tests to validate:                                ║
║  - Configuration loading                                        ║
║  - IdaaS token generation                                       ║
║  - Gemini API calls (sync & async)                             ║
║  - Tool/Function calling                                        ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════╝
""")

    results = {}

    # Test 1: Token Generation
    print("\n" + "█" * 60)
    print("█  RUNNING TEST 1: TOKEN GENERATION")
    print("█" * 60)
    try:
        from examples.test_1_token import main as test1_main
        results["Token Generation"] = await test1_main()
    except Exception as e:
        print(f"Test 1 failed with exception: {e}")
        results["Token Generation"] = False

    if not results.get("Token Generation"):
        print("\n⚠️  Token test failed. Stopping here - other tests require valid tokens.")
        print_summary(results)
        return False

    # Test 2: Basic Gemini Calls
    print("\n" + "█" * 60)
    print("█  RUNNING TEST 2: BASIC GEMINI CALLS")
    print("█" * 60)
    try:
        from examples.test_2_gemini_basic import main as test2_main
        results["Basic Gemini Calls"] = await test2_main()
    except Exception as e:
        print(f"Test 2 failed with exception: {e}")
        results["Basic Gemini Calls"] = False

    # Test 3: Gemini with Tools
    print("\n" + "█" * 60)
    print("█  RUNNING TEST 3: GEMINI WITH TOOLS")
    print("█" * 60)
    try:
        from examples.test_3_gemini_tools import main as test3_main
        results["Gemini with Tools"] = await test3_main()
    except Exception as e:
        print(f"Test 3 failed with exception: {e}")
        results["Gemini with Tools"] = False

    print_summary(results)

    return all(results.values())


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "═" * 60)
    print("                    TEST SUMMARY")
    print("═" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("═" * 60)

    if all_passed:
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                                 ║
║                  🎉 ALL TESTS PASSED! 🎉                        ║
║                                                                 ║
║  Your Lumi LLM setup is working correctly.                     ║
║                                                                 ║
║  Next steps:                                                    ║
║  1. Start the MCP server: ./toolbox --tools-file tools.yaml    ║
║  2. Run the chat: python examples/chat.py                      ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                                 ║
║                  ❌ SOME TESTS FAILED                           ║
║                                                                 ║
║  Please check the error messages above.                        ║
║                                                                 ║
║  Common issues:                                                 ║
║  - Missing .env file with credentials                          ║
║  - Invalid CIBIS_CONSUMER_INTEGRATION_ID                       ║
║  - Invalid CIBIS_CONSUMER_SECRET                               ║
║  - Network connectivity issues                                  ║
║  - SSL certificate issues (verify_ssl: false may help)         ║
║                                                                 ║
╚═══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
