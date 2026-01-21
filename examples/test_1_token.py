#!/usr/bin/env python3
"""
Test Script 1: Token Generation

Tests that the IdaaS client can successfully:
1. Load configuration from config.yaml
2. Authenticate with the IdaaS service
3. Obtain an access token (both sync and async)

Usage:
    python examples/test_1_token.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumi_llm.config.settings import load_settings
from lumi_llm.auth.idaas import IdaaSClient


def test_config_loading():
    """Test 1a: Verify config loads correctly."""
    print("\n" + "=" * 60)
    print("TEST 1a: Configuration Loading")
    print("=" * 60)

    try:
        config_path = Path(__file__).parent.parent / "config.yaml"
        env_path = Path(__file__).parent.parent / ".env"
        settings = load_settings(config_path=config_path, env_path=env_path)

        print(f"  [OK] Config loaded successfully")
        print(f"  IdaaS URL: {settings.idaas.url}")
        print(f"  IdaaS Scope: {settings.idaas.scope}")
        print(f"  IdaaS Originator: {settings.idaas.originator_source}")
        print(f"  IdaaS verify_ssl: {settings.idaas.verify_ssl}")
        print(f"  LLM URL: {settings.llm.url}")
        print(f"  LLM verify_ssl: {settings.llm.verify_ssl}")
        print(f"  MCP Servers: {list(settings.mcp.servers.keys())}")

        # Verify critical fields are populated
        assert settings.idaas.id, "IdaaS ID is empty - check CIBIS_CONSUMER_INTEGRATION_ID env var"
        assert settings.idaas.secret, "IdaaS secret is empty - check CIBIS_CONSUMER_SECRET env var"
        print(f"  [OK] IdaaS credentials populated (ID length: {len(settings.idaas.id)})")

        return settings
    except Exception as e:
        print(f"  [FAIL] {e}")
        raise


def test_token_sync(settings):
    """Test 1b: Synchronous token acquisition."""
    print("\n" + "=" * 60)
    print("TEST 1b: Synchronous Token Acquisition")
    print("=" * 60)

    try:
        auth_client = IdaaSClient(settings.idaas)
        print(f"  IdaaS client created")
        print(f"  Requesting token from: {settings.idaas.url}")
        print(f"  With scope: {settings.idaas.scope}")

        # Show what headers will be sent
        headers = auth_client._get_auth_headers()
        print(f"  Headers being sent:")
        print(f"    X-Auth-AppID: {headers['X-Auth-AppID'][:20]}...")
        print(f"    X-Auth-Version: {headers['X-Auth-Version']}")
        print(f"    X-Auth-Timestamp: {headers['X-Auth-Timestamp']}")
        print(f"    X-Auth-Signature: {headers['X-Auth-Signature'][:20]}...")
        print(f"    Content-Type: {headers['Content-Type']}")

        token = auth_client.get_token_sync()

        print(f"  [OK] Token acquired successfully")
        print(f"  Token length: {len(token)} characters")
        print(f"  Token prefix: {token[:20]}...")

        # Verify token is cached
        token2 = auth_client.get_token_sync()
        assert token == token2, "Token should be cached"
        print(f"  [OK] Token caching works")

        return auth_client, token
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_token_async(settings):
    """Test 1c: Asynchronous token acquisition."""
    print("\n" + "=" * 60)
    print("TEST 1c: Asynchronous Token Acquisition")
    print("=" * 60)

    try:
        auth_client = IdaaSClient(settings.idaas)
        print(f"  IdaaS client created (async)")

        token = await auth_client.get_token()

        print(f"  [OK] Async token acquired successfully")
        print(f"  Token length: {len(token)} characters")
        print(f"  Token prefix: {token[:20]}...")

        # Verify token is cached
        token2 = await auth_client.get_token()
        assert token == token2, "Token should be cached"
        print(f"  [OK] Async token caching works")

        return token
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


def test_token_refresh(settings):
    """Test 1d: Token refresh after clear."""
    print("\n" + "=" * 60)
    print("TEST 1d: Token Refresh After Clear")
    print("=" * 60)

    try:
        auth_client = IdaaSClient(settings.idaas)

        token1 = auth_client.get_token_sync()
        print(f"  First token: {token1[:20]}...")

        auth_client.clear_token()
        print(f"  Token cleared")

        token2 = auth_client.get_token_sync()
        print(f"  New token: {token2[:20]}...")

        # Tokens might be the same if server returns same token, that's ok
        print(f"  [OK] Token refresh works")

        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Run all token tests."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          TEST SCRIPT 1: TOKEN GENERATION                       ║
║                                                                 ║
║  Tests IdaaS authentication and token acquisition              ║
╚═══════════════════════════════════════════════════════════════╝
""")

    try:
        # Test 1a: Config loading
        settings = test_config_loading()

        # Test 1b: Sync token
        auth_client, sync_token = test_token_sync(settings)

        # Test 1c: Async token
        async_token = await test_token_async(settings)

        # Test 1d: Token refresh
        test_token_refresh(settings)

        print("\n" + "=" * 60)
        print("ALL TOKEN TESTS PASSED!")
        print("=" * 60)
        print(f"\nSync token:  {sync_token[:30]}...")
        print(f"Async token: {async_token[:30]}...")

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TOKEN TESTS FAILED: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
