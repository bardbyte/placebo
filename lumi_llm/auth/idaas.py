"""
IdaaS (Identity as a Service) client for token acquisition and management.
Uses signature-based authentication with X-Auth headers.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from lumi_llm.config.settings import IdaaSConfig


@dataclass
class TokenInfo:
    """Holds token information with expiry."""
    access_token: str
    token_type: str
    expires_at: float  # Unix timestamp
    scope: str | None = None


class IdaaSClient:
    """
    Client for IdaaS token acquisition with signature-based auth.
    Thread-safe for sync usage, async-safe for async usage.
    """

    def __init__(self, config: "IdaaSConfig"):
        """
        Initialize the IdaaS client.

        Args:
            config: IdaaS configuration with URL, credentials, scope, etc.
        """
        self.config = config
        self._token: TokenInfo | None = None
        self._lock = Lock()
        self._async_lock: asyncio.Lock | None = None

        # Buffer time before expiry to refresh (10 seconds)
        self._refresh_buffer = 10

    def _get_verify_ssl(self) -> bool:
        """Get SSL verification setting from config."""
        return getattr(self.config, 'verify_ssl', True)

    @property
    def _async_lock_instance(self) -> asyncio.Lock:
        """Lazily create async lock to avoid event loop issues."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _is_token_valid(self) -> bool:
        """Check if the current token is still valid."""
        if self._token is None:
            return False
        return time.time() < (self._token.expires_at - self._refresh_buffer)

    def _generate_signature(self, timestamp: int) -> str:
        """
        Generate HMAC-SHA256 signature for authentication.

        The signature is computed as: HMAC-SHA256(secret, app_id + timestamp)
        Then base64 URL-safe encoded.
        """
        message = f"{self.config.id}{timestamp}"
        signature = hmac.new(
            self.config.secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        # Base64 URL-safe encoding (replace + with -, / with _, remove padding)
        encoded = base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')
        return encoded

    def _get_auth_headers(self) -> dict[str, str]:
        """Generate authentication headers for IdaaS request."""
        timestamp = int(time.time() * 1000)  # Milliseconds
        signature = self._generate_signature(timestamp)

        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Auth-AppID": self.config.id,
            "X-Auth-Signature": signature,
            "X-Auth-Version": "2",
            "X-Auth-Timestamp": str(timestamp),
        }

    def _build_request_body(self, scope: list[str] | None = None) -> dict:
        """Build the JSON request body."""
        effective_scope = scope if scope is not None else self.config.scope
        return {"scope": effective_scope} if effective_scope else {}

    def _parse_token_response(self, response_data: dict) -> TokenInfo:
        """Parse token response into TokenInfo."""
        # Calculate expiry time - use response expires_in or fall back to config
        expires_in = response_data.get("expires_in", self.config.token_refresh_interval)
        expires_at = time.time() + expires_in

        return TokenInfo(
            access_token=response_data["access_token"],
            token_type=response_data.get("token_type", "Bearer"),
            expires_at=expires_at,
            scope=response_data.get("scope"),
        )

    def get_token_sync(self, scope: list[str] | None = None) -> str:
        """
        Get a valid access token synchronously.
        Will fetch a new token if the current one is expired.

        Args:
            scope: Optional list of scopes to request. Defaults to config scope.

        Returns:
            Valid access token string.
        """
        with self._lock:
            if self._is_token_valid():
                return self._token.access_token

            # Fetch new token
            with httpx.Client(
                timeout=30.0,
                verify=self._get_verify_ssl()
            ) as client:
                headers = self._get_auth_headers()
                body = self._build_request_body(scope)

                response = client.post(
                    self.config.url,
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()

                self._token = self._parse_token_response(response.json())
                return self._token.access_token

    async def get_token(self, scope: list[str] | None = None) -> str:
        """
        Get a valid access token asynchronously.
        Will fetch a new token if the current one is expired.

        Args:
            scope: Optional list of scopes to request. Defaults to config scope.

        Returns:
            Valid access token string.
        """
        async with self._async_lock_instance:
            if self._is_token_valid():
                return self._token.access_token

            # Fetch new token
            async with httpx.AsyncClient(
                timeout=30.0,
                verify=self._get_verify_ssl()
            ) as client:
                headers = self._get_auth_headers()
                body = self._build_request_body(scope)

                response = await client.post(
                    self.config.url,
                    headers=headers,
                    json=body,
                )
                response.raise_for_status()

                self._token = self._parse_token_response(response.json())
                return self._token.access_token

    def clear_token(self) -> None:
        """Clear the cached token, forcing a refresh on next request."""
        with self._lock:
            self._token = None
