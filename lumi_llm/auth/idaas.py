"""
IdaaS (Identity as a Service) client for token acquisition and management.
Handles OAuth2 client credentials flow with automatic token refresh.
"""

import asyncio
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
    Client for IdaaS token acquisition with caching.
    Thread-safe for sync usage, async-safe for async usage.
    """

    def __init__(self, config: "IdaaSConfig"):
        """
        Initialize the IdaaS client.

        Args:
            config: IdaaS configuration with URL, credentials, etc.
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

    def _parse_token_response(self, response_data: dict) -> TokenInfo:
        """Parse token response into TokenInfo."""
        # Calculate expiry time
        expires_in = response_data.get("expires_in", self.config.token_refresh_time)
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
            scope: Optional list of scopes to request.

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
                payload = {
                    "grant_type": "client_credentials",
                    "client_id": self.config.id,
                    "client_secret": self.config.secret,
                }

                if scope:
                    payload["scope"] = " ".join(scope)

                response = client.post(
                    self.config.url,
                    data=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()

                self._token = self._parse_token_response(response.json())
                return self._token.access_token

    async def get_token(self, scope: list[str] | None = None) -> str:
        """
        Get a valid access token asynchronously.
        Will fetch a new token if the current one is expired.

        Args:
            scope: Optional list of scopes to request.

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
                payload = {
                    "grant_type": "client_credentials",
                    "client_id": self.config.id,
                    "client_secret": self.config.secret,
                }

                if scope:
                    payload["scope"] = " ".join(scope)

                response = await client.post(
                    self.config.url,
                    data=payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()

                self._token = self._parse_token_response(response.json())
                return self._token.access_token

    def clear_token(self) -> None:
        """Clear the cached token, forcing a refresh on next request."""
        with self._lock:
            self._token = None
