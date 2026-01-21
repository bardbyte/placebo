"""
Configuration management for Lumi LLM.
Loads from config.yaml with environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class IdaaSConfig(BaseModel):
    """IdaaS authentication configuration."""
    id: str
    version: str = "2"
    url: str
    secret: str
    token_refresh_time: int = 60
    verify_ssl: bool = True


class LLMConfig(BaseModel):
    """LLM configuration - direct provider settings."""
    url: str
    scope: list[str] = Field(default_factory=list)
    temperature: float = 0.5
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int = 1024
    verify_ssl: bool = True


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    url: str
    transport: str = "streamable-http"
    verify_ssl: bool = True


class MCPConfig(BaseModel):
    """MCP configuration."""
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class Settings(BaseModel):
    """Main settings container."""
    idaas: IdaaSConfig
    llm: LLMConfig
    mcp: MCPConfig


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for match in matches:
            env_value = os.getenv(match, "")
            value = value.replace(f"${{{match}}}", env_value)
        return value
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_settings(
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
) -> Settings:
    """
    Load settings from config.yaml and .env file.

    Args:
        config_path: Path to config.yaml. Defaults to ./config.yaml
        env_path: Path to .env file. Defaults to ./.env

    Returns:
        Settings object with all configuration loaded.
    """
    # Load .env file first
    if env_path is None:
        env_path = Path.cwd() / ".env"
    load_dotenv(env_path)

    # Load config.yaml
    if config_path is None:
        config_path = Path.cwd() / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    config = _substitute_env_vars(raw_config)

    return Settings(**config)
