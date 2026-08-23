"""Shared MCP management payload helpers.

The dashboard and the Gateway API expose the same Agent-level MCP
configuration.  Keep request normalization and response redaction here so
the two HTTP surfaces cannot drift from the existing CLI/config contract.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from hermes_cli.mcp_config import (
    _bearer_auth_headers,
    _strip_bearer_prefix,
)
from hermes_cli.mcp_security import validate_mcp_server_entry


def normalize_mcp_server_create(
    payload: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Validate an HTTP MCP create payload and build the persisted config.

    The returned config never contains the submitted bearer token.  Callers
    persist that one-time provisioning value with ``_save_bearer_auth_token``
    after entering the intended profile scope.
    """
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Server name is required")

    raw_url = payload.get("url")
    raw_command = payload.get("command")
    url = str(raw_url or "").strip()
    command = str(raw_command or "").strip()
    auth = str(payload.get("auth") or "none").strip().lower()
    args = payload.get("args") or []
    env = payload.get("env") or {}
    bearer_token = payload.get("bearer_token")

    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("args must be an array of strings")
    if not isinstance(env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in env.items()
    ):
        raise ValueError("env must be an object of string values")
    if bearer_token is not None and not isinstance(bearer_token, str):
        raise ValueError("bearer_token must be a string")

    if bool(url) == bool(command):
        raise ValueError("Provide exactly one of URL (HTTP/SSE) or command (stdio)")
    if auth not in {"none", "header", "oauth"}:
        raise ValueError(f"Unsupported auth mode: {auth}")

    server_config: Dict[str, Any] = {}
    if url:
        if args:
            raise ValueError("Arguments are only supported for stdio MCP servers")
        if env:
            raise ValueError(
                "Environment variables are only supported for stdio MCP servers"
            )
        if auth == "header":
            normalized = _strip_bearer_prefix(bearer_token or "")
            if not normalized or normalized.lower() == "bearer":
                raise ValueError("Bearer token is required")
            server_config["headers"] = _bearer_auth_headers(name)
        elif bearer_token is not None:
            raise ValueError("Bearer token requires header authentication")

        server_config["url"] = url
        if auth == "oauth":
            server_config["auth"] = "oauth"
    else:
        if auth != "none" or bearer_token is not None:
            raise ValueError(
                "HTTP authentication is not supported for stdio MCP servers"
            )
        server_config["command"] = command
        if args:
            server_config["args"] = list(args)
        if env:
            server_config["env"] = dict(env)

    issues = validate_mcp_server_entry(name, server_config)
    if issues:
        raise ValueError(f"Server '{name}' rejected: {'; '.join(issues)}")
    return name, server_config, bearer_token


def redact_mcp_env(env: Mapping[str, Any]) -> Dict[str, str]:
    """Mask MCP environment values for API responses."""
    from hermes_cli.config import redact_key

    output: Dict[str, str] = {}
    for key, value in (env or {}).items():
        try:
            output[str(key)] = redact_key(str(value)) if value else ""
        except Exception:
            output[str(key)] = "***"
    return output


def mcp_server_summary(name: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the stable, secret-redacted representation of an MCP server."""
    transport = "http" if cfg.get("url") else (
        "stdio" if cfg.get("command") else "unknown"
    )
    auth = cfg.get("auth")
    headers = cfg.get("headers") or {}
    if not auth and isinstance(headers, dict) and any(
        str(key).lower() == "authorization" for key in headers
    ):
        auth = "header"
    return {
        "name": name,
        "transport": transport,
        "url": cfg.get("url"),
        "command": cfg.get("command"),
        "args": list(cfg.get("args") or []),
        "env": redact_mcp_env(cfg.get("env") or {}),
        "auth": auth,
        "enabled": cfg.get("enabled", True) is not False,
        "tools": cfg.get("tools"),
    }
