"""Deterministic offline MCP server used by the middleware integration tests.

No network and no sleep: the tools only exercise the MCP handshake, tool listing, result
mapping and error paths that the middleware adapter must handle.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("toolguard-test-tools")

_calls = {"flaky": 0}


@mcp.tool()
def echo(text: str) -> dict:
    """Echo the given text back."""
    return {"echo": text}


@mcp.tool()
def strict_integer(value: str) -> dict:
    """Return an integer, rejecting non-integer strings (schema/backend mismatch fixture)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("TOOL_ERROR[invalid_argument]: value must be an integer") from None
    return {"value": number}


@mcp.tool()
def needs_integer_value(value: str | int) -> dict:
    """Require an integer value and report the requirement through the error text."""
    if isinstance(value, str) and not value.lstrip("-").isdigit():
        raise ValueError("TOOL_ERROR[invalid_argument]: value must be an integer")
    return {"value": int(value)}


@mcp.tool()
def flaky_tool(attempts_before_success: int = 2) -> dict:
    """Fail the first attempts with a transient MCP error, then succeed."""
    _calls["flaky"] += 1
    if _calls["flaky"] <= attempts_before_success:
        raise ValueError(f"TOOL_ERROR[timeout]: simulated timeout (call {_calls['flaky']})")
    return {"ok": True, "calls": _calls["flaky"]}


@mcp.tool()
def failing_tool(message: str = "resource does not exist") -> dict:
    """Always fail with a non-transient MCP error."""
    raise ValueError(f"TOOL_ERROR[not_found]: {message}")


if __name__ == "__main__":  # pragma: no cover - manual debugging only
    mcp.run(transport="stdio")
