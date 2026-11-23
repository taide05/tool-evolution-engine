"""Tests for the MCP transport adapter and the middleware running over MCP tools.

The integration tests use the MCP SDK's in-memory connected session with a small offline
server (``mcp_fixture_server.py``): real MCP protocol, no subprocess pipes (which the
sandbox blocks), no network, no clock.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from tool_evolution.toolguard.guard import ToolGuard
from tool_evolution.toolguard.mcp_backend import (
    MCPBackend,
    mcp_result_to_call_result,
    sanitize_json_schema,
)


@pytest.fixture(scope="module")
def fixture_server():
    path = Path(__file__).with_name("mcp_fixture_server.py")
    spec = importlib.util.spec_from_file_location("mcp_fixture_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcp_fixture_server"] = module
    spec.loader.exec_module(module)
    return module.mcp


class InMemoryMCPBridge:
    """Blocking facade over an in-memory MCP session (mirrors MCPBridge's interface)."""

    def __init__(self, server: Any):
        self.server = server
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: Any = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    def start(self) -> "InMemoryMCPBridge":
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(30):
            raise RuntimeError("in-memory MCP session did not start")
        if self._error is not None:
            raise RuntimeError("in-memory MCP session failed") from self._error
        return self

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            self._error = exc
        finally:
            self._ready.set()

    async def _async_main(self) -> None:
        async with create_connected_server_and_client_session(self.server) as session:
            self._loop = asyncio.get_running_loop()
            self._session = session
            self._stop = asyncio.Event()
            self._ready.set()
            await self._stop.wait()

    def close(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=30)

    def _run(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(30)

    def list_tools(self) -> list[Any]:
        return list(self._run(self._session.list_tools()).tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._run(self._session.call_tool(name, arguments=dict(arguments or {})))


@pytest.fixture()
def backend(fixture_server):
    bridge = InMemoryMCPBridge(fixture_server).start()
    try:
        yield MCPBackend(bridge)
    finally:
        bridge.close()


# ---------------------------------------------------------------------------
# pure mapping helpers
# ---------------------------------------------------------------------------
def test_sanitize_json_schema_drops_presentation_keys() -> None:
    schema = {
        "title": "Arguments",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"name": {"title": "Name", "type": "string"}},
        "required": ["name"],
    }
    cleaned = sanitize_json_schema(schema)
    assert cleaned == {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


def test_mcp_error_result_maps_onto_tool_taxonomy() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="TOOL_ERROR[invalid_argument]: x must be an integer")],
                            isError=True)
    mapped = mcp_result_to_call_result(result)
    assert mapped.status == "error"
    assert mapped.error == "invalid_argument: x must be an integer"


def test_mcp_ok_result_json_content_is_parsed() -> None:
    result = CallToolResult(content=[TextContent(type="text", text='{"echo": "hi"}')], isError=False)
    mapped = mcp_result_to_call_result(result)
    assert mapped.ok
    assert mapped.response == {"echo": "hi"}


def test_mcp_unknown_error_text_is_kept() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="Unknown tool: nope")], isError=True)
    mapped = mcp_result_to_call_result(result)
    assert mapped.status == "error"
    assert mapped.error is not None and "Unknown tool" in mapped.error


# ---------------------------------------------------------------------------
# real MCP session through the adapter + middleware
# ---------------------------------------------------------------------------
def test_backend_lists_mcp_tools_as_function_schemas(backend: MCPBackend) -> None:
    names = backend.tool_names()
    assert {"echo", "strict_integer", "flaky_tool", "failing_tool"} <= set(names)
    schemas = {schema["function"]["name"]: schema for schema in backend.schemas()}
    assert schemas["echo"]["function"]["parameters"]["properties"]["text"]["type"] == "string"
    assert schemas["strict_integer"]["function"]["parameters"]["required"] == ["value"]


def test_backend_calls_mcp_tool(backend: MCPBackend) -> None:
    result = backend.call("echo", {"text": "hello mcp"})
    assert result.ok
    assert result.response == {"echo": "hello mcp"}


def test_baseline_sees_mcp_error_text(backend: MCPBackend) -> None:
    result = backend.call("strict_integer", {"value": "not-a-number"})
    assert not result.ok
    assert result.error == "invalid_argument: value must be an integer"


def test_guard_repairs_type_error_from_mcp_error_hint(backend: MCPBackend) -> None:
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("needs_integer_value", {"value": " 42 "})
    assert result.ok
    assert result.response == {"value": 42}
    assert guard.stats["recovered"] == 1
    assert [event.rule for event in guard.events] == ["error_hint_coercion"]


def test_guard_retries_transient_mcp_error_within_one_turn(backend: MCPBackend) -> None:
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("flaky_tool", {"attempts_before_success": 2})
    assert result.ok
    assert result.response["ok"] is True
    assert guard.stats["recovered"] == 1
    assert guard.events[0].rule == "transient_retry:2"


def test_guard_does_not_retry_non_transient_mcp_error(backend: MCPBackend) -> None:
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("failing_tool", {})
    assert not result.ok
    assert guard.stats["recovered"] == 0
    assert "not_found" in (result.error or "")


# ---------------------------------------------------------------------------
# multi-server routing + catalog drift (unknown tool -> function-name resolution)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def market_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("toolguard-test-market")

    @server.tool()
    def get_fx_rate(base: str, target: str) -> dict:
        """Return a fixed FX rate (test fixture)."""
        return {"rate": 1.25, "base": base, "target": target}

    return server


@pytest.fixture()
def multi_backend(fixture_server, market_server):
    from tool_evolution.toolguard.mcp_backend import MCPServer, MCPMultiBackend

    worldbank_bridge = InMemoryMCPBridge(fixture_server).start()
    market_bridge = InMemoryMCPBridge(market_server).start()
    multi = MCPMultiBackend([
        MCPServer("worldbank", worldbank_bridge),
        MCPServer("market", market_bridge),
    ])
    try:
        multi.load_tools()
        yield multi
    finally:
        multi.close()


def test_multi_backend_routes_tools_across_servers(multi_backend) -> None:
    echo = multi_backend.call("echo", {"text": "hello"})
    fx = multi_backend.call("get_fx_rate", {"base": "USD", "target": "CNY"})
    assert echo.ok and echo.response == {"echo": "hello"}
    assert fx.ok and fx.response["rate"] == 1.25
    names = set(multi_backend.available_tool_names())
    assert {"echo", "get_fx_rate"} <= names


def test_catalog_drift_alias_is_advertised_but_not_registered(multi_backend) -> None:
    multi_backend.hide("get_fx_rate")
    multi_backend.advertise_alias("fetch_fx_rate", "get_fx_rate", "market")
    advertised = {schema["function"]["name"] for schema in multi_backend.schemas()}
    assert "fetch_fx_rate" in advertised
    assert "get_fx_rate" not in advertised
    assert set(multi_backend.available_tool_names()) == {
        "echo", "strict_integer", "flaky_tool", "failing_tool", "needs_integer_value", "get_fx_rate"}
    # The alias is routed to a server that does not register it -> real unknown-tool error.
    alias_result = multi_backend.call("fetch_fx_rate", {"base": "USD", "target": "CNY"})
    assert not alias_result.ok
    assert "Unknown tool" in (alias_result.error or "")


def test_guard_resolves_catalog_drift_alias(multi_backend) -> None:
    multi_backend.hide("get_fx_rate")
    multi_backend.advertise_alias("fetch_fx_rate", "get_fx_rate", "market")
    guard = ToolGuard(multi_backend, sleep_scale=0.0)
    result = guard.call("fetch_fx_rate", {"base": "USD", "target": "CNY"})
    assert result.ok
    assert result.response["rate"] == 1.25
    assert guard.events[0].rule == "function_name_resolution"
    assert guard.events[0].repaired_function == "get_fx_rate"
    assert guard.stats["recovered"] == 1
