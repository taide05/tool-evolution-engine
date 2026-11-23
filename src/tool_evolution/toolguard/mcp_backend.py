"""MCP transport adapter: expose real MCP stdio servers as ToolGuard backends.

ToolGuard was built against the offline tool-call traces and validated on plain Python
function tools.  This adapter plugs the *same* middleware into the Model Context Protocol
tool layer without changing a line of the middleware:

* the tool catalogue comes from MCP ``tools/list`` (``inputSchema`` JSON Schema);
* every ``tools/call`` goes through the middleware as one ``CallResult``;
* MCP ``isError`` results are mapped back onto the same error taxonomy the middleware
  learned offline (``timeout`` / ``rate_limit`` / ``invalid_argument`` / ...), so the
  retry and argument-repair rules apply unchanged.

Because the MCP client is async and the middleware is synchronous, a :class:`MCPBridge`
runs the stdio session on a dedicated event-loop thread and exposes blocking calls.  The
bridge also survives a **real transport failure**: if the server process dies mid-request
the session is torn down, the failed call surfaces as a retryable ``request_error`` and
the next call reconnects a fresh session/process (see ``TOOLGUARD_MCP_CRASH_AFTER``).

:class:`MCPMultiBackend` merges several MCP servers into one tool catalogue, which is how
the demo exercises multi-server routing and catalog drift (a tool that is advertised to
the agent but no longer registered on any server).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .backend import CallResult

# Errors returned by the demo MCP server carry a ``TOOL_ERROR[kind]: message`` marker so
# the middleware sees the same taxonomy it uses for function tools.  The marker is the
# server's structured-error convention, not something ToolGuard requires.
_TOOL_ERROR_RE = re.compile(r"TOOL_ERROR\[(?P<kind>[a-z_]+)\]:\s*(?P<message>.*)", re.DOTALL)
# JSON-schema keywords that MCP servers may emit but OpenAI-style function schemas drop.
_DROPPED_SCHEMA_KEYS = ("title", "$schema", "$id")


def sanitize_json_schema(schema: Any) -> Any:
    """Recursively drop presentation-only JSON Schema keywords for function calls."""
    if isinstance(schema, dict):
        return {key: sanitize_json_schema(value) for key, value in schema.items() if key not in _DROPPED_SCHEMA_KEYS}
    if isinstance(schema, list):
        return [sanitize_json_schema(item) for item in schema]
    return schema


def mcp_result_to_call_result(result: Any) -> CallResult:
    """Map an MCP ``CallToolResult`` onto ToolGuard's ``CallResult``."""
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
        elif hasattr(block, "model_dump"):
            texts.append(json.dumps(block.model_dump(), ensure_ascii=False, default=str))
        else:  # pragma: no cover - defensive for future content block types
            texts.append(str(block))
    text = "\n".join(texts).strip()
    structured = getattr(result, "structuredContent", None)
    if getattr(result, "isError", False):
        match = _TOOL_ERROR_RE.search(text)
        if match:
            return CallResult("error", error=f"{match.group('kind')}: {match.group('message').strip()}", source="mcp")
        return CallResult("error", error=f"tool_error: {text or 'MCP tool returned an error'}", source="mcp")
    if not structured and not text:
        return CallResult("ok", response=None, source="mcp")
    if structured is not None:
        # FastMCP wraps dict returns as {"result": <dict>}; unwrap so the middleware and
        # the agent see the same response shape as the function-tool demo.
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return CallResult("ok", response=structured["result"], source="mcp")
        try:
            return CallResult("ok", response=json.loads(text), source="mcp") if text else CallResult(
                "ok", response=structured, source="mcp")
        except json.JSONDecodeError:
            return CallResult("ok", response=structured, source="mcp")
    try:
        return CallResult("ok", response=json.loads(text), source="mcp")
    except json.JSONDecodeError:
        return CallResult("ok", response=text, source="mcp")


class MCPBridge:
    """Runs an MCP stdio session on a background event loop; exposes blocking calls.

    The bridge is restartable: a transport failure (server process crash / broken pipe)
    marks the session as dead, and the next call starts a fresh server process and session
    before retrying.  ``restart_env`` replaces the subprocess environment after the first
    start, which lets the demo disable a crash trigger on restart (crash-once semantics).
    """

    def __init__(self, command: list[str], cwd: str | Path | None = None,
                 env: dict[str, str] | None = None, timeout_s: float = 120.0,
                 restart_env: dict[str, str] | None = None):
        if not command:
            raise ValueError("command must not be empty")
        self.command = list(command)
        self.cwd = str(cwd) if cwd is not None else None
        self.env = env
        self.restart_env = restart_env
        self.timeout_s = timeout_s
        self.restarts = 0
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._needs_restart = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "MCPBridge":
        self._ready = threading.Event()
        self._startup_error = None
        self._thread = threading.Thread(target=self._thread_main, name="mcp-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.timeout_s):
            raise RuntimeError(f"MCP server did not start within {self.timeout_s}s")
        if self._startup_error is not None:
            raise RuntimeError(f"MCP session failed: {self._startup_error}") from self._startup_error
        if self._session is None:
            raise RuntimeError("MCP session is not available")
        self._needs_restart = False
        return self

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            self._startup_error = exc
        finally:
            self._ready.set()

    async def _async_main(self) -> None:
        params = StdioServerParameters(command=self.command[0], args=self.command[1:], cwd=self.cwd, env=self.env)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(
                read_stream, write_stream,
                read_timeout_seconds=_dt.timedelta(seconds=self.timeout_s),
            ) as session:
                await session.initialize()
                self._loop = asyncio.get_running_loop()
                self._session = session
                self._stop = asyncio.Event()
                self._ready.set()
                await self._stop.wait()

    def _request_stop(self) -> None:
        loop, stop, thread = self._loop, self._stop, self._thread
        if loop is not None and stop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(stop.set)
            except RuntimeError:  # pragma: no cover - loop already torn down
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=self.timeout_s)
        self._session = None
        self._loop = None
        self._stop = None

    def restart(self) -> None:
        """Tear the current session down and start a fresh server process."""
        self._request_stop()
        if self.restart_env is not None:
            self.env = self.restart_env
        self.restarts += 1
        self.start()

    def close(self) -> None:
        self._request_stop()

    def __enter__(self) -> "MCPBridge":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- blocking RPC over the async session -------------------------------
    def _run(self, coro: Any) -> Any:
        if self._loop is None or self._session is None:
            raise RuntimeError("MCP bridge is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(self.timeout_s)

    def _ensure_session(self) -> None:
        if self._needs_restart or self._session is None:
            self.restart()

    def list_tools(self) -> list[Any]:
        self._ensure_session()
        try:
            result = self._run(self._session.list_tools())
        except Exception as exc:  # noqa: BLE001 - transport failure
            self._needs_restart = True
            raise RuntimeError(f"MCP transport failure: {exc}") from exc
        return list(result.tools or [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self._ensure_session()
        try:
            return self._run(self._session.call_tool(name, arguments=dict(arguments or {})))
        except Exception as exc:  # noqa: BLE001 - session died mid-request (e.g. server crash)
            self._needs_restart = True
            raise RuntimeError(f"MCP transport failure: {exc}") from exc


class MCPBackend:
    """ToolGuard backend over one MCP server (same interface as the live-API backend)."""

    def __init__(self, bridge: MCPBridge, tools: list[Any] | None = None):
        self.bridge = bridge
        self.stats = {"calls": 0, "ok": 0, "errors": 0}
        self._tools: list[Any] = list(tools or [])

    # -- catalogue ---------------------------------------------------------
    def load_tools(self) -> list[Any]:
        self._tools = self.bridge.list_tools()
        return self._tools

    def tool_names(self) -> list[str]:
        if not self._tools:
            self.load_tools()
        return [tool.name for tool in self._tools]

    def available_tool_names(self) -> list[str]:
        """Names the server can actually execute (used by function-name resolution)."""
        return self.tool_names()

    def schemas(self) -> list[dict[str, Any]]:
        if not self._tools:
            self.load_tools()
        return [self._tool_schema(tool) for tool in self._tools]

    @staticmethod
    def _tool_schema(tool: Any, name: str | None = None) -> dict[str, Any]:
        parameters = sanitize_json_schema(tool.inputSchema or {"type": "object", "properties": {}})
        return {
            "type": "function",
            "function": {
                "name": name or tool.name,
                "description": (tool.description or "")[:1024],
                "parameters": parameters,
            },
        }

    # -- calls -------------------------------------------------------------
    def call(self, function_name: str, arguments: dict[str, Any]) -> CallResult:
        self.stats["calls"] += 1
        try:
            result = self.bridge.call_tool(function_name, arguments)
        except Exception as exc:  # noqa: BLE001 - transport failure is a retryable tool failure
            self.stats["errors"] += 1
            return CallResult("error", error=f"request_error: MCP transport failure: {exc}", source="mcp")
        call_result = mcp_result_to_call_result(result)
        self.stats["ok" if call_result.ok else "errors"] += 1
        return call_result


@dataclass
class MCPServer:
    label: str
    bridge: MCPBridge
    backend: MCPBackend | None = None


class MCPMultiBackend:
    """Merges several MCP servers into one tool catalogue with routing.

    Features the demo needs beyond a single server:

    * **multi-server routing**: tools are routed to the server that registers them;
    * **hidden tools**: a tool can be callable but not advertised (used to force the agent
      to use a drifted alias);
    * **advertised aliases (catalog drift)**: an alias is advertised in ``schemas()`` but
      not registered on the server that is asked to run it, so calls fail with an MCP
      unknown-tool error -- exactly what happens after a server upgrade renames a tool
      while a stale client config still advertises the old name.  ToolGuard resolves the
      alias against ``available_tool_names()`` with the same similarity rule it validated
      offline.
    """

    def __init__(self, servers: list[MCPServer]):
        self.servers = servers
        self.routing: dict[str, MCPBackend] = {}
        self.hidden: set[str] = set()
        self.aliases: dict[str, MCPBackend] = {}
        self.stats = {"calls": 0, "ok": 0, "errors": 0}
        self._advertised: list[dict[str, Any]] = []

    def load_tools(self) -> None:
        self.routing = {}
        self._advertised = []
        for server in self.servers:
            server.backend = MCPBackend(server.bridge)
            server.backend.load_tools()
            for tool in server.backend._tools:
                self.routing[tool.name] = server.backend
            for schema in server.backend.schemas():
                self._advertised.append(schema)
        self._rebuild_schemas()

    def hide(self, tool_name: str) -> None:
        self.hidden.add(tool_name)
        self._rebuild_schemas()

    def advertise_alias(self, alias_name: str, real_tool_name: str, target_label: str) -> None:
        """Advertise ``alias_name`` (with the real tool's parameters) but route it to a server
        that does not register it, so the call fails as a real unknown-tool error."""
        target = next((server.backend for server in self.servers if server.label == target_label), None)
        if target is None:
            raise ValueError(f"unknown MCP server label: {target_label}")
        real_schema = next(
            (schema for schema in self._advertised
             if schema["function"]["name"] == real_tool_name), None)
        if real_schema is None:
            raise ValueError(f"cannot advertise alias for unknown tool {real_tool_name}")
        alias_schema = json.loads(json.dumps(real_schema))
        alias_schema["function"]["name"] = alias_name
        alias_schema["function"]["description"] = (
            f"{alias_schema['function']['description']} (advertised name; the server may register it "
            f"under a different name)")
        self.aliases[alias_name] = target
        self._advertised.append(alias_schema)
        self._rebuild_schemas()

    def _rebuild_schemas(self) -> None:
        self.schemas_list = [
            schema for schema in self._advertised
            if schema["function"]["name"] not in self.hidden
        ]

    def schemas(self) -> list[dict[str, Any]]:
        if not self.routing:
            self.load_tools()
        return list(self.schemas_list)

    def available_tool_names(self) -> list[str]:
        """Real (registered) tool names across servers; used by function-name resolution."""
        if not self.routing:
            self.load_tools()
        return list(self.routing)

    def real_tool_names(self) -> list[str]:
        return self.available_tool_names()

    def call(self, function_name: str, arguments: dict[str, Any]) -> CallResult:
        self.stats["calls"] += 1
        backend = self.routing.get(function_name)
        if backend is None:
            backend = self.aliases.get(function_name)
        if backend is None:
            self.stats["errors"] += 1
            return CallResult("error", error=f"tool_error: Unknown tool: {function_name}", source="mcp")
        call_result = backend.call(function_name, arguments)
        self.stats["ok" if call_result.ok else "errors"] += 1
        return call_result

    def close(self) -> None:
        for server in self.servers:
            server.bridge.close()
