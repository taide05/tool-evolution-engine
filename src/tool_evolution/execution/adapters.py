"""工具适配器三实现——Mock（评测基准）/ HTTP（真实 JSON 工具）/ MCP（标准 Agent 协议）。"""

import asyncio
import json
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from ..collection.schemas import ErrorType
from .tool_specs import TOOL_SPECS, validate_params


class ToolResult(BaseModel):
    tool_name: str
    params: dict[str, Any]
    success: bool
    result: dict[str, Any] | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None
    latency_ms: int
    token_count: int = 0


class AsyncToolAdapter(Protocol):
    async def execute(self, tool_name: str, params: dict) -> ToolResult: ...

    async def close(self) -> None: ...


class MockAdapter:
    """确定性模拟适配器——与评测基准工具集同源，同输入同输出。"""

    def __init__(self, delay_s: float = 0):
        self._delay_s = delay_s

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        spec = TOOL_SPECS.get(tool_name)
        if spec is None:
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.PARAM_ERROR,
                error_message=f"unknown tool '{tool_name}'",
                latency_ms=0, token_count=0,
            )
        invalid = validate_params(tool_name, params)
        if invalid is not None:
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.PARAM_ERROR, error_message=invalid,
                latency_ms=spec["mock_latency_ms"], token_count=0,
            )
        max_results = params.get("max_results", 10)
        result = {
            key: [f"{tool_name}-mock-{i}" for i in range(min(int(max_results), 5))]
            for key in spec["mock_result_keys"]
        }
        result["_query"] = params.get("query", "")
        return ToolResult(
            tool_name=tool_name, params=params, success=True, result=result,
            latency_ms=spec["mock_latency_ms"],
            token_count=spec["mock_token_cost"],
        )

    async def close(self) -> None:
        return None


class HTTPAdapter:
    """真实 HTTP JSON 工具适配器——POST {base_url}/{tool_name}。"""

    def __init__(self, base_url: str, timeout: float = 10.0,
                 headers: dict[str, str] | None = None,
                 client: httpx.AsyncClient | None = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = headers or {}
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        client = self._get_client()
        url = f"{self._base_url}/{tool_name}"
        try:
            resp = await client.post(url, json=params, headers=self._headers,
                                     timeout=self._timeout)
        except httpx.TimeoutException:
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.TIMEOUT, error_message=f"timeout calling {url}",
                latency_ms=0, token_count=0,
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                error_message=f"transport error: {exc}",
                latency_ms=0, token_count=0,
            )
        if resp.status_code < 300:
            try:
                payload = resp.json()
            except ValueError:
                payload = {"raw": resp.text}
            return ToolResult(
                tool_name=tool_name, params=params, success=True,
                result=payload if isinstance(payload, dict) else {"value": payload},
                latency_ms=0, token_count=0,
            )
        mapping = {
            403: ErrorType.PERMISSION_DENIED,
            429: ErrorType.QUOTA_EXHAUSTED,
            502: ErrorType.SERVICE_UNAVAILABLE,
            503: ErrorType.SERVICE_UNAVAILABLE,
        }
        err_type = mapping.get(resp.status_code, ErrorType.PARAM_ERROR)
        return ToolResult(
            tool_name=tool_name, params=params, success=False, error_type=err_type,
            error_message=f"HTTP {resp.status_code}: {resp.text[:200]}",
            latency_ms=0, token_count=0,
        )

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


class MCPAdapter:
    """MCP 客户端适配器——stdio 子进程或注入会话（测试用 in-memory transport）。

    会话生命周期 = 适配器实例（单任务），首次 execute 惰性建立，
    调用方 finally 调 close（计划 D 决策：实例级是协议无任务级生命周期下的自洽粒度）。
    """

    def __init__(self, command: str | None = None, args: list[str] | None = None,
                 cwd: str | None = None, timeout_s: float = 30.0,
                 session=None):
        self._command = command
        self._args = args or []
        self._cwd = cwd
        self._timeout_s = timeout_s
        self._injected_session = session
        self._session = None
        self._ctx = None

    async def _ensure_session(self):
        if self._session is not None:
            return self._session
        if self._injected_session is not None:
            self._session = self._injected_session
            return self._session
        if self._command is None:
            raise RuntimeError("MCPAdapter needs command (stdio) or session (injected)")
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self._command, args=self._args, cwd=self._cwd
        )
        self._ctx = stdio_client(params)
        read, write = await self._ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.initialize()
        return self._session

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        try:
            session = await self._ensure_session()
        except Exception as exc:
            # 契约：MCP 错误一律转 ToolResult 不抛穿（stdio 启动失败/initialize 失败等）
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.SERVICE_UNAVAILABLE, error_message=str(exc),
                latency_ms=0, token_count=0,
            )
        try:
            call = await asyncio.wait_for(
                session.call_tool(tool_name, params), timeout=self._timeout_s
            )
        except asyncio.TimeoutError:
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.TIMEOUT,
                error_message=f"MCP call '{tool_name}' timed out",
                latency_ms=0, token_count=0,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                error_message=f"MCP transport error: {exc}",
                latency_ms=0, token_count=0,
            )
        if call.isError:
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.PARAM_ERROR,
                error_message=_content_to_text(call.content)[:500],
                latency_ms=0, token_count=0,
            )
        text = _content_to_text(call.content)
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            payload = {"text": text}
        return ToolResult(
            tool_name=tool_name, params=params, success=True,
            result=payload if isinstance(payload, dict) else {"value": payload},
            latency_ms=0, token_count=0,
        )

    async def close(self) -> None:
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
        self._session = None


def _content_to_text(content) -> str:
    parts = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)
