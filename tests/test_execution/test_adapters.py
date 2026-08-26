import time

import httpx

from tool_evolution.collection.schemas import ErrorType
from tool_evolution.execution.adapters import HTTPAdapter, MCPAdapter, MockAdapter


class TestMockAdapter:
    async def test_success(self):
        adapter = MockAdapter()
        try:
            result = await adapter.execute(
                "search_api", {"query": "hello", "max_results": 5}
            )
            assert result.success is True
            assert result.error_type is None
            for key in ("documents", "total"):
                assert key in result.result
            assert result.latency_ms == 175
            assert result.token_count == 120
        finally:
            await adapter.close()

    async def test_invalid_params(self):
        adapter = MockAdapter()
        try:
            result = await adapter.execute("search_api", {"max_results": 5})
            assert result.success is False
            assert result.error_type == ErrorType.PARAM_ERROR
            assert "query" in (result.error_message or "")
        finally:
            await adapter.close()

    async def test_unknown_tool(self):
        adapter = MockAdapter()
        try:
            result = await adapter.execute("nope", {"query": "x"})
            assert result.success is False
            assert result.error_type == ErrorType.PARAM_ERROR
        finally:
            await adapter.close()

    async def test_deterministic_two_calls(self):
        adapter = MockAdapter()
        try:
            params = {"query": "q", "max_results": 3, "lang": "zh"}
            r1 = await adapter.execute("search_api", params)
            r2 = await adapter.execute("search_api", params)
            assert r1.result == r2.result
            assert r1.latency_ms == r2.latency_ms
            assert r1.token_count == r2.token_count
        finally:
            await adapter.close()

    async def test_delay_applied(self):
        adapter = MockAdapter(delay_s=0.3)
        try:
            start = time.monotonic()
            await adapter.execute("search_api", {"query": "q"})
            elapsed = time.monotonic() - start
            assert elapsed >= 0.25
        finally:
            await adapter.close()


class TestHTTPAdapter:
    async def test_success_200(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True, "data": [1, 2]})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://fake"
        )
        adapter = HTTPAdapter("http://fake", client=client)
        try:
            result = await adapter.execute("search_api", {"query": "q"})
            assert result.success is True
            assert result.result == {"ok": True, "data": [1, 2]}
        finally:
            await adapter.close()

    async def test_error_mapping(self):
        cases = [
            (403, ErrorType.PERMISSION_DENIED),
            (429, ErrorType.QUOTA_EXHAUSTED),
            (503, ErrorType.SERVICE_UNAVAILABLE),
            (400, ErrorType.PARAM_ERROR),
        ]
        for status, expected_err in cases:
            def handler(request, status=status):
                return httpx.Response(status, json={"detail": "err"})

            client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://fake"
            )
            adapter = HTTPAdapter("http://fake", client=client)
            try:
                result = await adapter.execute("search_api", {"query": "q"})
                assert result.success is False
                assert result.error_type == expected_err
            finally:
                await adapter.close()

    async def test_timeout(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://fake"
        )
        adapter = HTTPAdapter("http://fake", client=client, timeout=1.0)
        try:
            result = await adapter.execute("search_api", {"query": "q"})
            assert result.success is False
            assert result.error_type == ErrorType.TIMEOUT
        finally:
            await adapter.close()


class TestMCPAdapter:
    # stdio 子进程路径的测试曾挂死（Windows 管道 + 会话关闭问题，spec 坑清单
    # 警告场景）——契约由 in-memory 测试覆盖，stdio 真实路径留终验冒烟验证

    async def test_call_tool_success(self, db_conn):
        # 会话必须在同一 task 内进出（anyio CancelScope 跨 task 退出会炸）——
        # spec 坑清单纪律，fixture 跨 yield 持有会触发
        from tool_evolution.governance.mcp_bridge import MCPBridge, mcp, set_bridge
        from mcp.shared.memory import create_connected_server_and_client_session

        set_bridge(MCPBridge(db_conn))
        async with create_connected_server_and_client_session(mcp) as session:
            adapter = MCPAdapter(session=session)
            result = await adapter.execute("search_memory", {"query": "entity"})
            assert result.success is True
            assert isinstance(result.result, dict)

    async def test_unknown_tool_error(self, db_conn):
        from tool_evolution.governance.mcp_bridge import MCPBridge, mcp, set_bridge
        from mcp.shared.memory import create_connected_server_and_client_session

        set_bridge(MCPBridge(db_conn))
        async with create_connected_server_and_client_session(mcp) as session:
            adapter = MCPAdapter(session=session)
            result = await adapter.execute("nonexistent_mcp_tool", {"query": "x"})
            assert result.success is False
            assert result.error_type is not None
