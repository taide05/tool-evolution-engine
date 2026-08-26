import json

import httpx

from tool_evolution.execution.planner import LLMPlanner


def _llm_response(steps):
    return httpx.Response(
        200, json={
            "choices": [{"message": {"content": json.dumps({"steps": steps})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _mock_client(response):
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: response),
        base_url="https://api.deepseek.com",
    )


class TestLLMPlanner:
    async def test_parses_valid_plan(self, monkeypatch):
        monkeypatch.setattr("tool_evolution.utils.config.settings.deepseek_api_key",
                            "sk-test")
        client = _mock_client(_llm_response([
            {"tool": "search_api", "params": {"query": "hello"}},
            {"tool": "analyze_api", "params": {"query": "hello"}},
        ]))
        planner = LLMPlanner(client=client)
        try:
            steps = await planner.plan("查资料并分析")
            assert steps == [
                {"tool": "search_api", "params": {"query": "hello"}},
                {"tool": "analyze_api", "params": {"query": "hello"}},
            ]
        finally:
            await client.aclose()

    async def test_unknown_tool_step_dropped(self, monkeypatch):
        monkeypatch.setattr("tool_evolution.utils.config.settings.deepseek_api_key",
                            "sk-test")
        client = _mock_client(_llm_response([
            {"tool": "ghost_tool", "params": {"query": "x"}},
            {"tool": "search_api", "params": {"query": "x"}},
        ]))
        planner = LLMPlanner(client=client)
        try:
            steps = await planner.plan("任务")
            assert steps == [{"tool": "search_api", "params": {"query": "x"}}]
        finally:
            await client.aclose()

    async def test_all_unknown_tools_none(self, monkeypatch):
        monkeypatch.setattr("tool_evolution.utils.config.settings.deepseek_api_key",
                            "sk-test")
        client = _mock_client(_llm_response([
            {"tool": "ghost_tool", "params": {"query": "x"}},
        ]))
        planner = LLMPlanner(client=client)
        try:
            assert await planner.plan("任务") is None
        finally:
            await client.aclose()

    async def test_missing_required_param_none(self, monkeypatch):
        monkeypatch.setattr("tool_evolution.utils.config.settings.deepseek_api_key",
                            "sk-test")
        client = _mock_client(_llm_response([
            {"tool": "search_api", "params": {"max_results": 5}},
        ]))
        planner = LLMPlanner(client=client)
        try:
            assert await planner.plan("任务") is None
        finally:
            await client.aclose()

    async def test_http_failure_none(self, monkeypatch):
        monkeypatch.setattr("tool_evolution.utils.config.settings.deepseek_api_key",
                            "sk-test")
        client = _mock_client(httpx.Response(500, json={"detail": "boom"}))
        planner = LLMPlanner(client=client, retries=0)
        try:
            assert await planner.plan("任务") is None
        finally:
            await client.aclose()

    async def test_no_key_zero_requests(self, monkeypatch):
        monkeypatch.setattr("tool_evolution.utils.config.settings.deepseek_api_key",
                            None)
        sent = {"count": 0}

        def handler(request):
            sent["count"] += 1
            return _llm_response([])

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://api.deepseek.com",
        )
        planner = LLMPlanner(client=client)
        try:
            assert await planner.plan("任务") is None
            assert sent["count"] == 0
        finally:
            await client.aclose()

    async def test_unknown_param_keys_stripped(self, monkeypatch):
        monkeypatch.setattr("tool_evolution.utils.config.settings.deepseek_api_key",
                            "sk-test")
        client = _mock_client(_llm_response([
            {"tool": "search_api",
             "params": {"query": "x", "bogus_key": 1, "max_results": 3}},
        ]))
        planner = LLMPlanner(client=client)
        try:
            steps = await planner.plan("任务")
            assert steps == [{"tool": "search_api",
                              "params": {"query": "x", "max_results": 3}}]
        finally:
            await client.aclose()
