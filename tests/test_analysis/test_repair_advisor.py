import asyncio
import json
import pytest
import httpx
from tool_evolution.analysis.repair_advisor import RepairAdvisor
from tool_evolution.utils.config import settings


def _rule(row_id=1, tool="repair_api", rule_type="range_rule",
          condition=None, action=None):
    return {
        "id": row_id, "tool_name": tool, "tool_version": "1.0.0",
        "rule_type": rule_type,
        "condition": json.dumps(condition if condition is not None
                                else {"param_names": ["max_results"]}),
        "action": json.dumps(action if action is not None
                             else {"validate_before_call": True,
                                   "error_msg_template": "max_results must be between 1 and 20, got 30"}),
        "status": "active",
    }


def _mock_client(payload: dict | None, calls: list, fail_times: int = 0):
    """payload = 响应信封内 choices[0].message.content 的 JSON 对象；None 表示内容为非法 JSON 字符串。
    响应形状 = OpenAI 兼容信封（Task 3 Interfaces 定义）。"""
    def handler(request):
        calls.append(request)
        if fail_times and len(calls) <= fail_times:
            raise httpx.ConnectError("boom")
        content = json.dumps(payload) if payload is not None else "not json"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        })
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
async def seeded_rule(db_conn):
    cursor = await db_conn.execute(
        """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
           VALUES ('repair_api', '1.0.0', 'range_rule', ?, ?)""",
        (_rule()["condition"], _rule()["action"]))
    await db_conn.commit()
    return cursor.lastrowid


class TestRepairAdvisorGenerate:
    async def test_generate_stores_hint_with_fix(self, db_conn, seeded_rule, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({
            "suggestion": "max_results 必须调整到 1-20 区间",
            "fix": {"param": "max_results", "suggested_value": 10},
            "reason": "错误信息指明有效范围",
        }, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        hint = await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        assert hint["rule_id"] == seeded_rule
        assert json.loads(hint["fix"]) == {"param": "max_results", "suggested_value": 10}
        assert "max_results" in hint["suggestion"]
        assert hint["input_tokens"] == 100 and hint["output_tokens"] == 30
        assert len(calls) == 1

    async def test_idempotent_copy_on_hit(self, db_conn, seeded_rule, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "s", "fix": {"param": "p", "suggested_value": 1},
                               "reason": "r"}, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        # 第二条同内容规则（新 id）→ copy-on-hit，0 API 调用
        cursor = await db_conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_api', '1.0.0', 'range_rule', ?, ?)""",
            (_rule()["condition"], _rule()["action"]))
        rule2 = cursor.lastrowid
        await db_conn.commit()
        hint2 = await advisor.generate_for_rule(_rule(row_id=rule2))
        assert hint2["rule_id"] == rule2
        assert json.loads(hint2["fix"]) == {"param": "p", "suggested_value": 1}
        assert len(calls) == 1  # 只有第一轮 1 次 API 调用

    async def test_content_evolution_updates_and_regenerates(self, db_conn, seeded_rule, monkeypatch):
        # 同 rule_id、内容演进（condition 变化）→ 重新生成并 UPDATE 既有行
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "s", "fix": {"param": "p", "suggested_value": 1},
                               "reason": "r"}, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        await advisor.generate_for_rule(_rule(row_id=seeded_rule,
                                              condition={"param_names": ["max_results"]}))
        await advisor.generate_for_rule(_rule(row_id=seeded_rule,
                                              condition={"param_names": ["query"]}))
        assert len(calls) == 2
        cursor = await db_conn.execute(
            "SELECT COUNT(*) FROM repair_hints WHERE rule_id=?", (seeded_rule,))
        assert (await cursor.fetchone())[0] == 1  # UPDATE 不增行

    async def test_content_hash_key_order_independent(self, db_conn, seeded_rule, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "s", "fix": {"param": "p", "suggested_value": 1},
                               "reason": "r"}, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        cursor = await db_conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_api', '1.0.0', 'range_rule', ?, ?)""",
            (json.dumps({"b": 1, "a": 2}), _rule()["action"]))
        rule2 = cursor.lastrowid
        await db_conn.commit()
        await advisor.generate_for_rule(_rule(row_id=seeded_rule,
                                              condition={"a": 2, "b": 1}))
        await advisor.generate_for_rule(_rule(row_id=rule2,
                                              condition={"b": 1, "a": 2}))
        assert len(calls) == 1  # 键序不同但内容同 → 同 hash

    async def test_content_hash_includes_tool_name(self, db_conn, seeded_rule, monkeypatch):
        # 裁决⑤：同 condition+action 不同工具 → 不同 hash → 各自生成（跨工具不复用建议）
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "s", "fix": {"param": "p", "suggested_value": 1},
                               "reason": "r"}, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        cursor = await db_conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_fetch', '1.0.0', 'range_rule', ?, ?)""",
            (_rule()["condition"], _rule()["action"]))
        rule2 = cursor.lastrowid
        await db_conn.commit()
        await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        await advisor.generate_for_rule(_rule(row_id=rule2, tool="repair_fetch"))
        assert len(calls) == 2

    async def test_no_api_key_template_fallback(self, db_conn, seeded_rule, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", None)
        calls = []
        client = _mock_client({"suggestion": "s", "fix": None, "reason": "r"}, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        hint = await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        assert hint["fix"] is None
        assert "max_results" in hint["suggestion"]  # 模板含参数名
        assert len(calls) == 0  # 无 key 不发 HTTP

    async def test_transport_error_retries_then_fallback(self, db_conn, seeded_rule, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "s", "fix": {"param": "p", "suggested_value": 1},
                               "reason": "r"}, calls, fail_times=3)
        advisor = RepairAdvisor(db_conn, client=client)
        hint = await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        assert hint["fix"] is None
        assert len(calls) == 3  # 1 + 2 次重试全部失败 → 降级

    async def test_invalid_json_fallback(self, db_conn, seeded_rule, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client(None, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        hint = await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        assert hint["fix"] is None
        assert "max_results" in hint["suggestion"]

    async def test_fix_null_falls_back_to_template(self, db_conn, seeded_rule, monkeypatch):
        # spec 明文：fix 为 null → 回退模板建议文本（不保留 LLM suggestion）
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "无法给出确定性修复",
                               "fix": None, "reason": "信息不足"}, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        hint = await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        assert hint["fix"] is None
        assert "max_results" in hint["suggestion"]
        assert hint["suggestion"] != "无法给出确定性修复"


class TestRepairAdvisorQuery:
    async def test_get_hint_found_and_missing(self, db_conn, seeded_rule, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "s", "fix": {"param": "p", "suggested_value": 1},
                               "reason": "r"}, calls)
        advisor = RepairAdvisor(db_conn, client=client)
        await advisor.generate_for_rule(_rule(row_id=seeded_rule))
        hint = await advisor.get_hint(seeded_rule)
        assert hint is not None and json.loads(hint["fix"])["param"] == "p"
        assert await advisor.get_hint(9999) is None


class TestRepairAdvisorStringParams:
    async def test_template_fallback_with_json_string_params(self, db_conn, monkeypatch):
        # 跨批接口回归：run_demo/eval 的 examples 来自 trajectories 表（params 为 JSON 字符串），
        # 非 param_error 规则（timeout_rule 等）无 param_names 时走兜底——必须解析字符串而非 .keys() 崩溃
        monkeypatch.setattr(settings, "deepseek_api_key", None)
        cursor = await db_conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_fetch', '1.0.0', 'timeout_rule', '{"on_error": "timeout"}', '{}')""")
        rule_id = cursor.lastrowid
        await db_conn.commit()
        advisor = RepairAdvisor(db_conn)
        hint = await advisor.generate_for_rule(
            _rule(row_id=rule_id, tool="repair_fetch", rule_type="timeout_rule",
                  condition={"on_error": "timeout"}, action={}),
            examples=[{"params": '{"timeout_ms": 1000}',
                       "error_message": "timeout_ms must be at least 5000, got 1000"}])
        assert hint["fix"] is None
        assert "timeout_ms" in hint["suggestion"]

    async def test_call_llm_parses_string_params_for_prompt(self, db_conn, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
        calls = []
        client = _mock_client({"suggestion": "s",
                               "fix": {"param": "timeout_ms", "suggested_value": 6000},
                               "reason": "r"}, calls)
        cursor = await db_conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_fetch', '1.0.0', 'timeout_rule', '{"on_error": "timeout"}', '{}')""")
        rule_id = cursor.lastrowid
        await db_conn.commit()
        advisor = RepairAdvisor(db_conn, client=client)
        hint = await advisor.generate_for_rule(
            _rule(row_id=rule_id, tool="repair_fetch", rule_type="timeout_rule",
                  condition={"on_error": "timeout"}, action={}),
            examples=[{"params": '{"timeout_ms": 1000}',
                       "error_message": "timeout_ms must be at least 5000, got 1000"}])
        body = json.loads(calls[0].content)
        payload = json.loads(body["messages"][1]["content"])
        assert payload["param_names"] == ["timeout_ms"]
        assert payload["example_params"] == [{"timeout_ms": 1000}]
        assert json.loads(hint["fix"])["param"] == "timeout_ms"


class TestRepairAdvisorBatch:
    async def test_concurrency_limited(self, db_conn, monkeypatch):
        monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")

        class TrackingTransport(httpx.AsyncBaseTransport):
            def __init__(self):
                self.inflight = 0
                self.max_inflight = 0

            async def handle_async_request(self, request):
                self.inflight += 1
                self.max_inflight = max(self.max_inflight, self.inflight)
                await asyncio.sleep(0.01)
                self.inflight -= 1
                return httpx.Response(200, json={
                    "choices": [{"message": {"content": json.dumps({
                        "suggestion": "s",
                        "fix": {"param": "p", "suggested_value": 1},
                        "reason": "r"})}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                })

        transport = TrackingTransport()
        client = httpx.AsyncClient(transport=transport)
        advisor = RepairAdvisor(db_conn, client=client, concurrency=4)
        rules = []
        for i in range(10):
            cursor = await db_conn.execute(
                """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
                   VALUES (?, '1.0.0', 'range_rule', ?, ?)""",
                (f"tool_{i}", json.dumps({"param_names": [f"p{i}"]}),
                 json.dumps({"validate_before_call": True, "error_msg_template": f"p{i} bad"})))
            rules.append(_rule(row_id=cursor.lastrowid, tool=f"tool_{i}",
                               condition={"param_names": [f"p{i}"]},
                               action={"validate_before_call": True,
                                       "error_msg_template": f"p{i} bad"}))
        await db_conn.commit()
        stats = await advisor.generate_for_rules(rules)
        assert stats["generated"] == 10
        assert transport.max_inflight == 4  # 精确断言：并发上限真实生效（串行实现只有 1）
