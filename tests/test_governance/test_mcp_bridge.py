import json
import time

import pytest
from tool_evolution.governance.mcp_bridge import MCPBridge, mcp, set_bridge
from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport, TraceType


@pytest.fixture
async def bridge(db_conn):
    return MCPBridge(db_conn)


class TestMCPBridge:
    async def test_update_and_search_memory(self, bridge):
        await bridge.update_memory("产品手册", ["配置说明", "依赖关系"])
        result = await bridge.search_memory("产品手册")
        assert len(result) >= 1
        item = next((e for e in result if e["entity"] == "产品手册"), None)
        assert item is not None
        assert "配置说明" in item["relations"]

    async def test_get_user_preferences_empty(self, bridge):
        prefs = await bridge.get_user_preferences()
        assert isinstance(prefs, dict)

    async def test_memory_cache_crud(self, bridge):
        await bridge._set_cache("test_key", {"value": 42})
        val = await bridge._get_cache("test_key")
        assert val == {"value": 42}

    async def test_search_memory_miss(self, bridge):
        result = await bridge.search_memory("nonexistent_entity_xyz")
        assert result == []

    async def test_update_memory_overwrites(self, bridge):
        await bridge.update_memory("E1", ["R1"])
        await bridge.update_memory("E1", ["R2", "R3"])
        result = await bridge.search_memory("E1")
        assert len(result) == 1
        assert result[0]["relations"] == ["R2", "R3"]

    async def test_extract_entities_from_trace_result(self, bridge):
        trace = {
            "success": True,
            "result": {"entity": "entity-a", "title": "Title B"},
            "tool_name": "search_api",
        }
        await bridge.extract_and_update(trace)
        results = await bridge.search_memory("entity-a")
        assert len(results) == 1
        results_b = await bridge.search_memory("Title B")
        assert len(results_b) == 1

    async def test_extract_skips_failed_trace(self, bridge):
        trace = {"success": False, "result": {"entity": "should-not-store"}}
        await bridge.extract_and_update(trace)
        results = await bridge.search_memory("should-not-store")
        assert results == []

    async def test_mcp_tools_registered(self, bridge):
        set_bridge(bridge)
        tools = mcp._tool_manager._tools
        names = {t.name for t in tools.values()}
        assert {"search_memory", "update_memory", "get_user_preferences",
                "search_relations", "get_repair_hint"} <= names


class TestMCPToolLatency:
    async def test_five_tools_under_1s(self, bridge, db_conn):
        # I#3: MCP 工具桥方法级耗时实测（in-process；stdio 传输开销为协议层不计）
        ts = TraceStore(db_conn)
        await ts.insert(TraceReport(trace_id="lat-root", agent_id="a", tool_name="task",
                                    trace_type=TraceType.TASK_ROOT, success=True, latency_ms=0))
        await ts.insert(TraceReport(trace_id="lat-c0", parent_trace_id="lat-root",
                                    agent_id="a", tool_name="t", success=True,
                                    latency_ms=1, result={"entity": "LatA"}))
        await ts.insert(TraceReport(trace_id="lat-c1", parent_trace_id="lat-root",
                                    agent_id="a", tool_name="t", success=True,
                                    latency_ms=1, result={"title": "LatB"}))
        await bridge.extract_relations("lat-root")
        cur = await db_conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('lat_api', '1.0.0', 'range_rule', '{}', '{}')""")
        rule_id = cur.lastrowid
        await db_conn.execute(
            """INSERT INTO repair_hints (rule_id, content_hash, suggestion, fix, model)
               VALUES (?, 'lat-hash', '检查参数', ?, 'deepseek-v4-flash')""",
            (rule_id, json.dumps({"param": "max_results", "suggested_value": 10})))
        await db_conn.commit()
        calls = {
            "search_memory": lambda: bridge.search_memory("LatA"),
            "update_memory": lambda: bridge.update_memory("LatA", ["LatB"]),
            "get_user_preferences": lambda: bridge.get_user_preferences(),
            "search_relations": lambda: bridge.search_relations("LatA"),
            "get_repair_hint": lambda: bridge.get_repair_hint(rule_id),
        }
        timings = {}
        for name, call in calls.items():
            t0 = time.perf_counter()
            await call()
            timings[name] = round(time.perf_counter() - t0, 4)
        assert all(t < 1.0 for t in timings.values()), f"工具耗时超限: {timings}"


class TestMCPBridgeRepairHint:
    async def test_get_repair_hint_parses_fix(self, bridge, db_conn):
        cur = await db_conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_api', '1.0.0', 'range_rule', '{}', '{}')""")
        rule_id = cur.lastrowid
        await db_conn.execute(
            """INSERT INTO repair_hints (rule_id, content_hash, suggestion, fix, model)
               VALUES (?, 'abc', '检查 max_results 取值范围', ?, 'deepseek-v4-flash')""",
            (rule_id, json.dumps({"param": "max_results", "suggested_value": 10})))
        await db_conn.commit()
        hint = await bridge.get_repair_hint(rule_id)
        assert hint["fix"] == {"param": "max_results", "suggested_value": 10}

    async def test_get_repair_hint_missing(self, bridge, db_conn):
        assert await bridge.get_repair_hint(9999) is None


class TestMCPBridgeRelations:
    async def test_extract_relations_delegates(self, bridge, db_conn):
        ts = TraceStore(db_conn)
        await ts.insert(TraceReport(trace_id="r1", agent_id="a", tool_name="task",
                                    trace_type=TraceType.TASK_ROOT, success=True, latency_ms=0))
        await ts.insert(TraceReport(trace_id="x1", parent_trace_id="r1", agent_id="a",
                                    tool_name="t", success=True, latency_ms=1,
                                    result={"entity": "E1"}))
        await ts.insert(TraceReport(trace_id="x2", parent_trace_id="r1", agent_id="a",
                                    tool_name="t", success=True, latency_ms=1,
                                    result={"title": "E2"}))
        count = await bridge.extract_relations("r1")
        assert count == 1
        rows = await bridge.search_relations("E1")
        assert len(rows) == 1
        assert rows[0]["target_entity"] == "E2"
