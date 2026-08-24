from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport, TraceType
from tool_evolution.governance.mcp_bridge import MCPBridge


class TestMemoryRelationsApi:
    async def test_relations_endpoint(self, client, setup_db):
        ts = TraceStore(setup_db)
        await ts.insert(TraceReport(trace_id="api-root", agent_id="a", tool_name="task",
                                    trace_type=TraceType.TASK_ROOT, success=True, latency_ms=0))
        await ts.insert(TraceReport(trace_id="api-c0", parent_trace_id="api-root",
                                    agent_id="a", tool_name="t", success=True,
                                    latency_ms=1, result={"entity": "Alpha"}))
        await ts.insert(TraceReport(trace_id="api-c1", parent_trace_id="api-root",
                                    agent_id="a", tool_name="t", success=True,
                                    latency_ms=1, result={"title": "Beta"}))
        await MCPBridge(setup_db).extract_relations("api-root")

        resp = await client.get("/api/memory/relations", params={"entity": "Alpha"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"] == "Alpha"
        assert data["count"] == 1
        # 字典序：Alpha < Beta，方向无关断言
        assert {data["relations"][0]["source_entity"],
                data["relations"][0]["target_entity"]} == {"Alpha", "Beta"}

    async def test_relations_endpoint_miss(self, client):
        resp = await client.get("/api/memory/relations", params={"entity": "不存在"})
        assert resp.status_code == 200
        assert resp.json()["relations"] == []
