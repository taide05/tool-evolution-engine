from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport
from tool_evolution.knowledge.param_template import ParamTemplateManager
from tool_evolution.governance.mcp_bridge import MCPBridge


class TestTemplatesApi:
    async def test_generate_applies_cached_prefs(self, client, setup_db):
        mgr = ParamTemplateManager(setup_db)
        await mgr.save("search_api", "1.0.0", {
            "max_results": {"param_type": "int", "default_value": 10,
                            "lower_bound": 0, "upper_bound": 100, "sample_count": 200},
        })
        ts = TraceStore(setup_db)
        for i in range(40):
            await ts.insert(TraceReport(
                trace_id=f"tp{i}", agent_id="a", tool_name="search_api",
                tool_version="1.0.0", success=True, latency_ms=5,
                params={"max_results": 10},
            ))
        bridge = MCPBridge(setup_db)
        await bridge._set_cache("user_preferences", {"search_api": {"max_results": 25}})

        resp = await client.post("/api/templates/generate",
                                 json={"tool_name": "search_api", "tool_version": "1.0.0"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["template"]["max_results"]["default_value"] == 25
        assert data["template"]["max_results"]["source"] == "user_preference"
        assert data["prefs_applied"] == {"max_results": 25}
