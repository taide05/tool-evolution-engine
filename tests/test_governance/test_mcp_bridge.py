import pytest
from tool_evolution.governance.mcp_bridge import MCPBridge, mcp, set_bridge


@pytest.fixture
async def bridge(db_conn):
    return MCPBridge(db_conn)


class TestMCPBridge:
    async def test_update_and_search_memory(self, bridge):
        await bridge.update_memory("劳动合同法", ["经济补偿", "解除劳动关系"])
        result = await bridge.search_memory("劳动合同法")
        assert len(result) >= 1
        item = next((e for e in result if e["entity"] == "劳动合同法"), None)
        assert item is not None
        assert "经济补偿" in item["relations"]

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
            "tool_name": "search_law",
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
        tool_names = {name for name in tools if not name.startswith("_")}
        assert "search_memory" in tool_names
        assert "update_memory" in tool_names
        assert "get_user_preferences" in tool_names
