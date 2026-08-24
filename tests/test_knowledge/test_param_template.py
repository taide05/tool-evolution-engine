import pytest
from tool_evolution.knowledge.param_template import ParamTemplateManager


@pytest.fixture
async def manager(db_conn):
    return ParamTemplateManager(db_conn)


class TestParamTemplateManager:
    async def test_save_and_get_template(self, manager):
        dists = {
            "max_results": {"param_type": "int", "default_value": 10,
                           "lower_bound": 1, "upper_bound": 100, "sample_count": 50},
            "lang": {"param_type": "str", "default_value": "zh", "sample_count": 50},
        }
        await manager.save("search", "1.0.0", dists)
        tmpl = await manager.get_template("search", "1.0.0")
        assert tmpl is not None
        assert tmpl["max_results"]["default_value"] == 10
        assert tmpl["lang"]["default_value"] == "zh"

    async def test_get_nonexistent_returns_none(self, manager):
        tmpl = await manager.get_template("nonexistent", "1.0.0")
        assert tmpl is None

    async def test_save_updates_existing(self, manager):
        dists = {"x": {"param_type": "int", "default_value": 5, "sample_count": 30}}
        await manager.save("t", "1.0.0", dists)
        dists2 = {"x": {"param_type": "int", "default_value": 8, "sample_count": 60}}
        await manager.save("t", "1.0.0", dists2)
        tmpl = await manager.get_template("t", "1.0.0")
        assert tmpl["x"]["default_value"] == 8
        assert tmpl["x"]["sample_count"] == 60

    async def test_generate_from_store(self, manager, db_conn):
        from tool_evolution.collection.store import TraceStore
        from tool_evolution.collection.schemas import TraceReport

        store = TraceStore(db_conn)
        for i in range(40):
            await store.insert(TraceReport(
                trace_id=f"g{i}", agent_id="a", tool_name="search",
                tool_version="1.0.0", success=True, latency_ms=10,
                params={"max_results": 10 + (i % 5)}
            ))
        tmpl = await manager.generate("search", "1.0.0")
        assert tmpl is not None
        assert "max_results" in tmpl

    async def test_generate_with_user_prefs(self, manager, db_conn):
        from tool_evolution.collection.store import TraceStore
        from tool_evolution.collection.schemas import TraceReport

        store = TraceStore(db_conn)
        for i in range(40):
            await store.insert(TraceReport(
                trace_id=f"p{i}", agent_id="a", tool_name="pref_tool",
                tool_version="1.0.0", success=True, latency_ms=10,
                params={"max_results": 10 + (i % 5)}
            ))
        tmpl = await manager.generate("pref_tool", "1.0.0",
                                       user_prefs={"max_results": 99})
        assert tmpl is not None
        assert tmpl["max_results"]["default_value"] == 99
        assert tmpl["max_results"]["source"] == "user_preference"


class TestFlattenUserPrefs:
    def test_unpacks_nested_by_tool(self):
        from tool_evolution.knowledge.param_template import flatten_user_prefs
        prefs = {"search": {"temperature": 0.7}, "other": {"x": 1}}
        assert flatten_user_prefs(prefs, "search") == {"temperature": 0.7}
        assert flatten_user_prefs(prefs, "missing") == {}
