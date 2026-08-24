import json
import pytest
from tool_evolution.analysis.preference_learner import PreferenceLearner
from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport
from tool_evolution.knowledge.param_template import ParamTemplateManager


@pytest.fixture
async def learner(db_conn):
    return PreferenceLearner(db_conn)


async def _seed_global_mode(db_conn, tool, param, value):
    mgr = ParamTemplateManager(db_conn)
    await mgr.save(tool, "1.0.0", {param: {"param_type": "int", "default_value": value,
                                            "lower_bound": 0, "upper_bound": 100,
                                            "sample_count": 200}})


async def _seed_agent(db_conn, agent_id, tool, values):
    ts = TraceStore(db_conn)
    for i, v in enumerate(values):
        await ts.insert(TraceReport(
            trace_id=f"{agent_id}-{i}", agent_id=agent_id, tool_name=tool,
            tool_version="1.0.0", success=True, latency_ms=5,
            params={"max_results": v},
        ))


class TestPreferenceLearner:
    async def test_learns_skewed_preference(self, learner, db_conn):
        await _seed_global_mode(db_conn, "search_api", "max_results", 10)
        await _seed_agent(db_conn, "agent_p1", "search_api", [20] * 40)
        prefs = await learner.learn()
        assert prefs["search_api"]["max_results"] == 20

    async def test_ignores_matching_global(self, learner, db_conn):
        await _seed_global_mode(db_conn, "search_api", "max_results", 10)
        await _seed_agent(db_conn, "agent_g", "search_api", [10] * 30 + [11] * 10)
        prefs = await learner.learn()
        assert "search_api" not in prefs

    async def test_single_trace_not_learned(self, learner, db_conn):
        await _seed_global_mode(db_conn, "search_api", "max_results", 10)
        await _seed_agent(db_conn, "agent_solo", "search_api", [20])
        prefs = await learner.learn()
        assert "search_api" not in prefs

    async def test_below_share_threshold_not_learned(self, learner, db_conn):
        await _seed_global_mode(db_conn, "search_api", "max_results", 10)
        await _seed_agent(db_conn, "agent_mix", "search_api", [20] * 11 + [10] * 10)
        prefs = await learner.learn()
        assert "search_api" not in prefs

    async def test_exact_60pct_share_not_learned(self, learner, db_conn):
        # spec 阈值是严格 >60%——恰好 60% 不触发
        await _seed_global_mode(db_conn, "search_api", "max_results", 10)
        await _seed_agent(db_conn, "agent_60", "search_api", [20] * 12 + [10] * 8)
        prefs = await learner.learn()
        assert "search_api" not in prefs

    async def test_excludes_executor_agents(self, learner, db_conn):
        await _seed_global_mode(db_conn, "search_api", "max_results", 10)
        await _seed_agent(db_conn, "executor:demo", "search_api", [20] * 40)
        prefs = await learner.learn()
        assert prefs == {}

    async def test_save_to_cache_roundtrip(self, learner, db_conn):
        await learner.save_to_cache({"search_api": {"max_results": 20}})
        cursor = await db_conn.execute(
            "SELECT value FROM memory_cache WHERE key='user_preferences'"
        )
        row = await cursor.fetchone()
        assert json.loads(row[0]) == {"search_api": {"max_results": 20}}
