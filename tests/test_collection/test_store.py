import pytest
import pytest_asyncio
import aiosqlite
from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport, ErrorType, TraceType


@pytest_asyncio.fixture
async def store(db_conn):
    return TraceStore(db_conn)


class TestTraceStore:
    async def test_insert_and_retrieve_success(self, store):
        r = TraceReport(
            trace_id="t1", agent_id="a1", tool_name="search",
            success=True, latency_ms=100, token_count=50
        )
        await store.insert(r)
        rows = await store.get_by_tool("search", limit=10)
        assert len(rows) == 1
        assert rows[0]["trace_id"] == "t1"
        assert rows[0]["success"] == 1

    async def test_insert_failure_with_error_type(self, store):
        r = TraceReport(
            trace_id="t2", agent_id="a1", tool_name="search",
            success=False, latency_ms=5000,
            error_type=ErrorType.TIMEOUT, error_message="timed out"
        )
        await store.insert(r)
        count = await store.count_failures(ErrorType.TIMEOUT)
        assert count == 1

    async def test_task_tree(self, store):
        root = TraceReport(
            trace_id="root-1", agent_id="orch", tool_name="run_check",
            trace_type=TraceType.TASK_ROOT, success=True, latency_ms=5000
        )
        c1 = TraceReport(
            trace_id="c1", parent_trace_id="root-1", agent_id="a1",
            tool_name="search", success=True, latency_ms=200
        )
        c2 = TraceReport(
            trace_id="c2", parent_trace_id="root-1", agent_id="a2",
            tool_name="analyze", success=True, latency_ms=300
        )
        for r in [root, c1, c2]:
            await store.insert(r)
        tree = await store.get_task_tree("root-1")
        assert len(tree) == 3
        trace_ids = {t["trace_id"] for t in tree}
        assert trace_ids == {"root-1", "c1", "c2"}

    async def test_fts_search(self, store):
        r = TraceReport(
            trace_id="t3", agent_id="a1", tool_name="github_api",
            success=False, latency_ms=1000,
            error_type=ErrorType.QUOTA_EXHAUSTED,
            error_message="API rate limit exceeded for github_api"
        )
        await store.insert(r)
        results = await store.search("rate limit")
        assert len(results) >= 1

    async def test_count_all_failures(self, store):
        for i in range(3):
            r = TraceReport(
                trace_id=f"f{i}", agent_id="a", tool_name="t",
                success=False, latency_ms=100,
                error_type=ErrorType.PARAM_ERROR
            )
            await store.insert(r)
        count = await store.count_failures(None)
        assert count == 3

    async def test_get_success_params(self, store):
        for i in range(5):
            r = TraceReport(
                trace_id=f"s{i}", agent_id="a", tool_name="search",
                tool_version="1.0.0", success=True, latency_ms=100,
                params={"query": f"test {i}", "max_results": 10 + i}
            )
            await store.insert(r)
        params = await store.get_success_params("search", "1.0.0", limit=10)
        assert len(params) == 5
        assert "max_results" in params[0]

    async def test_insert_stores_source(self, store):
        r = TraceReport(
            trace_id="src-1", agent_id="a", tool_name="search",
            success=True, latency_ms=100, source="synthetic_demo"
        )
        await store.insert(r)
        rows = await store.get_by_tool("search", limit=10)
        assert rows[0]["source"] == "synthetic_demo"

    async def test_insert_atomic_rolls_back_on_fts_failure(self, store):
        from tool_evolution.utils.database import init_db
        await store.conn.execute("DROP TABLE trajectories_fts")
        await store.conn.commit()
        r = TraceReport(trace_id="atomic-1", agent_id="a", tool_name="t",
                        success=True, latency_ms=10)
        with pytest.raises(aiosqlite.OperationalError):
            await store.insert(r)
        rows = await store.get_by_tool("t", limit=10)
        assert rows == []
        # 正向对照：重建 FTS 后同一报告可成功插入——锁定"失败即回滚"而非"失败即永久拒插"
        await init_db(store.conn)
        await store.insert(TraceReport(trace_id="atomic-1", agent_id="a", tool_name="t",
                                       success=True, latency_ms=10))
        rows = await store.get_by_tool("t", limit=10)
        assert len(rows) == 1

    async def test_search_escapes_fts_syntax(self, store):
        r = TraceReport(
            trace_id="esc-1", agent_id="a", tool_name="github_api",
            success=False, latency_ms=1000,
            error_type=ErrorType.QUOTA_EXHAUSTED,
            error_message="API rate limit exceeded"
        )
        await store.insert(r)
        results = await store.search('" OR 1=1 --')
        assert results == []
        results2 = await store.search("rate limit")
        assert len(results2) >= 1
