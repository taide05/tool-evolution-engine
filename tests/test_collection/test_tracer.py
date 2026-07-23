import pytest
import asyncio
from tool_evolution.collection.tracer import Tracer
from tool_evolution.collection.schemas import ErrorType


@pytest.fixture
async def tracer(db_conn):
    t = Tracer(db_conn, batch_size=3, flush_interval_s=60)
    yield t
    await t.close()


class TestTracer:
    async def test_start_trace_returns_report(self, tracer):
        report = tracer.start_trace(
            agent_id="researcher", tool_name="search_law",
            params={"query": "劳动合同法"}
        )
        assert report.trace_id is not None
        assert report.agent_id == "researcher"
        assert report.tool_name == "search_law"
        assert report.params == {"query": "劳动合同法"}

    async def test_report_success_flushes(self, tracer):
        report = tracer.start_trace("a", "t", params={})
        report.success = True
        report.result = {"found": 5}
        report.latency_ms = 100
        await tracer.report(report)
        await tracer.flush()
        traces = await tracer.store.get_by_tool("t", limit=10)
        assert len(traces) == 1
        assert traces[0]["success"] == 1

    async def test_batch_auto_flush(self, tracer):
        for i in range(tracer.batch_size):
            report = tracer.start_trace("a", "t", params={})
            report.success = True
            report.latency_ms = 10
            await tracer.report(report)
        await asyncio.sleep(0.1)
        traces = await tracer.store.get_by_tool("t", limit=10)
        assert len(traces) == tracer.batch_size

    async def test_report_failure_with_error(self, tracer):
        report = tracer.start_trace("a", "t", params={})
        report.success = False
        report.latency_ms = 5000
        report.error_type = ErrorType.TIMEOUT
        report.error_message = "timeout"
        await tracer.report(report)
        await tracer.flush()
        count = await tracer.store.count_failures(ErrorType.TIMEOUT)
        assert count == 1

    async def test_trace_id_uniqueness(self, tracer):
        ids = set()
        for _ in range(100):
            r = tracer.start_trace("a", "t", params={})
            ids.add(r.trace_id)
        assert len(ids) == 100
