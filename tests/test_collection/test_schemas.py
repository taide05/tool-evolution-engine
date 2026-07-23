import pytest
from tool_evolution.collection.schemas import TraceReport, TraceSnapshot, TraceType, ErrorType


class TestTraceReport:
    def test_minimal_success_report(self):
        r = TraceReport(
            trace_id="abc-123", agent_id="researcher",
            tool_name="search_law", success=True, latency_ms=42
        )
        assert r.trace_id == "abc-123"
        assert r.success is True
        assert r.error_type is None

    def test_failure_report_with_error(self):
        r = TraceReport(
            trace_id="abc-456", agent_id="researcher",
            tool_name="search_law", success=False, latency_ms=5000,
            error_type=ErrorType.TIMEOUT, error_message="Request timed out after 5s"
        )
        assert r.error_type == ErrorType.TIMEOUT
        assert "timed out" in r.error_message

    def test_parent_trace_chain(self):
        root = TraceReport(
            trace_id="root-1", agent_id="orchestrator",
            tool_name="run_compliance_check", trace_type=TraceType.TASK_ROOT,
            success=True, latency_ms=10000
        )
        child = TraceReport(
            trace_id="child-1", parent_trace_id="root-1", agent_id="researcher",
            tool_name="search_law", trace_type=TraceType.ATOMIC,
            success=True, latency_ms=200
        )
        assert root.trace_type == TraceType.TASK_ROOT
        assert root.parent_trace_id is None
        assert child.parent_trace_id == "root-1"
        assert child.trace_type == TraceType.ATOMIC

    def test_defaults(self):
        r = TraceReport(
            trace_id="t1", agent_id="a", tool_name="t",
            success=True, latency_ms=10
        )
        assert r.tool_version == "1.0.0"
        assert r.trace_type == TraceType.ATOMIC
        assert r.params == {}
        assert r.token_count == 0


class TestTraceSnapshot:
    def test_snapshot_extends_report(self):
        snap = TraceSnapshot(
            trace_id="t1", agent_id="a", tool_name="t",
            success=True, latency_ms=10
        )
        assert snap.created_at is not None
        assert snap.tool_version == "1.0.0"
