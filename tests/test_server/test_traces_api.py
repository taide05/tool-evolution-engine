import pytest
import aiosqlite
from httpx import AsyncClient, ASGITransport
from tool_evolution.server import app as server_app
from tool_evolution.server.app import app
from tool_evolution.utils.database import init_db


@pytest.fixture(autouse=True)
async def setup_db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await init_db(conn)
    server_app._conn = conn
    yield
    await conn.close()
    server_app._conn = None


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


class TestTracesAPI:
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    async def test_report_trace(self, client):
        import uuid
        r = await client.post("/api/traces/report", json={
            "trace_id": f"api-{uuid.uuid4().hex[:8]}", "agent_id": "test", "tool_name": "search",
            "success": True, "latency_ms": 42
        })
        assert r.status_code == 200

    async def test_seed_traces(self, client):
        import uuid
        sid1, sid2 = f"seed-{uuid.uuid4().hex[:8]}", f"seed-{uuid.uuid4().hex[:8]}"
        r = await client.post("/api/traces/seed", json=[
            {"trace_id": sid1, "agent_id": "test", "tool_name": "search", "success": True, "latency_ms": 10},
            {"trace_id": sid2, "agent_id": "test", "tool_name": "search", "success": False, "latency_ms": 500,
             "error_type": "timeout", "error_message": "timeout"},
        ])
        assert r.status_code == 200
        assert r.json()["count"] == 2

    async def test_analytics_summary(self, client):
        r = await client.get("/api/analytics/summary")
        assert r.status_code == 200
        data = r.json()
        assert "total_traces" in data
