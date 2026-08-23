import asyncio
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from tool_evolution.server.app import app
from tool_evolution.utils.config import settings
from tool_evolution.utils.database import get_connection, init_db


@pytest.mark.asyncio
async def test_concurrent_reports_file_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    monkeypatch.setattr(settings, "api_key", "test-key")
    conn = await get_connection()
    await init_db(conn)
    await conn.close()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"X-API-Key": "test-key"}) as ac:
        results = await asyncio.gather(*[
            ac.post("/api/traces/report", json={
                "trace_id": f"conc-{uuid.uuid4().hex[:8]}", "agent_id": "test",
                "tool_name": "search", "success": True, "latency_ms": 42,
            })
            for _ in range(10)
        ])
    assert all(r.status_code == 200 for r in results)

    verify = await get_connection()
    cursor = await verify.execute("SELECT COUNT(*) FROM trajectories")
    assert (await cursor.fetchone())[0] == 10
    await verify.close()
