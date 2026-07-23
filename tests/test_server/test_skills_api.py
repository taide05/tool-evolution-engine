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


class TestSkillsAPI:
    async def test_list_discoveries_empty(self, client):
        r = await client.get("/api/skills/discoveries")
        assert r.status_code == 200
        assert r.json()["discoveries"] == []

    async def test_list_deployed_empty(self, client):
        r = await client.get("/api/skills/deployed")
        assert r.status_code == 200
        assert r.json()["skills"] == []

    async def test_list_rules(self, client):
        r = await client.get("/api/rules")
        assert r.status_code == 200
        assert "rules" in r.json()
