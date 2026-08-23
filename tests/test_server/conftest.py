import pytest
import aiosqlite
from httpx import AsyncClient, ASGITransport
from tool_evolution.server.app import app
from tool_evolution.server.deps import get_db
from tool_evolution.utils.config import settings
from tool_evolution.utils.database import init_db


@pytest.fixture(autouse=True)
async def setup_db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await init_db(conn)

    async def _override():
        yield conn

    app.dependency_overrides[get_db] = _override
    yield conn
    app.dependency_overrides.pop(get_db, None)
    await conn.close()


@pytest.fixture(autouse=True)
def auth_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "test-key")


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test",
                           headers={"X-API-Key": "test-key"}) as ac:
        yield ac
