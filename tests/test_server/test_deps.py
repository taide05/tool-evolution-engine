import pytest
from tool_evolution.server.deps import get_db
from tool_evolution.utils.config import settings


@pytest.mark.asyncio
async def test_get_db_yields_working_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    gen = get_db()
    conn = await gen.__anext__()
    cursor = await conn.execute("SELECT 1")
    assert (await cursor.fetchone())[0] == 1
    await gen.aclose()
    with pytest.raises(Exception):
        await conn.execute("SELECT 1")
