import pytest
import aiosqlite
from tool_evolution.utils.database import get_connection, transaction


@pytest.mark.asyncio
async def test_transaction_commits(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    await conn.commit()
    async with transaction(conn):
        await conn.execute("INSERT INTO t (v) VALUES ('a')")
        await conn.execute("INSERT INTO t (v) VALUES ('b')")
    cursor = await conn.execute("SELECT COUNT(*) FROM t")
    assert (await cursor.fetchone())[0] == 2
    await conn.close()


@pytest.mark.asyncio
async def test_transaction_rolls_back_on_error(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    await conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    await conn.commit()
    with pytest.raises(aiosqlite.OperationalError):
        async with transaction(conn):
            await conn.execute("INSERT INTO t (v) VALUES ('a')")
            await conn.execute("INSERT INTO nonexistent (v) VALUES ('b')")
    cursor = await conn.execute("SELECT COUNT(*) FROM t")
    assert (await cursor.fetchone())[0] == 0
    await conn.close()
