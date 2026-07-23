import pytest
import aiosqlite
from pathlib import Path
from tool_evolution.utils.database import init_db


@pytest.fixture
async def db_conn():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await init_db(conn)
    yield conn
    await conn.close()
