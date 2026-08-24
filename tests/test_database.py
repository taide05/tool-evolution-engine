import pytest
import aiosqlite
from tool_evolution.utils.database import (
    get_connection,
    transaction,
    init_db,
    run_migrations,
    CURRENT_SCHEMA_VERSION,
)


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


@pytest.mark.asyncio
async def test_fresh_db_gets_latest_version(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    await init_db(conn)
    await run_migrations(conn)
    cursor = await conn.execute("SELECT version FROM schema_meta")
    assert (await cursor.fetchone())[0] == CURRENT_SCHEMA_VERSION
    cursor = await conn.execute("PRAGMA table_info(trajectories)")
    names = [row["name"] for row in await cursor.fetchall()]
    assert "source" in names
    await conn.close()


@pytest.mark.asyncio
async def test_v1_db_migrates(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    try:
        # 手工建完整 v1 形状库（git 历史真实 14 列，无 source 列、无 schema_meta）
        await conn.executescript("""
            CREATE TABLE trajectories (
                trace_id TEXT PRIMARY KEY,
                parent_trace_id TEXT,
                agent_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tool_version TEXT NOT NULL DEFAULT '1.0.0',
                trace_type TEXT NOT NULL CHECK(trace_type IN ('atomic','task_root')),
                params TEXT NOT NULL,
                success INTEGER NOT NULL,
                result TEXT,
                error_type TEXT,
                error_message TEXT,
                latency_ms INTEGER NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        await conn.execute(
            """INSERT INTO trajectories (trace_id, agent_id, tool_name, trace_type, params,
               success, latency_ms) VALUES ('v1-row', 'a', 't', 'atomic', '{}', 1, 100)"""
        )
        await conn.commit()
        await run_migrations(conn)
        cursor = await conn.execute("PRAGMA table_info(trajectories)")
        names = [row["name"] for row in await cursor.fetchall()]
        assert "source" in names
        cursor = await conn.execute("SELECT version FROM schema_meta")
        assert (await cursor.fetchone())[0] == CURRENT_SCHEMA_VERSION
        # 数据保留断言：迁移后老行存活且 source 取 DEFAULT
        cursor = await conn.execute("SELECT source FROM trajectories WHERE trace_id='v1-row'")
        assert (await cursor.fetchone())[0] == "synthetic"
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='repair_hints'"
        )
        assert await cursor.fetchone() is not None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_migration_idempotent(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    await init_db(conn)
    await run_migrations(conn)
    await run_migrations(conn)  # 第二次不报错
    cursor = await conn.execute("SELECT version FROM schema_meta")
    assert (await cursor.fetchone())[0] == CURRENT_SCHEMA_VERSION
    await conn.close()


@pytest.mark.asyncio
async def test_newer_db_rejected(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    try:
        await conn.execute("CREATE TABLE schema_meta (version INTEGER NOT NULL)")
        await conn.execute("INSERT INTO schema_meta (version) VALUES (99)")
        await conn.commit()
        with pytest.raises(RuntimeError, match="newer"):
            await run_migrations(conn)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_repair_hints_cascade_on_rule_delete(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    try:
        await init_db(conn)
        cursor = await conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_api', '1.0.0', 'range_rule', '{}', '{}')"""
        )
        rule_id = cursor.lastrowid
        await conn.execute(
            """INSERT INTO repair_hints (rule_id, content_hash, suggestion, fix, model)
               VALUES (?, 'abc123', '检查参数', NULL, 'deepseek-chat')""",
            (rule_id,)
        )
        await conn.commit()
        await conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
        await conn.commit()
        cursor = await conn.execute("SELECT COUNT(*) FROM repair_hints")
        assert (await cursor.fetchone())[0] == 0
    finally:
        # 红灯期失败会遗留打开的 WAL 连接，teardown 死锁挂进程——必须保证关闭
        await conn.close()


@pytest.mark.asyncio
async def test_fresh_db_has_repair_hints(tmp_path, monkeypatch):
    from tool_evolution.utils.config import settings
    monkeypatch.setattr(settings, "db_path", tmp_path / "engine.db")
    conn = await get_connection()
    try:
        await init_db(conn)
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='repair_hints'"
        )
        assert await cursor.fetchone() is not None
    finally:
        await conn.close()
