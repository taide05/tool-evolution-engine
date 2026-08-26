import aiosqlite
from pathlib import Path
from contextlib import asynccontextmanager
from .config import settings


async def get_connection() -> aiosqlite.Connection:
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@asynccontextmanager
async def transaction(conn: aiosqlite.Connection):
    await conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        await conn.commit()
    except BaseException:
        await conn.rollback()
        raise


CURRENT_SCHEMA_VERSION = 5

# version: (ddl, table_to_check, column_to_check)
MIGRATIONS: dict[int, tuple[str, str, str]] = {
    2: ("ALTER TABLE trajectories ADD COLUMN source TEXT NOT NULL DEFAULT 'synthetic'",
        "trajectories", "source"),
    3: ("""CREATE TABLE IF NOT EXISTS entity_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL DEFAULT 'co_occur',
            strength INTEGER NOT NULL DEFAULT 1,
            evidence_trace_ids TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_er_source ON entity_relations(source_entity);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_er_pair ON entity_relations(source_entity, target_entity, relation_type);""",
        "entity_relations", "source_entity"),
    4: ("""CREATE TABLE IF NOT EXISTS repair_hints (
            rule_id INTEGER PRIMARY KEY REFERENCES rules(id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            fix TEXT,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );""",
        "repair_hints", "rule_id"),
    5: ("""CREATE TABLE IF NOT EXISTS execution_tasks (
            task_id TEXT PRIMARY KEY,
            task_description TEXT NOT NULL,
            skill_name TEXT,
            mode TEXT NOT NULL CHECK(mode IN ('skill_plan','llm_plan')),
            plan TEXT,
            status TEXT NOT NULL CHECK(status IN ('pending','running','success','failed','cancelled')),
            summary TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS execution_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES execution_tasks(task_id),
            step_index INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            params TEXT,
            result TEXT,
            status TEXT NOT NULL,
            latency_ms INTEGER,
            tokens INTEGER NOT NULL DEFAULT 0,
            rules_triggered TEXT,
            repair_hint_applied TEXT,
            adapter TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_es_task ON execution_steps(task_id);""",
        "execution_tasks", "task_id"),
}


async def _column_exists(conn: aiosqlite.Connection, table: str, column: str) -> bool:
    cursor = await conn.execute(f"PRAGMA table_info({table})")
    return any(row["name"] == column for row in await cursor.fetchall())


async def run_migrations(conn: aiosqlite.Connection) -> None:
    await conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)")
    cursor = await conn.execute("SELECT version FROM schema_meta")
    row = await cursor.fetchone()
    if row is None:
        current = 1  # 无版本记录的库按 v1 处理
        await conn.execute("INSERT INTO schema_meta (version) VALUES (1)")
        await conn.commit()
    else:
        current = row["version"]
    if current > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema v{current} is newer than supported v{CURRENT_SCHEMA_VERSION}"
        )
    for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
        ddl, table, column = MIGRATIONS[version]
        async with transaction(conn):
            if not await _column_exists(conn, table, column):
                for stmt in ddl.split(";"):
                    if stmt.strip():
                        await conn.execute(stmt)
            await conn.execute("UPDATE schema_meta SET version=?", (version,))


async def init_db(conn: aiosqlite.Connection) -> None:
    await conn.executescript("""
        CREATE TABLE IF NOT EXISTS trajectories (
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
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            source TEXT NOT NULL DEFAULT 'synthetic'
        );
        CREATE INDEX IF NOT EXISTS idx_traj_tool ON trajectories(tool_name, tool_version);
        CREATE INDEX IF NOT EXISTS idx_traj_parent ON trajectories(parent_trace_id);
        CREATE INDEX IF NOT EXISTS idx_traj_success ON trajectories(success);
        CREATE INDEX IF NOT EXISTS idx_traj_created ON trajectories(created_at);
        CREATE VIRTUAL TABLE IF NOT EXISTS trajectories_fts USING fts5(
            tool_name, error_message, content='trajectories', content_rowid='rowid'
        );

        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            tool_version TEXT NOT NULL,
            rule_type TEXT NOT NULL,
            condition TEXT NOT NULL,
            action TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0,
            miss_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','deprecated')),
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_rules_tool ON rules(tool_name, tool_version);

        CREATE TABLE IF NOT EXISTS param_distributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            tool_version TEXT NOT NULL,
            param_name TEXT NOT NULL,
            param_type TEXT NOT NULL,
            kde_params TEXT,
            default_value TEXT,
            lower_bound TEXT,
            upper_bound TEXT,
            sample_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_param_uniq ON param_distributions(tool_name, tool_version, param_name);

        CREATE TABLE IF NOT EXISTS discovered_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dag_definition TEXT NOT NULL,
            param_template TEXT,
            frequency REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'canary' CHECK(status IN ('canary','promoted')),
            discovered_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS deployed_skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            discovery_id INTEGER REFERENCES discovered_skills(id),
            name TEXT NOT NULL UNIQUE,
            dag_definition TEXT NOT NULL,
            param_template TEXT,
            credit_score REAL NOT NULL DEFAULT 50.0,
            success_count INTEGER NOT NULL DEFAULT 0,
            total_calls INTEGER NOT NULL DEFAULT 0,
            total_latency_ms INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'canary_5'
                CHECK(status IN ('canary_5','canary_15','canary_50','active','deprecated','offline')),
            last_used_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS canary_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill_id INTEGER NOT NULL REFERENCES deployed_skills(id),
            variant TEXT NOT NULL CHECK(variant IN ('stable','canary')),
            success INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            tokens INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_canary_skill ON canary_invocations(skill_id, variant);

        CREATE TABLE IF NOT EXISTS memory_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS entity_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL DEFAULT 'co_occur',
            strength INTEGER NOT NULL DEFAULT 1,
            evidence_trace_ids TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_er_source ON entity_relations(source_entity);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_er_pair ON entity_relations(source_entity, target_entity, relation_type);

        CREATE TABLE IF NOT EXISTS repair_hints (
            rule_id INTEGER PRIMARY KEY REFERENCES rules(id) ON DELETE CASCADE,
            content_hash TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            fix TEXT,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS execution_tasks (
            task_id TEXT PRIMARY KEY,
            task_description TEXT NOT NULL,
            skill_name TEXT,
            mode TEXT NOT NULL CHECK(mode IN ('skill_plan','llm_plan')),
            plan TEXT,
            status TEXT NOT NULL CHECK(status IN ('pending','running','success','failed','cancelled')),
            summary TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            finished_at TEXT
        );

        CREATE TABLE IF NOT EXISTS execution_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL REFERENCES execution_tasks(task_id),
            step_index INTEGER NOT NULL,
            tool_name TEXT NOT NULL,
            params TEXT,
            result TEXT,
            status TEXT NOT NULL,
            latency_ms INTEGER,
            tokens INTEGER NOT NULL DEFAULT 0,
            rules_triggered TEXT,
            repair_hint_applied TEXT,
            adapter TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_es_task ON execution_steps(task_id);

        CREATE TABLE IF NOT EXISTS schema_meta (
            version INTEGER NOT NULL
        );
    """)
    await conn.commit()
