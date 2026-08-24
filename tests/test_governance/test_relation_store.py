class TestEntityRelationsTable:
    async def test_table_exists(self, db_conn):
        cursor = await db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entity_relations'"
        )
        row = await cursor.fetchone()
        assert row is not None

    async def test_unique_index_includes_relation_type(self, db_conn):
        cursor = await db_conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_er_pair'"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert "relation_type" in row["sql"]
        assert "UNIQUE" in row["sql"].upper()

    async def test_columns(self, db_conn):
        cursor = await db_conn.execute("PRAGMA table_info(entity_relations)")
        cols = {row["name"] for row in await cursor.fetchall()}
        assert {"id", "source_entity", "target_entity", "relation_type",
                "strength", "evidence_trace_ids", "created_at"} <= cols
