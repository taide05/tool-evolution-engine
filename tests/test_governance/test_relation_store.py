import json

import pytest

from tool_evolution.governance.relation_store import RelationStore, extract_entities


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


@pytest.fixture
async def store(db_conn):
    return RelationStore(db_conn)


class TestExtractEntities:
    def test_extracts_known_fields(self):
        result = {"entity": "A", "entities": ["B", "C"], "law_name": "D",
                  "title": "E", "subject": "F", "other": "ignored"}
        assert set(extract_entities(result)) == {"A", "B", "C", "D", "E", "F"}

    def test_skips_empty_and_non_string(self):
        result = {"entity": "", "entities": [1, None, "ok"], "count": 3}
        assert extract_entities(result) == ["ok"]


class TestRelationStoreUpsert:
    async def test_upsert_orders_lexicographically(self, store):
        await store.upsert_cooccurrence("B", "A", ["t1"])
        rows = await store.search_relations("A")
        assert len(rows) == 1
        assert rows[0]["source_entity"] == "A"
        assert rows[0]["target_entity"] == "B"
        assert rows[0]["relation_type"] == "co_occur"
        assert rows[0]["strength"] == 1

    async def test_upsert_idempotent_same_trace(self, store):
        await store.upsert_cooccurrence("A", "B", ["t1"])
        await store.upsert_cooccurrence("A", "B", ["t1"])
        rows = await store.search_relations("A")
        assert rows[0]["strength"] == 1

    async def test_upsert_new_trace_increments(self, store):
        await store.upsert_cooccurrence("A", "B", ["t1"])
        await store.upsert_cooccurrence("A", "B", ["t2"])
        rows = await store.search_relations("A")
        assert rows[0]["strength"] == 2

    async def test_evidence_capped_at_20(self, store):
        ids = [f"t{i}" for i in range(25)]
        await store.upsert_cooccurrence("A", "B", ids)
        row = (await store.search_relations("A"))[0]
        assert row["strength"] == 25
        assert len(json.loads(row["evidence_trace_ids"])) == 20

    async def test_duplicate_trace_ids_deduped(self, store):
        await store.upsert_cooccurrence("A", "B", ["t1", "t1"])
        rows = await store.search_relations("A")
        assert rows[0]["strength"] == 1
