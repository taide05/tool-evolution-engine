import json

import pytest

from tool_evolution.governance.relation_store import RelationStore, extract_entities
from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport, TraceType


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


async def _seed_task(db_conn, root_id: str, results: list[dict]):
    ts = TraceStore(db_conn)
    await ts.insert(TraceReport(trace_id=root_id, agent_id="seed", tool_name="task",
                                trace_type=TraceType.TASK_ROOT, success=True, latency_ms=0))
    for i, result in enumerate(results):
        await ts.insert(TraceReport(
            trace_id=f"{root_id}-c{i}", parent_trace_id=root_id, agent_id="seed",
            tool_name="tool_a", success=True, latency_ms=5, result=result,
        ))


class TestBuildForTask:
    async def test_builds_pairs_across_task_traces(self, store, db_conn):
        await _seed_task(db_conn, "root1", [{"entity": "Alpha"}, {"subject": "Beta"}])
        count = await store.build_for_task("root1")
        assert count == 1
        rows = await store.search_relations("Alpha")
        assert len(rows) == 1
        # 字典序：Alpha < Beta，方向无关断言
        assert {rows[0]["source_entity"], rows[0]["target_entity"]} == {"Alpha", "Beta"}

    async def test_multi_entity_trace_forms_all_pairs(self, store, db_conn):
        await _seed_task(db_conn, "root2", [{"entities": ["A", "B", "C"]}])
        count = await store.build_for_task("root2")
        assert count == 3  # (A,B) (A,C) (B,C)

    async def test_rebuild_same_task_is_idempotent(self, store, db_conn):
        await _seed_task(db_conn, "root3", [{"entity": "X"}, {"title": "Y"}])
        await store.build_for_task("root3")
        await store.build_for_task("root3")
        row = (await store.search_relations("X"))[0]
        # strength 语义 = 累计不同贡献轨迹数（用户裁决 A）：2 条轨迹 → 2，重建不叠加
        assert row["strength"] == 2

    async def test_two_tasks_accumulate_strength(self, store, db_conn):
        await _seed_task(db_conn, "root4", [{"entity": "X"}, {"title": "Y"}])
        await _seed_task(db_conn, "root5", [{"entity": "X"}, {"title": "Y"}])
        await store.build_for_task("root4")
        await store.build_for_task("root5")
        row = (await store.search_relations("X"))[0]
        # 两个任务各 2 条轨迹 → 4
        assert row["strength"] == 4

    async def test_ignores_failed_traces(self, store, db_conn):
        ts = TraceStore(db_conn)
        await ts.insert(TraceReport(trace_id="root6", agent_id="seed", tool_name="task",
                                    trace_type=TraceType.TASK_ROOT, success=True, latency_ms=0))
        await ts.insert(TraceReport(trace_id="root6-c0", parent_trace_id="root6",
                                    agent_id="seed", tool_name="tool_a", success=True,
                                    latency_ms=5, result={"entity": "X"}))
        await ts.insert(TraceReport(trace_id="root6-c1", parent_trace_id="root6",
                                    agent_id="seed", tool_name="tool_a", success=False,
                                    latency_ms=5, result={"title": "Y"}))
        count = await store.build_for_task("root6")
        assert count == 0
