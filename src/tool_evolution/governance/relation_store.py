import json
import aiosqlite

ENTITY_FIELDS = ("entity", "entities", "doc_name", "title", "subject")


def extract_entities(result: dict) -> list[str]:
    entities: list[str] = []
    for field in ENTITY_FIELDS:
        val = result.get(field)
        if isinstance(val, str) and val:
            entities.append(val)
        elif isinstance(val, list):
            entities.extend(str(v) for v in val if isinstance(v, str))
    return entities


class RelationStore:
    """Co-occurrence relation storage on the entity_relations table."""

    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def _get_pair(self, source: str, target: str, relation_type: str) -> dict | None:
        cursor = await self.conn.execute(
            """SELECT * FROM entity_relations
               WHERE source_entity=? AND target_entity=? AND relation_type=?""",
            (source, target, relation_type)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def upsert_cooccurrence(self, entity_a: str, entity_b: str,
                                  trace_ids: list[str]) -> None:
        source, target = sorted((entity_a, entity_b))
        existing = await self._get_pair(source, target, "co_occur")
        old_evidence: list[str] = []
        if existing:
            old_evidence = json.loads(existing["evidence_trace_ids"] or "[]")
        # dict.fromkeys 去重（同一 trace 可能对同一实体对贡献多次）
        new_ids = list(dict.fromkeys(t for t in trace_ids if t not in old_evidence))
        if not new_ids:
            return
        # 全量存证据：去重基于完整 id 集，重建幂等精确（无截断重计数边界）
        evidence = old_evidence + new_ids
        if existing:
            await self.conn.execute(
                """UPDATE entity_relations
                   SET strength = strength + ?, evidence_trace_ids = ?
                   WHERE source_entity=? AND target_entity=? AND relation_type=?""",
                (len(new_ids), json.dumps(evidence), source, target, "co_occur")
            )
        else:
            await self.conn.execute(
                """INSERT INTO entity_relations
                   (source_entity, target_entity, relation_type, strength, evidence_trace_ids)
                   VALUES (?, ?, 'co_occur', ?, ?)""",
                (source, target, len(new_ids), json.dumps(evidence))
            )
        await self.conn.commit()

    async def build_for_task(self, root_id: str) -> int:
        """Build co-occurrence pairs from all successful atomic traces of a task tree.

        跨 trace 实体池化：收集任务内全部成功 atomic trace 的实体为一个池，
        池内所有实体两两成对（同 trace 或跨 trace），evidence 为贡献实体的 trace id。
        """
        from ..collection.store import TraceStore
        traces = await TraceStore(self.conn).get_task_tree(root_id)
        entity_locations: list[tuple[str, str]] = []  # (trace_id, entity)
        for t in traces:
            if t["trace_type"] != "atomic" or not t["success"] or not t["result"]:
                continue
            for entity in extract_entities(json.loads(t["result"])):
                entity_locations.append((t["trace_id"], entity))
        pair_map: dict[tuple[str, str], list[str]] = {}
        for i in range(len(entity_locations)):
            for j in range(i + 1, len(entity_locations)):
                (ti, a), (tj, b) = entity_locations[i], entity_locations[j]
                if a == b:
                    continue
                pair = tuple(sorted((a, b)))
                pair_map.setdefault(pair, []).extend([ti, tj])
        for (a, b), trace_ids in pair_map.items():
            await self.upsert_cooccurrence(a, b, trace_ids)
        return len(pair_map)

    async def search_relations(self, entity: str) -> list[dict]:
        cursor = await self.conn.execute(
            """SELECT source_entity, target_entity, relation_type, strength, evidence_trace_ids
               FROM entity_relations WHERE source_entity=? OR target_entity=?
               ORDER BY strength DESC""",
            (entity, entity)
        )
        return [dict(row) for row in await cursor.fetchall()]
