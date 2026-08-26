import asyncio
import json

import httpx

from tool_evolution.server.app import app
from tool_evolution.server.routes import execute as execute_module


async def _seed_active_skill(conn):
    cursor = await conn.execute(
        """INSERT INTO deployed_skills (name, dag_definition, param_template,
           credit_score, status)
           VALUES ('search_api → detail_api', ?, '{}', 50.0, 'active')""",
        (json.dumps({"nodes": [{"tool_name": "search_api"},
                               {"tool_name": "detail_api"}],
                     "edges": [{"from": "search_api", "to": "detail_api"}]}),),
    )
    await conn.commit()
    return cursor.lastrowid


class TestExecuteTask:
    async def test_401_without_key(self, setup_db, monkeypatch):
        from tool_evolution.utils.config import settings
        monkeypatch.setattr(settings, "api_key", "test-key")
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test") as ac:
            resp = await ac.post("/api/execute/task",
                                 json={"task_id": "t1",
                                       "task_description": "d"})
            assert resp.status_code == 401

    async def test_auto_hit_skill_plan(self, setup_db, client):
        await _seed_active_skill(setup_db)
        resp = await client.post("/api/execute/task", json={
            "task_id": "t1",
            "task_description": "调用 search_api 和 detail_api 完成检索",
            "params": {"query": "hello"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode_used"] == "skill_plan"
        assert data["status"] == "success"
        assert data["matched_score"] == 1.0
        assert len(data["steps"]) == 2
        # R2 闭环断言（I#1 修复）：真实执行后技能统计必须落值
        cursor = await setup_db.execute(
            "SELECT total_calls, success_count FROM deployed_skills WHERE name=?"
            , ("search_api → detail_api",))
        row = await cursor.fetchone()
        assert row["total_calls"] == 1
        assert row["success_count"] == 1

    async def test_auto_fallback_llm_plan(self, setup_db, client):
        resp = await client.post("/api/execute/task", json={
            "task_id": "t2",
            "task_description": "写一份季度报告",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode_used"] == "llm_plan"
        # 无 DeepSeek key（degraded）：LLM 规划不可用 → 任务 failed（对照组诚实）
        assert data["status"] == "failed"

    async def test_skill_plan_no_match_404(self, setup_db, client):
        resp = await client.post("/api/execute/task", json={
            "task_id": "t3", "task_description": "写一份季度报告",
            "mode": "skill_plan",
        })
        assert resp.status_code == 404

    async def test_repeat_submit_returns_stored(self, setup_db, client):
        await _seed_active_skill(setup_db)
        payload = {
            "task_id": "t4",
            "task_description": "调用 search_api 和 detail_api 完成检索",
            "params": {"query": "hello"},
        }
        r1 = await client.post("/api/execute/task", json=payload)
        assert r1.status_code == 200
        r2 = await client.post("/api/execute/task", json=payload)
        assert r2.status_code == 200
        assert r2.json()["task_id"] == "t4"
        cursor = await setup_db.execute(
            "SELECT COUNT(*) FROM execution_tasks WHERE task_id='t4'")
        assert (await cursor.fetchone())[0] == 1

    async def test_concurrent_same_key_409(self, setup_db, client,
                                           monkeypatch):
        await _seed_active_skill(setup_db)
        from tool_evolution.execution.adapters import MockAdapter
        monkeypatch.setattr(execute_module, "_make_adapter",
                            lambda req: MockAdapter(delay_s=0.3))
        payload = {
            "task_id": "t5",
            "task_description": "调用 search_api 和 detail_api 完成检索",
            "params": {"query": "hello"},
        }
        r1, r2 = await asyncio.gather(
            client.post("/api/execute/task", json=payload),
            client.post("/api/execute/task", json=payload),
        )
        statuses = sorted([r1.status_code, r2.status_code])
        assert statuses == [200, 409]
        conflict = r1 if r1.status_code == 409 else r2
        assert "poll" in conflict.json()["detail"]["hint"]

    async def test_get_task_with_steps(self, setup_db, client):
        await _seed_active_skill(setup_db)
        await client.post("/api/execute/task", json={
            "task_id": "t6",
            "task_description": "调用 search_api 和 detail_api 完成检索",
            "params": {"query": "hello"},
        })
        resp = await client.get("/api/execute/task/t6")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "t6"
        assert data["status"] == "success"
        assert len(data["steps"]) == 2

    async def test_get_missing_404(self, setup_db, client):
        resp = await client.get("/api/execute/task/nope")
        assert resp.status_code == 404

    async def test_invalid_mode_422(self, setup_db, client):
        resp = await client.post("/api/execute/task", json={
            "task_id": "t7", "task_description": "d", "mode": "bogus",
        })
        assert resp.status_code == 422

    async def test_adapter_param_validation(self, setup_db, client):
        # 非法 adapter 值
        resp = await client.post("/api/execute/task", json={
            "task_id": "t8", "task_description": "d", "adapter": "bogus",
        })
        assert resp.status_code == 422
        # http 缺 base_url
        resp = await client.post("/api/execute/task", json={
            "task_id": "t9", "task_description": "d", "adapter": "http",
        })
        assert resp.status_code == 422
        # mcp 缺 command
        resp = await client.post("/api/execute/task", json={
            "task_id": "t10", "task_description": "d", "adapter": "mcp",
        })
        assert resp.status_code == 422
