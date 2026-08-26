from tool_evolution.execution.audit import ExecutionAudit


async def _create_task(audit, task_id="task-1", mode="skill_plan"):
    await audit.create_task(
        task_id=task_id, task_description="desc", mode=mode,
        plan={"nodes": []}, skill_name="s1",
    )


class TestExecutionAudit:
    async def test_create_task_idempotent(self, db_conn):
        audit = ExecutionAudit(db_conn)
        assert await audit.create_task(
            task_id="t1", task_description="d", mode="skill_plan", plan={}
        ) is True
        assert await audit.create_task(
            task_id="t1", task_description="d", mode="skill_plan", plan={}
        ) is False

    async def test_update_status_transitions(self, db_conn):
        audit = ExecutionAudit(db_conn)
        await _create_task(audit)
        await audit.update_status("task-1", "running")
        task = await audit.get_task("task-1")
        assert task["status"] == "running"
        assert task["finished_at"] is None
        await audit.update_status("task-1", "success", summary="done")
        task = await audit.get_task("task-1")
        assert task["status"] == "success"
        assert task["summary"] == "done"
        assert task["finished_at"] is not None

    async def test_get_task_merges_steps(self, db_conn):
        audit = ExecutionAudit(db_conn)
        await _create_task(audit)
        await audit.add_step(
            task_id="task-1", step_index=0, tool_name="search_api",
            params={"q": 1}, result={"ok": True}, status="success",
            latency_ms=10, tokens=5, rules_triggered=None,
            repair_hint_applied=None, adapter="mock",
        )
        task = await audit.get_task("task-1")
        assert task["task_id"] == "task-1"
        assert len(task["steps"]) == 1
        assert task["steps"][0]["tool_name"] == "search_api"
        assert task["steps"][0]["status"] == "success"

    async def test_get_missing_task_none(self, db_conn):
        audit = ExecutionAudit(db_conn)
        assert await audit.get_task("nope") is None
