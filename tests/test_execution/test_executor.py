import json
import time

from tool_evolution.collection.store import TraceStore
from tool_evolution.execution.adapters import MockAdapter
from tool_evolution.execution.audit import ExecutionAudit
from tool_evolution.execution.executor import SkillExecutor
from tool_evolution.knowledge.rule_engine import RuleEngine


def _plan(nodes=None, edges=None, skill_id=1, skill_name="test_skill",
          blocked=False, precheck_rules=None):
    nodes = nodes or [
        {"tool_name": "search_api", "params": {"query": "q"}},
        {"tool_name": "detail_api", "params": {"query": "q"}},
    ]
    edges = edges if edges is not None else [{"from": 0, "to": 1}]
    return {
        "skill_id": skill_id, "skill_name": skill_name,
        "nodes": nodes, "edges": edges,
        "precheck_rules": precheck_rules or [],
        "blocked": blocked,
        "block_reason": "前置拦截触发" if blocked else None,
    }


async def _mk_executor(db_conn, delay_s=0.0):
    audit = ExecutionAudit(db_conn)
    await audit.create_task(
        task_id="task-1", task_description="desc", mode="skill_plan",
        plan={}, skill_name="test_skill",
    )
    adapter = MockAdapter(delay_s=delay_s)
    executor = SkillExecutor(db_conn, adapter, audit=audit)
    return executor, adapter


async def _trace_rows(db_conn):
    return await TraceStore(db_conn).get_all_traces(limit=100)


class TestExecutePlan:
    async def test_sequential_chain_success(self, db_conn):
        executor, adapter = await _mk_executor(db_conn)
        result = await executor.execute_plan("task-1", "desc", _plan())
        assert result["status"] == "success"
        rows = await _trace_rows(db_conn)
        root = [r for r in rows if r["trace_type"] == "task_root"]
        atomics = [r for r in rows if r["trace_type"] == "atomic"]
        assert len(root) == 1 and len(atomics) == 2
        assert all(r["parent_trace_id"] == root[0]["trace_id"] for r in atomics)
        await adapter.close()

    async def test_parallel_branches_faster_than_serial(self, db_conn):
        nodes = [
            {"tool_name": "search_api", "params": {"query": "q"}},
            {"tool_name": "detail_api", "params": {"query": "q"}},
            {"tool_name": "analyze_api", "params": {"query": "q"}},
        ]
        edges = [{"from": 0, "to": 2}]
        executor, adapter = await _mk_executor(db_conn, delay_s=0.3)
        start = time.monotonic()
        result = await executor.execute_plan("task-1", "desc", _plan(nodes=nodes, edges=edges))
        elapsed = time.monotonic() - start
        assert result["status"] == "success"
        # 3 节点两层（层1并行 0/1，层2 单节点 2）——0.3s 每节点，串行需 0.9s
        assert elapsed < 0.85
        await adapter.close()

    async def test_cycle_detected_failed(self, db_conn):
        executor, adapter = await _mk_executor(db_conn)
        edges = [{"from": 0, "to": 1}, {"from": 1, "to": 0}]
        result = await executor.execute_plan("task-1", "desc", _plan(edges=edges))
        assert result["status"] == "failed"
        assert "环" in (result["summary"] or "")
        await adapter.close()

    async def test_timeout_rule_triggers(self, db_conn):
        await RuleEngine(db_conn).add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "timeout_rule",
            "condition": {"on_error": "timeout"},
            "action": {"max_wait_ms": 100},
            "status": "active",
        })
        executor, adapter = await _mk_executor(db_conn, delay_s=0.5)
        result = await executor.execute_plan("task-1", "desc", _plan())
        assert result["status"] == "failed"
        assert "timeout_rule" in (result.get("rules_triggered") or [])
        await adapter.close()

    async def test_retry_rule_rescues(self, db_conn):
        await RuleEngine(db_conn).add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "retry_rule",
            "condition": {"on_error": "quota_exhausted"},
            "action": {"delay_seconds": 0, "max_retries": 2},
            "status": "active",
        })
        executor, adapter = await _mk_executor(db_conn)
        failures = {"count": 0}

        async def flaky_execute(tool_name, params):
            if tool_name == "search_api" and failures["count"] == 0:
                failures["count"] += 1
                from tool_evolution.collection.schemas import ErrorType
                from tool_evolution.execution.adapters import ToolResult
                return ToolResult(
                    tool_name=tool_name, params=params, success=False,
                    error_type=ErrorType.QUOTA_EXHAUSTED,
                    error_message="rate limit exceeded", latency_ms=0, token_count=0,
                )
            return await MockAdapter.execute(adapter, tool_name, params)

        adapter.execute = flaky_execute
        result = await executor.execute_plan("task-1", "desc", _plan())
        assert result["status"] == "success"
        assert "retry_rule" in (result.get("rules_triggered") or [])
        await adapter.close()

    async def test_circuit_breaker_stops_successors(self, db_conn):
        await RuleEngine(db_conn).add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "circuit_breaker_rule",
            "condition": {"on_error": "service_unavailable"},
            "action": {"failure_threshold": 1, "cooldown_seconds": 30},
            "status": "active",
        })
        executor, adapter = await _mk_executor(db_conn)
        # search_api 强制失败 → 熔断阈值 1 → 任务 failed，后继 detail_api 跳过
        async def fail_search(tool_name, params):
            if tool_name == "search_api":
                from tool_evolution.collection.schemas import ErrorType
                from tool_evolution.execution.adapters import ToolResult
                return ToolResult(
                    tool_name=tool_name, params=params, success=False,
                    error_type=ErrorType.SERVICE_UNAVAILABLE,
                    error_message="503", latency_ms=0, token_count=0,
                )
            return await MockAdapter.execute(adapter, tool_name, params)

        adapter.execute = fail_search
        result = await executor.execute_plan("task-1", "desc", _plan())
        assert result["status"] == "failed"
        steps = [s for s in result["steps"] if s["status"] == "skipped"]
        assert steps, "后继节点应被跳过"
        await adapter.close()

    async def test_repair_hint_applied_and_rescues(self, db_conn):
        engine = RuleEngine(db_conn)
        rule_id = await engine.add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {"param_names": ["max_results"]},
            "action": {"validate_before_call": True},
            "status": "active",
        })
        await db_conn.execute(
            """INSERT INTO repair_hints (rule_id, content_hash, suggestion, fix, model)
               VALUES (?, 'h1', '减小 max_results', ?, 'deepseek-v4-flash')""",
            (rule_id, json.dumps({"param": "max_results", "suggested_value": 5})),
        )
        await db_conn.commit()
        executor, adapter = await _mk_executor(db_conn)
        # search_api 首次调用失败（修复建议注入 max_results=5 后成功）
        calls = {"n": 0}

        async def invalid_then_ok(tool_name, params):
            if tool_name == "search_api" and calls["n"] == 0:
                calls["n"] += 1
                from tool_evolution.collection.schemas import ErrorType
                from tool_evolution.execution.adapters import ToolResult
                return ToolResult(
                    tool_name=tool_name, params=params, success=False,
                    error_type=ErrorType.PARAM_ERROR,
                    error_message="value -1 out of range", latency_ms=0, token_count=0,
                )
            return await MockAdapter.execute(adapter, tool_name, params)

        adapter.execute = invalid_then_ok
        plan = _plan(nodes=[
            {"tool_name": "search_api", "params": {"query": "q", "max_results": -1}},
            {"tool_name": "detail_api", "params": {"query": "q"}},
        ])
        result = await executor.execute_plan("task-1", "desc", plan)
        assert result["status"] == "success"
        applied = result.get("repair_hint_applied") or []
        assert any(a.get("rule_id") == rule_id for a in applied)
        # 修复证据持久化：审计 steps 落库值含 rule_id+fix
        task = await executor.audit.get_task("task-1")
        persisted = [s["repair_hint_applied"] for s in task["steps"]
                     if s["repair_hint_applied"] is not None]
        assert persisted
        assert any(p["rule_id"] == rule_id and p["fix"]["param"] == "max_results"
                   for p in persisted)
        await adapter.close()

    async def test_repair_fix_injected_when_param_missing(self, db_conn):
        engine = RuleEngine(db_conn)
        rule_id = await engine.add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {"param_names": ["max_results"]},
            "action": {"validate_before_call": True},
            "status": "active",
        })
        await db_conn.execute(
            """INSERT INTO repair_hints (rule_id, content_hash, suggestion, fix, model)
               VALUES (?, 'h1', '补 max_results', ?, 'deepseek-v4-flash')""",
            (rule_id, json.dumps({"param": "max_results", "suggested_value": 5})),
        )
        await db_conn.commit()
        executor, adapter = await _mk_executor(db_conn)
        calls = {"n": 0}

        async def fail_first(tool_name, params):
            if tool_name == "search_api" and calls["n"] == 0:
                calls["n"] += 1
                from tool_evolution.collection.schemas import ErrorType
                from tool_evolution.execution.adapters import ToolResult
                return ToolResult(
                    tool_name=tool_name, params=params, success=False,
                    error_type=ErrorType.PARAM_ERROR,
                    error_message="missing required parameter", latency_ms=0, token_count=0,
                )
            return await MockAdapter.execute(adapter, tool_name, params)

        adapter.execute = fail_first
        # 节点 params 无 max_results → fix 注入新键后重试
        plan = _plan(nodes=[
            {"tool_name": "search_api", "params": {"query": "q"}},
            {"tool_name": "detail_api", "params": {"query": "q"}},
        ])
        result = await executor.execute_plan("task-1", "desc", plan)
        assert result["status"] == "success"
        await adapter.close()

    async def test_empty_fix_not_retried(self, db_conn):
        engine = RuleEngine(db_conn)
        await engine.add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {"param_names": ["max_results"]},
            "action": {"validate_before_call": True},
            "status": "active",
        })
        executor, adapter = await _mk_executor(db_conn)
        calls = {"n": 0}

        async def always_fail(tool_name, params):
            calls["n"] += 1
            from tool_evolution.collection.schemas import ErrorType
            from tool_evolution.execution.adapters import ToolResult
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.PARAM_ERROR,
                error_message="bad", latency_ms=0, token_count=0,
            )

        adapter.execute = always_fail
        result = await executor.execute_plan("task-1", "desc", _plan())
        assert result["status"] == "failed"
        # 2 节点各执行一次即失败——无可用修复建议时不得有任何重试
        assert calls["n"] == 2
        await adapter.close()

    async def test_blocked_plan_fails_without_execution(self, db_conn):
        executor, adapter = await _mk_executor(db_conn)
        result = await executor.execute_plan(
            "task-1", "desc", _plan(blocked=True, precheck_rules=[{"rule_id": 1}]))
        assert result["status"] == "failed"
        rows = await _trace_rows(db_conn)
        assert len(rows) == 0, "blocked 计划不得产生任何执行轨迹"
        await adapter.close()

    async def test_cross_task_circuit_opens_and_rejects(self, db_conn):
        await RuleEngine(db_conn).add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "circuit_breaker_rule",
            "condition": {"on_error": "service_unavailable"},
            "action": {"failure_threshold": 2, "cooldown_seconds": 60},
            "status": "active",
        })
        executor, adapter = await _mk_executor(db_conn)
        calls = {"n": 0}

        async def fail_tool(tool_name, params):
            calls["n"] += 1
            from tool_evolution.collection.schemas import ErrorType
            from tool_evolution.execution.adapters import ToolResult
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                error_message="503", latency_ms=0, token_count=0,
            )

        adapter.execute = fail_tool
        single = _plan(nodes=[{"tool_name": "search_api",
                                "params": {"query": "q"}}], edges=[])
        # 任务 1：失败 1 次（count=1）；任务 2：失败第 2 次 → 开闸（count=2）
        await executor.execute_plan("task-1", "desc", single)
        await executor.execute_plan("task-1", "desc", single)
        # 任务 3：熔断中 → 直接拒绝，不再调 adapter
        result = await executor.execute_plan("task-1", "desc", single)
        assert result["status"] == "failed"
        assert "circuit" in (result["summary"] or "").lower()
        assert calls["n"] == 2, "熔断后不得再调用 adapter"
        await adapter.close()

    async def test_circuit_resets_on_success(self, db_conn):
        await RuleEngine(db_conn).add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "circuit_breaker_rule",
            "condition": {"on_error": "service_unavailable"},
            "action": {"failure_threshold": 1, "cooldown_seconds": 60},
            "status": "active",
        })
        executor, adapter = await _mk_executor(db_conn)
        single = _plan(nodes=[{"tool_name": "search_api",
                                "params": {"query": "q"}}], edges=[])
        async def fail_once(tool_name, params):
            from tool_evolution.collection.schemas import ErrorType
            from tool_evolution.execution.adapters import ToolResult
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.SERVICE_UNAVAILABLE,
                error_message="503", latency_ms=0, token_count=0,
            )
        adapter.execute = fail_once
        await executor.execute_plan("task-1", "desc", single)  # 开闸
        # 恢复后成功 → closed 重置
        adapter.execute = lambda t, p: MockAdapter.execute(adapter, t, p)
        result = await executor.execute_plan("task-1", "desc", single)
        assert result["status"] == "success"
        cursor = await db_conn.execute(
            "SELECT status, failure_count FROM circuit_states WHERE tool_name='search_api'")
        row = await cursor.fetchone()
        assert row["status"] == "closed"
        assert row["failure_count"] == 0
        await adapter.close()

    async def test_failed_task_has_root_trace(self, db_conn):
        executor, adapter = await _mk_executor(db_conn)

        async def always_fail(tool_name, params):
            from tool_evolution.collection.schemas import ErrorType
            from tool_evolution.execution.adapters import ToolResult
            return ToolResult(
                tool_name=tool_name, params=params, success=False,
                error_type=ErrorType.PARAM_ERROR, error_message="bad",
                latency_ms=0, token_count=0,
            )

        adapter.execute = always_fail
        result = await executor.execute_plan("task-1", "desc", _plan())
        assert result["status"] == "failed"
        rows = await _trace_rows(db_conn)
        roots = [r for r in rows if r["trace_type"] == "task_root"]
        atomics = [r for r in rows if r["trace_type"] == "atomic"]
        # I#4 修复：失败任务也有 root 轨迹，任务树可查询（无 dangling parent）
        assert len(roots) == 1
        assert roots[0]["success"] == 0
        assert all(r["parent_trace_id"] == roots[0]["trace_id"] for r in atomics)
        await adapter.close()

    async def test_executor_trace_prefixes(self, db_conn):
        executor, adapter = await _mk_executor(db_conn)
        await executor.execute_plan("task-1", "desc", _plan())
        rows = await _trace_rows(db_conn)
        assert all(r["trace_id"].startswith("exec-") for r in rows)
        assert all(r["agent_id"].startswith("executor:") for r in rows)
        assert all(r["source"] == "executor" for r in rows)
        await adapter.close()


class TestExecuteSkill:
    async def test_record_call_updated(self, db_conn):
        await db_conn.execute(
            """INSERT INTO deployed_skills (id, name, dag_definition, param_template,
               credit_score, status) VALUES (1, 'test_skill', '{}', '{}', 50.0, 'active')"""
        )
        await db_conn.commit()
        audit = ExecutionAudit(db_conn)
        await audit.create_task(
            task_id="task-1", task_description="desc", mode="skill_plan",
            plan={}, skill_name="test_skill",
        )
        executor = SkillExecutor(db_conn, MockAdapter(), audit=audit)
        try:
            result = await executor.execute_skill(
                "task-1", "desc",
                skill={"id": 1, "name": "test_skill",
                       "dag_definition": json.dumps({
                           "nodes": [{"tool_name": "search_api"}],
                           "edges": []}),
                       "param_template": None},
                task_params={"query": "q"}, agent_id="anon")
            assert result["status"] == "success"
            cursor = await db_conn.execute(
                "SELECT total_calls, success_count FROM deployed_skills WHERE id=1")
            row = await cursor.fetchone()
            assert row["total_calls"] == 1
            assert row["success_count"] == 1
        finally:
            await executor.adapter.close()
