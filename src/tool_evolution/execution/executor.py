"""执行器——DAG 拓扑调度 + 运行时规则 + 修复建议重试 + 闭环采集 + 审计。"""

import asyncio
import time
import uuid

import aiosqlite
import networkx as nx

from ..collection.schemas import ErrorType, TraceReport, TraceType
from ..collection.store import TraceStore
from ..governance.governor import SkillGovernor
from ..governance.mcp_bridge import MCPBridge
from ..knowledge.rule_engine import RuleEngine
from .adapters import AsyncToolAdapter, ToolResult
from .assembler import PlanAssembler
from .audit import ExecutionAudit


class SkillExecutor:
    def __init__(self, conn: aiosqlite.Connection, adapter: AsyncToolAdapter,
                 audit: ExecutionAudit | None = None):
        self.conn = conn
        self.adapter = adapter
        self.audit = audit or ExecutionAudit(conn)
        # aiosqlite 单连接：并行节点的写库操作必须串行（BEGIN IMMEDIATE 不可嵌套）
        self._db_lock = asyncio.Lock()

    async def execute_plan(self, task_id: str, task_description: str,
                           plan: dict, agent_id: str = "anonymous") -> dict:
        """执行装配好的计划（前置条件：task 已 create——状态归 API 层管理）。"""
        start = time.monotonic()
        executor_agent = f"executor:{agent_id}"
        rules_triggered: list[str] = []
        repair_applied: list[dict] = []
        steps: list[dict] = []
        node_results: list[ToolResult | None] = []
        failure_count = 0
        circuit_limit: int | None = None
        trace_store = TraceStore(self.conn)

        if plan.get("blocked"):
            summary = f"前置拦截：{plan.get('block_reason')}"
            return self._result(
                "failed", steps, summary, rules_triggered, repair_applied,
                time.monotonic() - start)

        root_trace_id = f"exec-{uuid.uuid4().hex[:16]}"

        nodes = plan.get("nodes", [])
        edges = plan.get("edges", [])
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from((e["from"], e["to"]) for e in edges)
        if not nx.is_directed_acyclic_graph(graph):
            return self._result(
                "failed", steps, "DAG 存在环，无法拓扑排序",
                rules_triggered, repair_applied, time.monotonic() - start)

        layers = _topo_layers(graph)
        step_index = 0
        stopped = False
        for layer in layers:
            if stopped:
                # 熔断停后继：剩余层全部记 skipped（不留无记录节点）
                for idx in layer:
                    tool = nodes[idx]["tool_name"]
                    params = dict(nodes[idx].get("params", {}))
                    node_result = {
                        "tool_name": tool, "params": params, "result": None,
                        "status": "skipped", "latency_ms": 0, "tokens": 0,
                        "rules_triggered": [], "repair_hint_applied": None,
                    }
                    node_results.append(node_result)
                    await self.audit.add_step(
                        task_id=task_id, step_index=step_index, tool_name=tool,
                        params=params, result=None, status="skipped",
                        latency_ms=0, tokens=0, rules_triggered=None,
                        repair_hint_applied=None,
                        adapter=type(self.adapter).__name__,
                    )
                    step_index += 1
                continue

            async def _run(idx):
                tool = nodes[idx]["tool_name"]
                params = dict(nodes[idx].get("params", {}))
                return await self._run_node(
                    idx, tool, params, task_id, executor_agent,
                    trace_store, repair_applied, root_trace_id)

            results = await asyncio.gather(*(_run(i) for i in layer))
            for idx, node_result in zip(layer, results):
                if node_result.get("circuit_limit"):
                    circuit_limit = node_result["circuit_limit"]
                if node_result["status"] == "failed":
                    failure_count += 1
                    if circuit_limit is not None and failure_count >= circuit_limit:
                        stopped = True
                for rule in node_result["rules_triggered"]:
                    if rule not in rules_triggered:
                        rules_triggered.append(rule)
                node_results.append(node_result)
                await self.audit.add_step(
                    task_id=task_id, step_index=step_index,
                    tool_name=node_result["tool_name"],
                    params=node_result["params"], result=node_result["result"],
                    status=node_result["status"],
                    latency_ms=node_result["latency_ms"],
                    tokens=node_result["tokens"],
                    rules_triggered=node_result["rules_triggered"] or None,
                    repair_hint_applied=node_result["repair_hint_applied"],
                    adapter=type(self.adapter).__name__,
                )
                step_index += 1

        total_latency_ms = int((time.monotonic() - start) * 1000)
        total_tokens = sum(r["tokens"] for r in node_results)
        success = (not stopped
                   and len(node_results) == len(nodes)
                   and all(r["status"] == "success" for r in node_results))
        status = "success" if success else "failed"
        ok_nodes = sum(1 for r in node_results if r["status"] == "success")
        failed_nodes = [r["tool_name"] for r in node_results if r["status"] == "failed"]
        skipped_nodes = [r["tool_name"] for r in node_results if r["status"] == "skipped"]
        summary = (f"成功节点 {ok_nodes}/{len(nodes)}"
                   + (f"，失败 {failed_nodes}" if failed_nodes else "")
                   + (f"，跳过 {skipped_nodes}" if skipped_nodes else ""))

        if success:
            await trace_store.insert(TraceReport(
                trace_id=root_trace_id,
                agent_id=executor_agent,
                tool_name=plan.get("skill_name", "executor"),
                trace_type=TraceType.TASK_ROOT, success=True,
                params={}, latency_ms=total_latency_ms,
                token_count=total_tokens, source="executor",
            ))

        return self._result(status, node_results, summary, rules_triggered,
                            repair_applied, total_latency_ms, total_tokens)

    async def _run_node(self, idx: int, tool: str, params: dict, task_id: str,
                        executor_agent: str, trace_store: TraceStore,
                        repair_applied: list[dict],
                        parent_trace_id: str) -> dict:
        engine = RuleEngine(self.conn)
        runtime_rules = await engine.get_runtime_rules(tool, "1.0.0")
        timeout_ms = None
        retry_delay, max_retries = None, 0
        circuit_threshold = None
        for rule in runtime_rules:
            if rule["rule_type"] == "timeout_rule":
                timeout_ms = rule["action"].get("max_wait_ms")
            elif rule["rule_type"] == "retry_rule":
                retry_delay = rule["action"].get("delay_seconds", 0)
                max_retries = rule["action"].get("max_retries", 0)
            elif rule["rule_type"] == "circuit_breaker_rule":
                circuit_threshold = rule["action"].get("failure_threshold")

        attempt = 0
        repair_used = False
        node_start = time.monotonic()
        result: ToolResult | None = None
        node_rules: list[str] = []

        while True:
            try:
                if timeout_ms:
                    result = await asyncio.wait_for(
                        self.adapter.execute(tool, params),
                        timeout=timeout_ms / 1000)
                else:
                    result = await self.adapter.execute(tool, params)
            except asyncio.TimeoutError:
                result = ToolResult(
                    tool_name=tool, params=params, success=False,
                    error_type=ErrorType.TIMEOUT,
                    error_message=f"节点超时（>{timeout_ms}ms）",
                    latency_ms=timeout_ms, token_count=0)
                if "timeout_rule" not in node_rules:
                    node_rules.append("timeout_rule")

            if result.success:
                async with self._db_lock:
                    await trace_store.insert(TraceReport(
                        trace_id=f"exec-{uuid.uuid4().hex[:16]}",
                        parent_trace_id=parent_trace_id,
                        agent_id=executor_agent, tool_name=tool,
                        trace_type=TraceType.ATOMIC, success=True,
                        params=params, result=result.result,
                        latency_ms=result.latency_ms,
                        token_count=result.token_count, source="executor",
                    ))
                    bridge = MCPBridge(self.conn)
                    await bridge.extract_and_update({
                        "success": True, "result": result.result, "tool_name": tool,
                    })
                return {
                    "tool_name": tool, "params": params, "result": result.result,
                    "status": "success", "latency_ms": result.latency_ms,
                    "tokens": result.token_count,
                    "rules_triggered": node_rules,
                    "repair_hint_applied": (
                        repair_applied[-1] if repair_used else None),
                    "circuit_limit": circuit_threshold,
                }

            if retry_delay is not None and attempt < max_retries:
                attempt += 1
                if "retry_rule" not in node_rules:
                    node_rules.append("retry_rule")
                if retry_delay:
                    await asyncio.sleep(retry_delay)
                continue

            hint = await self._first_fix_hint(tool) if not repair_used else None
            if hint is not None:
                repair_used = True
                params = dict(params)
                params[hint["fix"]["param"]] = hint["fix"]["suggested_value"]
                repair_applied.append({"rule_id": hint["rule_id"], "fix": hint["fix"]})
                attempt += 1
                continue

            latency_ms = int((time.monotonic() - node_start) * 1000)
            async with self._db_lock:
                await trace_store.insert(TraceReport(
                    trace_id=f"exec-{uuid.uuid4().hex[:16]}",
                    parent_trace_id=parent_trace_id,
                    agent_id=executor_agent, tool_name=tool,
                    trace_type=TraceType.ATOMIC, success=False,
                    params=params, error_type=result.error_type,
                    error_message=result.error_message,
                    latency_ms=latency_ms, token_count=result.token_count,
                    source="executor",
                ))
            return {
                "tool_name": tool, "params": params, "result": None,
                "status": "failed", "latency_ms": latency_ms,
                "tokens": result.token_count,
                "rules_triggered": node_rules,
                "repair_hint_applied": (
                    repair_applied[-1] if repair_used else None),
                "circuit_limit": circuit_threshold,
            }

    async def _first_fix_hint(self, tool_name: str) -> dict | None:
        """按 tool 查 active 规则（ORDER BY id）→ 第一个 fix 非空的修复建议（D4）。"""
        cursor = await self.conn.execute(
            """SELECT id FROM rules WHERE tool_name=? AND tool_version='1.0.0'
               AND status='active' ORDER BY id ASC""",
            (tool_name,))
        bridge = MCPBridge(self.conn)
        for row in await cursor.fetchall():
            hint = await bridge.get_repair_hint(row["id"])
            if hint is not None and hint.get("fix"):
                return hint
        return None

    async def execute_skill(self, task_id: str, task_description: str,
                            skill: dict, task_params: dict | None,
                            agent_id: str = "anonymous") -> dict:
        """组合入口：装配 + 执行 + record_call（R2）。不写 tasks 状态（API 层职责）。"""
        plan = await PlanAssembler(self.conn).assemble(skill, task_params=task_params)
        result = await self.execute_plan(task_id, task_description, plan,
                                         agent_id=agent_id)
        if plan.get("skill_id"):
            gov = SkillGovernor(self.conn)
            await gov.record_call(
                skill_id=plan["skill_id"],
                success=result["status"] == "success",
                latency_ms=result["total_latency_ms"],
                tokens=result["total_tokens"],
            )
        result["matched_skill"] = plan.get("skill_name")
        result["blocked"] = plan.get("blocked", False)
        result["block_reason"] = plan.get("block_reason")
        # matched_score 由调用方（API 层）用 matcher 的实际分数透传，不在此硬编码
        return result

    async def execute_llm_plan(self, task_id: str, task_description: str,
                               steps: list[dict],
                               agent_id: str = "anonymous") -> dict:
        """llm_plan 纯净顺序执行（D6）：无规则/无修复/无偏好/无 record_call。"""
        start = time.monotonic()
        executor_agent = f"executor:{agent_id}"
        trace_store = TraceStore(self.conn)
        node_results = []
        step_index = 0
        for step in steps:
            tool = step["tool"]
            params = dict(step.get("params", {}))
            node_start = time.monotonic()
            result = await self.adapter.execute(tool, params)
            latency_ms = int((time.monotonic() - node_start) * 1000)
            await trace_store.insert(TraceReport(
                trace_id=f"exec-{uuid.uuid4().hex[:16]}",
                agent_id=executor_agent, tool_name=tool,
                trace_type=TraceType.ATOMIC, success=result.success,
                params=params, result=result.result,
                error_type=result.error_type, error_message=result.error_message,
                latency_ms=result.latency_ms, token_count=result.token_count,
                source="executor",
            ))
            node_results.append({
                "tool_name": tool, "params": params,
                "result": result.result if result.success else None,
                "status": "success" if result.success else "failed",
                "latency_ms": latency_ms, "tokens": result.token_count,
                "rules_triggered": [], "repair_hint_applied": None,
            })
            await self.audit.add_step(
                task_id=task_id, step_index=step_index, tool_name=tool,
                params=params, result=result.result if result.success else None,
                status="success" if result.success else "failed",
                latency_ms=latency_ms, tokens=result.token_count,
                rules_triggered=None, repair_hint_applied=None,
                adapter=type(self.adapter).__name__,
            )
            step_index += 1

        success = all(r["status"] == "success" for r in node_results)
        total_latency_ms = int((time.monotonic() - start) * 1000)
        total_tokens = sum(r["tokens"] for r in node_results)
        summary = f"llm_plan 执行：{sum(1 for r in node_results if r['status']=='success')}/{len(node_results)} 节点成功"
        return self._result("success" if success else "failed", node_results,
                            summary, [], [], total_latency_ms, total_tokens)

    @staticmethod
    def _result(status: str, steps: list[dict], summary: str,
                rules_triggered: list[str], repair_applied: list[dict],
                total_latency_ms: int, total_tokens: int = 0) -> dict:
        return {
            "status": status, "steps": steps, "summary": summary,
            "rules_triggered": rules_triggered,
            "repair_hint_applied": repair_applied,
            "total_latency_ms": total_latency_ms,
            "total_tokens": total_tokens,
        }


def _topo_layers(graph: nx.DiGraph) -> list[list[int]]:
    """Kahn 分层——同层节点无相互依赖，可并行。"""
    remaining = set(graph.nodes)
    layers = []
    while remaining:
        layer = [n for n in remaining
                 if not any(p in remaining for p in graph.predecessors(n))]
        if not layer:
            raise ValueError("cycle")
        layers.append(layer)
        remaining.difference_update(layer)
    return layers
