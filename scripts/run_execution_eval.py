"""执行层消费端评测——50 任务 skill_plan vs llm_plan 对比 + 修复闭环复验 + 口径隔离验证。

独立于 run_eval 的 gsm 体系（口径隔离：管道指标 vs 消费端对比）。
固定 Random(42) 可复现；llm_plan 对比需 TOOLEVO_DEEPSEEK_API_KEY。
"""

import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiosqlite

from tool_evolution.analysis.dag_miner import DAGMiner
from tool_evolution.collection.store import TraceStore
from tool_evolution.execution.adapters import MockAdapter
from tool_evolution.execution.assembler import PlanAssembler
from tool_evolution.execution.audit import ExecutionAudit
from tool_evolution.execution.executor import SkillExecutor
from tool_evolution.execution.matcher import SkillMatcher
from tool_evolution.execution.planner import LLMPlanner
from tool_evolution.governance.governor import SkillGovernor
from tool_evolution.knowledge.rule_engine import RuleEngine
from tool_evolution.knowledge.skill_pack import SkillPackManager
from tool_evolution.utils.config import settings

_RNG = random.Random(42)
_TASKS_PATH = Path(__file__).parent / "benchmark_tasks.json"
_EVAL_PREFIX = "exec-eval-"
_LLM_PREFIX = "exec-llm-"
_REPAIR_PREFIX = "exec-repair-"


async def _clear_db(conn: aiosqlite.Connection) -> None:
    for stmt in (
        "DELETE FROM canary_invocations",
        "DELETE FROM execution_steps",
        "DELETE FROM execution_tasks",
        "DELETE FROM deployed_skills",
        "DELETE FROM discovered_skills",
        "DELETE FROM param_distributions",
        "DELETE FROM rules",
        "DELETE FROM trajectories_fts",
        "DELETE FROM trajectories",
        "DELETE FROM memory_cache",
    ):
        await conn.execute(stmt)
    await conn.commit()


async def _deploy_active_skills(conn: aiosqlite.Connection) -> list[dict]:
    """DAG 挖掘 → promote 到 active（走真实三级阶梯）。"""
    store = TraceStore(conn)
    all_traces = await store.get_all_traces(limit=50000)
    skills = DAGMiner().mine(all_traces)
    mgr = SkillPackManager(conn)
    gov = SkillGovernor(conn)
    for skill in skills:
        discovery_id = await mgr.add_discovery(skill)
        deployed_id = await mgr.promote_to_deployed(discovery_id)
        for _ in range(3):
            await gov.promote(deployed_id)
    return await mgr.list_deployed(status_filter="active")


async def _run_skill_plan_pass(conn, tasks, active_skills, matcher, executor,
                               audit, threshold) -> dict:
    assembler = PlanAssembler(conn)
    matched = 0
    success = 0
    blocked = 0
    failed = 0
    total_latency = 0
    total_tokens = 0
    hit_task_ids = []
    for task in tasks:
        desc = f"调用 {' '.join(task['tool_chain'])} 完成 {task['task_name']}"
        m = await matcher.match(desc)
        if m is None:
            continue
        matched += 1
        hit_task_ids.append(task["task_id"])
        await audit.create_task(
            task_id=f"{_EVAL_PREFIX}{task['task_id']}", task_description=desc,
            mode="skill_plan", plan=None, skill_name=m["skill"]["name"])
        plan = await assembler.assemble(m["skill"], task_params=task["root_params"])
        result = await executor.execute_plan(
            f"{_EVAL_PREFIX}{task['task_id']}", desc, plan, agent_id="eval")
        if plan["blocked"]:
            blocked += 1
        elif result["status"] == "success":
            success += 1
        else:
            failed += 1
        total_latency += result["total_latency_ms"]
        total_tokens += result["total_tokens"]
    return {
        "matched": matched,
        "match_rate": round(matched / len(tasks), 3),
        "threshold": threshold,
        "success": success, "blocked": blocked, "failed": failed,
        "success_rate": round(success / max(matched - blocked, 1), 3),
        "total_latency_ms": total_latency,
        "total_tokens": total_tokens,
        "planning_cost_ms": 0,
        "avg_planning_ms": 0.0,
        "hit_task_ids": hit_task_ids,
    }


async def _run_llm_plan_pass(conn, tasks, hit_task_ids, executor, audit) -> dict:
    if not settings.deepseek_api_key:
        return {
            "mode": "degraded_no_key",
            "success": 0, "failed": 0, "success_rate": None,
            "total_latency_ms": 0, "total_tokens": 0,
            "avg_planning_ms": None, "avg_planning_tokens": None,
        }
    planner = LLMPlanner()
    try:
        desc_by_id = {t["task_id"]: f"调用 {' '.join(t['tool_chain'])} 完成 {t['task_name']}"
                      for t in tasks}
        success = 0
        failed = 0
        total_latency = 0
        total_tokens = 0
        planning_ms_total = 0
        for task_id in hit_task_ids:
            desc = desc_by_id[task_id]
            t0 = time.monotonic()
            steps = await planner.plan(desc)
            planning_ms_total += (time.monotonic() - t0) * 1000
            if steps is None:
                failed += 1
                continue
            await audit.create_task(
                task_id=f"{_LLM_PREFIX}{task_id}", task_description=desc,
                mode="llm_plan", plan=None)
            result = await executor.execute_llm_plan(
                f"{_LLM_PREFIX}{task_id}", desc, steps, agent_id="eval")
            if result["status"] == "success":
                success += 1
            else:
                failed += 1
            total_latency += result["total_latency_ms"]
            total_tokens += result["total_tokens"]
        n = max(success + failed, 1)
        return {
            "mode": "live",
            "success": success, "failed": failed,
            "success_rate": round(success / n, 3),
            "total_latency_ms": total_latency, "total_tokens": total_tokens,
            "avg_planning_ms": round(planning_ms_total / max(len(hit_task_ids), 1), 1),
            "avg_planning_tokens": None,
        }
    finally:
        await planner.aclose()


async def _run_repair_replay(conn, executor, audit, n_cases: int = 5) -> dict:
    engine = RuleEngine(conn)
    rule_id = await engine.add_rule({
        "tool_name": "search_api", "tool_version": "1.0.0",
        "rule_type": "range_rule",
        "condition": {"param_names": ["max_results"]},
        "action": {"validate_before_call": True}, "status": "active",
    })
    await conn.execute(
        """INSERT INTO repair_hints (rule_id, content_hash, suggestion, fix, model)
           VALUES (?, 'eval-h', 'max_results 用整数', ?, 'deepseek-v4-flash')""",
        (rule_id, json.dumps({"param": "max_results", "suggested_value": 5})),
    )
    await conn.commit()
    rescued = 0
    for i in range(n_cases):
        task_id = f"{_REPAIR_PREFIX}{i}"
        await audit.create_task(
            task_id=task_id, task_description="修复复验", mode="skill_plan",
            plan=None, skill_name="repair_case")
        plan = {
            "skill_id": None, "skill_name": "repair_case",
            "nodes": [{"tool_name": "search_api",
                       "params": {"query": f"q{i}", "max_results": "many"}}],
            "edges": [], "precheck_rules": [], "blocked": False,
            "block_reason": None,
        }
        result = await executor.execute_plan(task_id, "修复复验", plan,
                                             agent_id="eval")
        if result["status"] == "success":
            rescued += 1
    return {"cases": n_cases, "rescued": rescued,
            "rescue_rate": round(rescued / n_cases, 3)}


async def _run_isolation_check(conn) -> dict:
    store = TraceStore(conn)
    miner = DAGMiner()
    with_exec = {s["name"] for s in miner.mine(await store.get_all_traces(limit=50000))}
    without_exec = {
        s["name"] for s in miner.mine(
            await store.get_all_traces(limit=50000, exclude_agent_prefix="executor:"))}
    exec_only = sorted(with_exec - without_exec)
    return {"exec_only_skills": len(exec_only), "exec_only_names": exec_only}


async def run_execution_eval(conn: aiosqlite.Connection) -> dict:
    from scripts.run_eval import seed_eval_data  # 延迟 import 防循环

    await _clear_db(conn)
    await seed_eval_data(conn, n_tasks=200)
    active_skills = await _deploy_active_skills(conn)
    tasks = json.loads(_TASKS_PATH.read_text(encoding="utf-8"))

    audit = ExecutionAudit(conn)
    adapter = MockAdapter()
    executor = SkillExecutor(conn, adapter, audit=audit)
    try:
        matcher = SkillMatcher(conn)
        threshold = settings.skill_match_threshold
        skill_plan = await _run_skill_plan_pass(
            conn, tasks, active_skills, matcher, executor, audit, threshold)
        # 阈值调定程序：命中率落 60-90% 区间，否则 0.3/0.7 两档扫
        for candidate in (0.3, 0.7):
            if 0.6 <= skill_plan["match_rate"] <= 0.9:
                break
            matcher = SkillMatcher(conn, threshold=candidate)
            skill_plan = await _run_skill_plan_pass(
                conn, tasks, active_skills, matcher, executor, audit, candidate)

        llm_plan = await _run_llm_plan_pass(
            conn, tasks, skill_plan["hit_task_ids"], executor, audit)
        repair_replay = await _run_repair_replay(conn, executor, audit)
        executor_isolation = await _run_isolation_check(conn)
    finally:
        await adapter.close()

    result = {
        "n_tasks": len(tasks),
        "skill_match": {
            "matched": skill_plan["matched"],
            "match_rate": skill_plan["match_rate"],
            "threshold": skill_plan["threshold"],
            "active_skills": [s["name"] for s in active_skills],
        },
        "skill_plan": {
            "success": skill_plan["success"],
            "blocked": skill_plan["blocked"],
            "failed": skill_plan["failed"],
            "success_rate": skill_plan["success_rate"],
            "total_latency_ms": skill_plan["total_latency_ms"],
            "total_tokens": skill_plan["total_tokens"],
            "planning_cost_ms": skill_plan["planning_cost_ms"],
            "avg_planning_ms": skill_plan["avg_planning_ms"],
        },
        "llm_plan": llm_plan,
        "comparison": {"same_task_subset": len(skill_plan["hit_task_ids"])},
        "repair_replay": repair_replay,
        "executor_isolation": executor_isolation,
    }
    return result


def _print_report(result: dict) -> None:
    print("=== 执行层消费端评测 ===")
    print(f"任务数: {result['n_tasks']} | active 技能: "
          f"{result['skill_match']['active_skills']}")
    sm = result["skill_match"]
    print(f"匹配: {sm['matched']}/{result['n_tasks']} = {sm['match_rate']} "
          f"(阈值 {sm['threshold']})")
    sp = result["skill_plan"]
    print(f"skill_plan: 成功 {sp['success']} / blocked {sp['blocked']} / "
          f"失败 {sp['failed']} | 成功率 {sp['success_rate']} "
          f"(分母剔除 blocked) | 规划成本 {sp['planning_cost_ms']}ms | "
          f"耗时 {sp['total_latency_ms']}ms | tokens {sp['total_tokens']}")
    lp = result["llm_plan"]
    if lp.get("mode") == "degraded_no_key":
        print("llm_plan: degraded（无 DEEPSEEK key）——对比指标 N/A（诚实声明）")
    else:
        print(f"llm_plan: 成功 {lp['success']} / 失败 {lp['failed']} | "
              f"成功率 {lp['success_rate']} | 平均规划 {lp['avg_planning_ms']}ms | "
              f"耗时 {lp['total_latency_ms']}ms | tokens {lp['total_tokens']}")
    rr = result["repair_replay"]
    print(f"修复闭环复验: {rr['rescued']}/{rr['cases']} rescued "
          f"({rr['rescue_rate']})")
    iso = result["executor_isolation"]
    print(f"executor 隔离: 排除后挖掘少 {iso['exec_only_skills']} 个技能 "
          f"{iso['exec_only_names']}")


async def _main() -> None:
    conn = await aiosqlite.connect(settings.db_path)
    conn.row_factory = aiosqlite.Row
    try:
        result = await run_execution_eval(conn)
        _print_report(result)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(_main())
