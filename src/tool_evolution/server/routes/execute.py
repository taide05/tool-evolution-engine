"""执行层 API——POST /api/execute/task（同步执行+幂等）/ GET 查询审计。"""

import json

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...execution.adapters import AsyncToolAdapter, HTTPAdapter, MCPAdapter, MockAdapter
from ...execution.audit import ExecutionAudit
from ...execution.executor import SkillExecutor
from ...execution.matcher import SkillMatcher
from ...execution.planner import LLMPlanner
from ..deps import get_db

router = APIRouter()

_VALID_MODES = ("auto", "skill_plan", "llm_plan")
_VALID_ADAPTERS = ("mock", "http", "mcp")


class ExecuteTaskRequest(BaseModel):
    task_id: str
    task_description: str
    mode: str = "auto"
    params: dict | None = None
    agent_id: str = "anonymous"
    adapter: str = "mock"
    http_base_url: str | None = None
    mcp_command: str | None = None
    mcp_args: list[str] = []
    mcp_cwd: str | None = None


def _make_adapter(req: ExecuteTaskRequest) -> AsyncToolAdapter:
    """按请求参数构造适配器（I#5 修复——HTTP/MCP 接入入口）。"""
    if req.adapter == "mock":
        return MockAdapter()
    if req.adapter == "http":
        if not req.http_base_url:
            raise HTTPException(status_code=422,
                                detail="adapter=http requires http_base_url")
        return HTTPAdapter(req.http_base_url)
    if req.adapter == "mcp":
        if not req.mcp_command:
            raise HTTPException(status_code=422,
                                detail="adapter=mcp requires mcp_command")
        return MCPAdapter(command=req.mcp_command, args=req.mcp_args,
                          cwd=req.mcp_cwd)
    raise HTTPException(status_code=422,
                        detail=f"invalid adapter '{req.adapter}'")


_planner: LLMPlanner | None = None


def _get_planner() -> LLMPlanner:
    """进程级复用 planner（httpx 连接池常驻——I#6 修复）。"""
    global _planner
    if _planner is None:
        _planner = LLMPlanner()
    return _planner


@router.post("/execute/task")
async def execute_task(req: ExecuteTaskRequest,
                       conn: aiosqlite.Connection = Depends(get_db)):
    if req.mode not in _VALID_MODES:
        raise HTTPException(status_code=422, detail=f"invalid mode '{req.mode}'")
    if req.adapter not in _VALID_ADAPTERS:
        raise HTTPException(status_code=422,
                            detail=f"invalid adapter '{req.adapter}'")

    matcher = SkillMatcher(conn)
    matched = await matcher.match(req.task_description) if req.mode != "llm_plan" else None

    if req.mode == "skill_plan" and matched is None:
        raise HTTPException(status_code=404, detail="no skill matched")

    mode_used = "skill_plan" if matched is not None else "llm_plan"
    audit = ExecutionAudit(conn)
    plan_for_audit = None
    if matched is not None:
        try:
            plan_for_audit = json.loads(matched["skill"]["dag_definition"])
        except (json.JSONDecodeError, TypeError):
            plan_for_audit = None
    inserted = await audit.create_task(
        task_id=req.task_id, task_description=req.task_description,
        mode=mode_used, plan=plan_for_audit,
        skill_name=matched["skill"]["name"] if matched else None,
    )
    if not inserted:
        existing = await audit.get_task(req.task_id)
        if existing is None:
            raise HTTPException(status_code=500, detail="task row vanished")
        if existing["status"] in ("pending", "running"):
            raise HTTPException(
                status_code=409,
                detail={
                    "detail": "task in progress",
                    "existing": {"task_id": existing["task_id"],
                                 "status": existing["status"]},
                    "hint": f"poll GET /api/execute/task/{req.task_id}",
                })
        return _task_response(existing)

    await audit.update_status(req.task_id, "running")

    adapter = _make_adapter(req)
    executor = SkillExecutor(conn, adapter, audit=audit)
    try:
        if matched is not None:
            # execute_skill 组合入口：内部 assemble + execute_plan + record_call（R2 闭环）
            result = await executor.execute_skill(
                req.task_id, req.task_description, matched["skill"],
                task_params=req.params, agent_id=req.agent_id)
            result["matched_skill"] = matched["skill"]["name"]
            result["matched_score"] = matched["score"]
        else:
            steps = await _get_planner().plan(req.task_description)
            if steps is None:
                result = {
                    "status": "failed",
                    "steps": [],
                    "summary": "LLM 规划不可用或校验失败（对照组 fail-closed）",
                    "rules_triggered": [], "repair_hint_applied": [],
                    "total_latency_ms": 0, "total_tokens": 0,
                }
            else:
                result = await executor.execute_llm_plan(
                    req.task_id, req.task_description, steps, agent_id=req.agent_id)
    finally:
        await adapter.close()

    await audit.update_status(
        req.task_id, result["status"], summary=result.get("summary"))
    task = await audit.get_task(req.task_id)
    return _task_response(task, result=result)


@router.get("/execute/task/{task_id}")
async def get_execute_task(task_id: str,
                           conn: aiosqlite.Connection = Depends(get_db)):
    audit = ExecutionAudit(conn)
    task = await audit.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


def _task_response(task: dict, result: dict | None = None) -> dict:
    resp = {
        "task_id": task["task_id"],
        "mode_used": task["mode"],
        "skill_name": task.get("skill_name"),
        "status": task["status"],
        "summary": task.get("summary"),
        "steps": task.get("steps", []),
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
    }
    if result is not None:
        resp.update({
            "matched_skill": result.get("matched_skill"),
            "matched_score": result.get("matched_score"),
            "rules_triggered": result.get("rules_triggered", []),
            "repair_hint_applied": result.get("repair_hint_applied", []),
            "total_latency_ms": result.get("total_latency_ms", 0),
            "total_tokens": result.get("total_tokens", 0),
        })
    return resp
