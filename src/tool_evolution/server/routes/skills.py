import aiosqlite
import hashlib
import json
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from ...knowledge.skill_pack import SkillPackManager
from ...governance.canary_router import CanaryRouter
from ..deps import get_db

router = APIRouter()


class SkillResponse(BaseModel):
    id: int
    name: str
    dag_definition: dict
    param_template: dict | None
    credit_score: float
    status: str


class InvokeRequest(BaseModel):
    params: dict = {}
    request_hash: str | None = None


@router.get("/discoveries")
async def list_discoveries(conn: aiosqlite.Connection = Depends(get_db)):
    mgr = SkillPackManager(conn)
    return {"discoveries": await mgr.list_discoveries()}


@router.get("/deployed")
async def list_deployed(status: str | None = None,
                        conn: aiosqlite.Connection = Depends(get_db)):
    mgr = SkillPackManager(conn)
    return {"skills": await mgr.list_deployed(status)}


@router.post("/{name}/invoke")
async def invoke_skill(name: str, req: InvokeRequest,
                       conn: aiosqlite.Connection = Depends(get_db)):
    """Route a request between stable and canary variants via consistent hashing.

    Pure routing decision — no execution, no metric writes. Real invocation
    receipts are recorded by the executor closed loop (increment 3):
    canary 桶执行优化装配、stable 桶执行基线装配，两变体实测均落
    canary_invocations（compare_variants 的 A/B 真实样本）。
    """
    router_inst = CanaryRouter(conn)
    skill = await router_inst.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

    if req.request_hash:
        request_hash = req.request_hash
    else:
        try:
            request_hash = hashlib.md5(
                json.dumps(req.params, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="params must be JSON-serializable")

    variant = router_inst.decide(request_hash, skill["status"])
    return {
        "skill": name,
        "status": skill["status"],
        "variant": variant,
        "canary_pct": CanaryRouter.canary_pct(skill["status"]),
        "result": {"status": "routed"},
    }


@router.post("/{discovery_id}/promote")
async def promote_skill(discovery_id: int, conn: aiosqlite.Connection = Depends(get_db)):
    mgr = SkillPackManager(conn)
    deployed_id = await mgr.promote_to_deployed(discovery_id)
    return {"deployed_id": deployed_id, "status": "promoted"}
