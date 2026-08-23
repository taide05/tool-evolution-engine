import aiosqlite
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from ...knowledge.skill_pack import SkillPackManager
from ...governance.canary_router import CanaryRouter
from ...governance.governor import SkillGovernor
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
    """Invoke a deployed skill with canary-aware routing.

    Uses consistent hashing on request_hash to deterministically route
    between stable (default behavior) and canary (optimized) variants.
    Records invocation metrics for subsequent A/B comparison.
    """
    router_inst = CanaryRouter(conn)
    skill = await router_inst.get_skill(name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")

    request_hash = req.request_hash or str(hash(frozenset(req.params.items())))
    variant = router_inst.decide(request_hash, skill["status"])

    # Simulate execution — in production this would call the actual tool DAG.
    # For benchmarking, the caller supplies the outcome via a follow-up report.
    simulated_success = True
    simulated_latency = 100
    simulated_tokens = 200

    await router_inst.record_invocation(
        skill["id"], variant,
        success=simulated_success,
        latency_ms=simulated_latency,
        tokens=simulated_tokens,
    )

    # Also update the skill governor stats
    gov = SkillGovernor(conn)
    await gov.record_call(skill["id"], success=True, latency_ms=100, tokens=200)

    return {
        "skill": name,
        "status": skill["status"],
        "variant": variant,
        "canary_pct": CanaryRouter.canary_pct(skill["status"]),
        "result": {"status": "ok"},
    }


@router.post("/{discovery_id}/promote")
async def promote_skill(discovery_id: int, conn: aiosqlite.Connection = Depends(get_db)):
    mgr = SkillPackManager(conn)
    deployed_id = await mgr.promote_to_deployed(discovery_id)
    return {"deployed_id": deployed_id, "status": "promoted"}
