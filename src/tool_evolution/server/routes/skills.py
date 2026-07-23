from fastapi import APIRouter
from pydantic import BaseModel
from ...knowledge.skill_pack import SkillPackManager

router = APIRouter()


class SkillResponse(BaseModel):
    id: int
    name: str
    dag_definition: dict
    param_template: dict | None
    credit_score: float
    status: str


@router.get("/discoveries")
async def list_discoveries():
    from ..app import _conn
    mgr = SkillPackManager(_conn)
    return {"discoveries": await mgr.list_discoveries()}


@router.get("/deployed")
async def list_deployed(status: str | None = None):
    from ..app import _conn
    mgr = SkillPackManager(_conn)
    return {"skills": await mgr.list_deployed(status)}


@router.post("/{discovery_id}/promote")
async def promote_skill(discovery_id: int):
    from ..app import _conn
    mgr = SkillPackManager(_conn)
    deployed_id = await mgr.promote_to_deployed(discovery_id)
    return {"deployed_id": deployed_id, "status": "promoted"}
