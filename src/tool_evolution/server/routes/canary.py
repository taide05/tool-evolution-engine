from fastapi import APIRouter
from ...governance.governor import SkillGovernor

router = APIRouter()


@router.post("/{skill_id}/promote")
async def promote_skill(skill_id: int):
    from ..app import _conn
    gov = SkillGovernor(_conn)
    new_status = await gov.promote(skill_id)
    return {"skill_id": skill_id, "new_status": new_status}


@router.post("/{skill_id}/compare")
async def ab_compare(skill_id: int, old_rate: float, new_rate: float):
    from ..app import _conn
    gov = SkillGovernor(_conn)
    result = await gov.ab_compare(skill_id, old_rate, new_rate)
    return result
