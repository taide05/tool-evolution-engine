import aiosqlite
from fastapi import APIRouter, Depends
from ...governance.governor import SkillGovernor
from ..deps import get_db

router = APIRouter()


@router.post("/{skill_id}/promote")
async def promote_skill(skill_id: int, conn: aiosqlite.Connection = Depends(get_db)):
    gov = SkillGovernor(conn)
    new_status = await gov.promote(skill_id)
    return {"skill_id": skill_id, "new_status": new_status}
