import json
import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from ..deps import get_db
from ...analysis.repair_advisor import RepairAdvisor

router = APIRouter()


@router.get("")
async def list_rules(tool_name: str | None = None,
                     conn: aiosqlite.Connection = Depends(get_db)):
    if tool_name:
        cursor = await conn.execute("SELECT * FROM rules WHERE tool_name=?", (tool_name,))
    else:
        cursor = await conn.execute("SELECT * FROM rules")
    rows = await cursor.fetchall()
    return {"rules": [dict(r) for r in rows]}


@router.get("/{rule_id}/hint")
async def get_rule_hint(rule_id: int, conn: aiosqlite.Connection = Depends(get_db)):
    advisor = RepairAdvisor(conn)
    hint = await advisor.get_hint(rule_id)
    if hint is None:
        raise HTTPException(status_code=404, detail="hint not found")
    if hint["fix"] is not None:
        hint["fix"] = json.loads(hint["fix"])   # REST 层解析（口径见计划 Task 3 Interfaces）
    return {"hint": hint}
