import aiosqlite
from fastapi import APIRouter, Depends
from ..deps import get_db

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
