from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def list_rules(tool_name: str | None = None):
    from ..app import _conn
    if tool_name:
        cursor = await _conn.execute("SELECT * FROM rules WHERE tool_name=?", (tool_name,))
    else:
        cursor = await _conn.execute("SELECT * FROM rules")
    rows = await cursor.fetchall()
    return {"rules": [dict(r) for r in rows]}
