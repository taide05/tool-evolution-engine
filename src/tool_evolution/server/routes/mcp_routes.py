from fastapi import APIRouter
from pydantic import BaseModel
from ...governance.mcp_bridge import MCPBridge, set_bridge

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


class UpdateRequest(BaseModel):
    entity: str
    relations: list[str]


def _bridge():
    from ..app import _conn
    bridge = MCPBridge(_conn)
    set_bridge(bridge)
    return bridge


@router.post("/memory/search")
async def search_memory(req: SearchRequest):
    results = await _bridge().search_memory(req.query)
    return {"results": results, "count": len(results)}


@router.post("/memory/update")
async def update_memory(req: UpdateRequest):
    await _bridge().update_memory(req.entity, req.relations)
    return {"status": "ok", "entity": req.entity}


@router.get("/memory/preferences")
async def get_preferences():
    prefs = await _bridge().get_user_preferences()
    return {"preferences": prefs}
