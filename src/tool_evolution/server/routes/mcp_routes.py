from fastapi import APIRouter
from pydantic import BaseModel
from ...governance.mcp_bridge import MCPBridge

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


class UpdateRequest(BaseModel):
    entity: str
    relations: list[str]


@router.post("/memory/search")
async def search_memory(req: SearchRequest):
    from ..app import _conn
    bridge = MCPBridge(_conn)
    results = await bridge.search_memory(req.query)
    return {"results": results, "count": len(results)}


@router.post("/memory/update")
async def update_memory(req: UpdateRequest):
    from ..app import _conn
    bridge = MCPBridge(_conn)
    await bridge.update_memory(req.entity, req.relations)
    return {"status": "ok", "entity": req.entity}


@router.get("/memory/preferences")
async def get_preferences():
    from ..app import _conn
    bridge = MCPBridge(_conn)
    prefs = await bridge.get_user_preferences()
    return {"preferences": prefs}
