import aiosqlite
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from ...governance.mcp_bridge import MCPBridge
from ..deps import get_db

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


class UpdateRequest(BaseModel):
    entity: str
    relations: list[str]


@router.post("/memory/search")
async def search_memory(req: SearchRequest, conn: aiosqlite.Connection = Depends(get_db)):
    results = await MCPBridge(conn).search_memory(req.query)
    return {"results": results, "count": len(results)}


@router.post("/memory/update")
async def update_memory(req: UpdateRequest, conn: aiosqlite.Connection = Depends(get_db)):
    await MCPBridge(conn).update_memory(req.entity, req.relations)
    return {"status": "ok", "entity": req.entity}


@router.get("/memory/preferences")
async def get_preferences(conn: aiosqlite.Connection = Depends(get_db)):
    prefs = await MCPBridge(conn).get_user_preferences()
    return {"preferences": prefs}


@router.get("/memory/relations")
async def get_relations(entity: str, conn: aiosqlite.Connection = Depends(get_db)):
    rels = await MCPBridge(conn).search_relations(entity)
    return {"entity": entity, "relations": rels, "count": len(rels)}
