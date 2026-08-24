import aiosqlite
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from ...knowledge.param_template import ParamTemplateManager, flatten_user_prefs
from ...governance.mcp_bridge import MCPBridge
from ..deps import get_db

router = APIRouter()


class GenerateRequest(BaseModel):
    tool_name: str
    tool_version: str = "1.0.0"


@router.post("/templates/generate")
async def generate_template(req: GenerateRequest, conn: aiosqlite.Connection = Depends(get_db)):
    bridge = MCPBridge(conn)
    prefs = await bridge.get_user_preferences()
    flat = flatten_user_prefs(prefs, req.tool_name)
    tmpl = await ParamTemplateManager(conn).generate(
        req.tool_name, req.tool_version, user_prefs=flat
    )
    return {"tool_name": req.tool_name, "template": tmpl, "prefs_applied": flat}
