import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from ..utils.database import get_connection, init_db
from ..utils.config import settings
from .routes import traces, skills, rules, analytics, canary, mcp_routes

_conn = None
_scoring_task: asyncio.Task | None = None


async def get_db():
    global _conn
    if _conn is None:
        _conn = await get_connection()
        await init_db(_conn)
    return _conn


async def _periodic_scoring(conn):
    from ..governance.governor import SkillGovernor
    gov = SkillGovernor(conn)
    while True:
        await asyncio.sleep(settings.credit_update_interval_s)
        try:
            await gov.update_all_scores()
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conn, _scoring_task
    _conn = await get_connection()
    await init_db(_conn)
    _scoring_task = asyncio.create_task(_periodic_scoring(_conn))
    yield
    if _scoring_task:
        _scoring_task.cancel()
    if _conn:
        await _conn.close()


app = FastAPI(title="Tool Evolution Engine", version="0.1.0", lifespan=lifespan)

app.include_router(traces.router, prefix="/api/traces", tags=["traces"])
app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
app.include_router(rules.router, prefix="/api/rules", tags=["rules"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(canary.router, prefix="/api/canary", tags=["canary"])
app.include_router(mcp_routes.router, prefix="/api", tags=["memory"])


@app.get("/health")
async def health():
    return {"status": "ok"}

app.extra = {"get_db": get_db}
