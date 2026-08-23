import asyncio
import logging
from contextlib import asynccontextmanager
import aiosqlite
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from ..utils.database import get_connection, init_db, run_migrations
from ..utils.config import settings
from ..utils.logging import setup_logging
from .auth import require_api_key, api_key_middleware
from .deps import get_db
from .routes import traces, skills, rules, analytics, canary, mcp_routes

logger = logging.getLogger(__name__)

_scoring_task: asyncio.Task | None = None


async def _periodic_scoring(conn):
    from ..governance.governor import SkillGovernor
    from ..governance.canary_router import CanaryRouter
    from ..knowledge.skill_pack import SkillPackManager
    gov = SkillGovernor(conn)
    router = CanaryRouter(conn)
    skill_mgr = SkillPackManager(conn)
    while True:
        await asyncio.sleep(settings.credit_update_interval_s)
        try:
            await gov.update_all_scores()
        except Exception:
            logger.exception("scoring task failed")
        try:
            deployed = await skill_mgr.list_deployed()
            for skill in deployed:
                if skill["status"] not in ("canary_5", "canary_15", "canary_50"):
                    continue
                comparison = await router.compare_variants(
                    skill["id"], min_samples=settings.canary_min_samples
                )
                if comparison is None:
                    continue
                if comparison["rollback"]:
                    await gov.demote(skill["id"], "canary underperformance")
                elif comparison["promote"]:
                    await gov.promote(skill["id"])
        except Exception:
            logger.exception("canary comparison failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scoring_task
    setup_logging(settings.log_level)
    require_api_key()
    conn = await get_connection()
    await init_db(conn)
    await run_migrations(conn)
    _scoring_task = asyncio.create_task(_periodic_scoring(conn))
    yield
    if _scoring_task:
        _scoring_task.cancel()
    await conn.close()


app = FastAPI(title="Tool Evolution Engine", version="0.1.0", lifespan=lifespan)

app.include_router(traces.router, prefix="/api/traces", tags=["traces"])
app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
app.include_router(rules.router, prefix="/api/rules", tags=["rules"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(canary.router, prefix="/api/canary", tags=["canary"])
app.include_router(mcp_routes.router, prefix="/api", tags=["memory"])


@app.get("/health")
async def health(conn: aiosqlite.Connection = Depends(get_db)):
    try:
        await conn.execute("SELECT 1")
    except Exception:
        logger.exception("health check failed")
        return JSONResponse(status_code=503, content={"status": "degraded"})
    return {"status": "ok"}


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(aiosqlite.Error)
async def db_error_handler(request, exc: aiosqlite.Error):
    logger.exception("database error")
    return JSONResponse(status_code=503, content={"detail": "database error"})


@app.exception_handler(Exception)
async def unhandled_error_handler(request, exc: Exception):
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


app.middleware("http")(api_key_middleware)
