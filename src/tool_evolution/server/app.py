import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from ..utils.database import get_connection, init_db, run_migrations
from ..utils.config import settings
from .routes import traces, skills, rules, analytics, canary, mcp_routes

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
            pass
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
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scoring_task
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
async def health():
    return {"status": "ok"}
