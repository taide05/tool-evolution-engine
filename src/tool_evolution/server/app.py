from contextlib import asynccontextmanager
from fastapi import FastAPI
from ..utils.database import get_connection, init_db
from .routes import traces, skills, rules, analytics, canary

_conn = None


async def get_db():
    global _conn
    if _conn is None:
        _conn = await get_connection()
        await init_db(_conn)
    return _conn


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conn
    _conn = await get_connection()
    await init_db(_conn)
    yield
    if _conn:
        await _conn.close()


app = FastAPI(title="Tool Evolution Engine", version="0.1.0", lifespan=lifespan)

app.include_router(traces.router, prefix="/api/traces", tags=["traces"])
app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
app.include_router(rules.router, prefix="/api/rules", tags=["rules"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(canary.router, prefix="/api/canary", tags=["canary"])


@app.get("/health")
async def health():
    return {"status": "ok"}

app.extra = {"get_db": get_db}
