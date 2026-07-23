from fastapi import APIRouter, Request
from ...collection.schemas import TraceReport
from ...collection.store import TraceStore
from ...collection.tracer import Tracer

router = APIRouter()


async def _get_store(request: Request) -> TraceStore:
    conn = request.app.state.db if hasattr(request.app.state, "db") else await request.app.extra["get_db"]()
    return TraceStore(conn)


@router.post("/report")
async def report_trace(report: TraceReport, request: Request):
    conn = request.app.state.db if hasattr(request.app.state, "db") else None
    if conn is None:
        from ..app import _conn
        conn = _conn
    store = TraceStore(conn)
    await store.insert(report)
    return {"status": "accepted", "trace_id": report.trace_id}


@router.post("/seed")
async def seed_traces(reports: list[TraceReport]):
    from ..app import _conn
    store = TraceStore(_conn)
    for report in reports:
        await store.insert(report)
    return {"status": "seeded", "count": len(reports)}


@router.get("/recent")
async def recent_traces(limit: int = 100, offset: int = 0):
    from ..app import _conn
    store = TraceStore(_conn)
    traces = await store.get_all_traces(limit=limit, offset=offset)
    return {"traces": traces, "count": len(traces)}
