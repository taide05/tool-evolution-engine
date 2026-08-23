import aiosqlite
from fastapi import APIRouter, Depends
from ...collection.schemas import TraceReport
from ...collection.store import TraceStore
from ..deps import get_db

router = APIRouter()


@router.post("/report")
async def report_trace(report: TraceReport, conn: aiosqlite.Connection = Depends(get_db)):
    store = TraceStore(conn)
    await store.insert(report)
    return {"status": "accepted", "trace_id": report.trace_id}


@router.post("/seed")
async def seed_traces(reports: list[TraceReport], conn: aiosqlite.Connection = Depends(get_db)):
    store = TraceStore(conn)
    for report in reports:
        await store.insert(report)
    return {"status": "seeded", "count": len(reports)}


@router.get("/recent")
async def recent_traces(limit: int = 100, offset: int = 0,
                        conn: aiosqlite.Connection = Depends(get_db)):
    store = TraceStore(conn)
    traces = await store.get_all_traces(limit=limit, offset=offset)
    return {"traces": traces, "count": len(traces)}
