import asyncio

import aiosqlite
from fastapi import APIRouter, Depends
from ...collection.schemas import TraceReport
from ...collection.store import TraceStore
from ..deps import get_db
from ...utils.database import get_connection

router = APIRouter()

_LEARN_THRESHOLD = 20
_learn_counter = 0
_learn_pending: set[asyncio.Task] = set()


async def _learn_prefs_bg() -> None:
    """后台偏好学习（D2-2 修复：偏好学习生产入口）。

    自开连接（请求连接在响应后关闭不可用）；失败静默——学习失败不阻塞轨迹上报。
    """
    from ...analysis.preference_learner import PreferenceLearner
    conn = await get_connection()
    try:
        learner = PreferenceLearner(conn)
        prefs = await learner.learn()
        await learner.save_to_cache(prefs)
    except Exception:
        pass
    finally:
        await conn.close()


def _maybe_schedule_learn() -> None:
    """节流：每 20 条轨迹上报触发一次后台学习（learn() 全库扫描，不可每次触发）。"""
    global _learn_counter
    _learn_counter += 1
    if _learn_counter % _LEARN_THRESHOLD != 0:
        return
    task = asyncio.create_task(_learn_prefs_bg())
    _learn_pending.add(task)
    task.add_done_callback(_learn_pending.discard)


@router.post("/report")
async def report_trace(report: TraceReport, conn: aiosqlite.Connection = Depends(get_db)):
    store = TraceStore(conn)
    await store.insert(report)
    _maybe_schedule_learn()
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
