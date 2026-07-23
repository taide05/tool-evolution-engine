from fastapi import APIRouter

router = APIRouter()


@router.get("/summary")
async def analytics_summary():
    from ..app import _conn
    cursor = await _conn.execute("SELECT COUNT(*) as total FROM trajectories")
    total = (await cursor.fetchone())[0]

    cursor = await _conn.execute("SELECT COUNT(*) FROM trajectories WHERE success=0")
    failures = (await cursor.fetchone())[0]

    cursor = await _conn.execute("SELECT AVG(latency_ms) FROM trajectories")
    avg_lat = (await cursor.fetchone())[0]

    cursor = await _conn.execute("SELECT AVG(token_count) FROM trajectories")
    avg_tok = (await cursor.fetchone())[0]

    cursor = await _conn.execute("SELECT COUNT(*) FROM deployed_skills WHERE status='active'")
    active_skills = (await cursor.fetchone())[0]

    return {
        "total_traces": total,
        "total_failures": failures,
        "failure_rate": round(failures / max(total, 1), 4),
        "avg_latency_ms": round(avg_lat or 0, 2),
        "avg_tokens": round(avg_tok or 0, 2),
        "active_skills": active_skills,
    }
