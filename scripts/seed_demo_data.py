"""Generate 200 simulated traces covering 7 tools, 5 error types, 3 common DAG patterns."""
import asyncio
import random
import sys
sys.path.insert(0, "src")

from tool_evolution.utils.database import get_connection, init_db, run_migrations
from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport, TraceType, ErrorType


TOOLS = ["search_law", "get_law_detail", "analyze_compliance", "generate_report",
         "github_api", "arxiv_api", "official_docs"]
ERRORS = list(ErrorType)
DAG_PATTERNS = [
    ["search_law", "get_law_detail", "analyze_compliance", "generate_report"],
    ["search_law", "analyze_compliance"],
    ["github_api", "analyze_compliance"],
]


async def main():
    conn = await get_connection()
    await init_db(conn)
    await run_migrations(conn)
    store = TraceStore(conn)

    traces = []
    for i in range(50):
        pattern = random.choice(DAG_PATTERNS)
        root_id = f"demo-root-{i}"
        root = TraceReport(
            trace_id=root_id, agent_id="orchestrator",
            tool_name="run_compliance_check", tool_version="1.0.0",
            trace_type=TraceType.TASK_ROOT, success=True, latency_ms=random.randint(2000, 15000),
            token_count=random.randint(500, 3000),
            source="synthetic_demo",
        )
        traces.append(root)
        for j, tool in enumerate(pattern):
            success = random.random() > 0.25
            report = TraceReport(
                trace_id=f"demo-{i}-{j}",
                parent_trace_id=root_id,
                agent_id=tool,
                tool_name=tool, tool_version="1.0.0",
                trace_type=TraceType.ATOMIC,
                success=success,
                params={"query": f"劳动合同法 第{random.randint(1,100)}条",
                        "max_results": random.randint(5, 20)},
                latency_ms=random.randint(50, 5000),
                token_count=random.randint(50, 500),
                source="synthetic_demo",
            )
            if not success:
                report.error_type = random.choice(ERRORS)
                report.error_message = f"[{report.error_type.value}] simulated failure"
            traces.append(report)

    for t in traces:
        await store.insert(t)

    print(f"Seeded {len(traces)} traces (50 tasks, {len(traces)-50} atomic calls)")
    await conn.close()


asyncio.run(main())
