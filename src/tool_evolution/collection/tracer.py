import uuid
import asyncio
import aiosqlite
from .schemas import TraceReport
from .store import TraceStore


class Tracer:
    def __init__(self, conn: aiosqlite.Connection, batch_size: int = 100,
                 flush_interval_s: int = 5, mcp_bridge=None):
        self.store = TraceStore(conn)
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._mcp_bridge = mcp_bridge
        self._queue: asyncio.Queue[TraceReport] = asyncio.Queue()
        self._flush_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def _periodic_flush(self) -> None:
        while self._running:
            await asyncio.sleep(self.flush_interval_s)
            if self._queue.qsize() > 0:
                await self._flush_batch()

    def start_trace(self, agent_id: str, tool_name: str, **kwargs) -> TraceReport:
        return TraceReport(
            trace_id=str(uuid.uuid4()),
            agent_id=agent_id,
            tool_name=tool_name,
            success=False,
            latency_ms=0,
            **kwargs
        )

    async def report(self, report: TraceReport) -> None:
        await self._queue.put(report)
        if self._queue.qsize() >= self.batch_size:
            await self._flush_batch()

    async def flush(self) -> None:
        await self._flush_batch()

    async def _flush_batch(self) -> None:
        batch: list[TraceReport] = []
        while not self._queue.empty() and len(batch) < self.batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        for report in batch:
            await self.store.insert(report)
            if self._mcp_bridge and report.success and report.result:
                await self._mcp_bridge.extract_and_update({
                    "success": True,
                    "result": report.result,
                    "tool_name": report.tool_name,
                })
                if report.parent_trace_id:
                    await self._mcp_bridge.extract_relations(report.parent_trace_id)

    async def close(self) -> None:
        await self.flush()
