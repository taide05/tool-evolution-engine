"""
DeepChoice integration example: demonstrating how DeepChoice's base.py
would integrate the tool-evolution-engine Tracer for observability.

This file is a standalone runnable demo. It mimics the integration pattern
that should be applied to real retriever subclasses in DeepChoice.

Usage:
    python examples/deepchoice_integration.py
"""

import asyncio
import time
from typing import Any

from tool_evolution.collection.tracer import Tracer
from tool_evolution.collection.schemas import ErrorType, TraceReport
from tool_evolution.utils.database import get_connection, init_db


# ---------------------------------------------------------------------------
# Error classification helper
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> ErrorType:
    """Map a Python exception to a tool-evolution ErrorType.

    Uses the closest matching category from the engine's five-class taxonomy:
    param_error, permission_denied, quota_exhausted, timeout, service_unavailable.
    """
    exc_name = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    if any(k in exc_name or k in exc_msg for k in ("timeout", "timed")):
        return ErrorType.TIMEOUT
    if any(k in exc_name or k in exc_msg for k in ("quota", "ratelimit", "rate")):
        return ErrorType.QUOTA_EXHAUSTED
    if any(k in exc_name or k in exc_msg for k in ("permission", "forbidden", "unauthorized", "auth")):
        return ErrorType.PERMISSION_DENIED
    if any(k in exc_name or k in exc_msg
           for k in ("typeerror", "valueerror", "keyerror", "attribute", "param")):
        return ErrorType.PARAM_ERROR
    return ErrorType.SERVICE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Integration pattern: what base.py's search() looks like with Tracer
# ---------------------------------------------------------------------------

class TracedBaseRetriever:
    """Reference retriever demonstrating tracer integration.

    In production, this pattern goes directly into DeepChoice's
    ``src/deepchoice/retrievers/base.py`` so every subclass automatically
    gets observability without changing its own search logic.
    """

    source: str = "base"

    def __init__(self, tracer: Tracer | None = None):
        self.tracer = tracer

    async def search(self, query: str, sub_questions: list[str],
                     max_results: int = 7,
                     adapted_queries: list[str] | None = None) -> dict:
        """Public entry-point.  Starts a trace, delegates to _do_search,
        and reports success or failure through the tracer before returning.
        """
        t0 = time.monotonic()

        # --- integration block (added) ---
        report: TraceReport | None = None
        if self.tracer:
            report = self.tracer.start_trace(
                agent_id=self.__class__.__name__,
                tool_name=self.__class__.__name__,
                params={
                    "query": query,
                    "max_results": max_results,
                    "adapted_queries": adapted_queries,
                },
            )

        try:
            result = await self._do_search(query, sub_questions, max_results,
                                           adapted_queries=adapted_queries or [])
            # --- integration block (added) ---
            if self.tracer and report:
                report.success = True
                report.latency_ms = int((time.monotonic() - t0) * 1000)
                report.result = {
                    "result_count": len(result),
                    "source": self.source,
                }
                await self.tracer.report(report)

            return {
                "source": self.source,
                "status": "success",
                "results": result,
                "error": None,
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }
        except Exception as exc:
            # --- integration block (added) ---
            if self.tracer and report:
                report.success = False
                report.latency_ms = int((time.monotonic() - t0) * 1000)
                report.error_type = _classify_error(exc)
                report.error_message = str(exc)
                await self.tracer.report(report)

            return {
                "source": self.source,
                "status": "failed",
                "results": [],
                "error": str(exc),
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }

    async def _do_search(self, query: str, sub_questions: list[str],
                         max_results: int,
                         adapted_queries: list[str] | None = None) -> list[dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Demo retriever: succeed or fail so we can observe both paths
# ---------------------------------------------------------------------------

class DemoRetriever(TracedBaseRetriever):
    """A concrete retriever for demonstration purposes."""

    source = "demo"

    async def _do_search(self, query: str, sub_questions: list[str],
                         max_results: int,
                         adapted_queries: list[str] | None = None) -> list[dict]:
        # Simulate a failure for the special sentinel query.
        if "RAISE_TIMEOUT" in query:
            raise TimeoutError("demo timeout simulation")
        if "RAISE_RATELIMIT" in query:
            raise RuntimeError("rate limit exceeded")
        if "RAISE_TYPEERROR" in query:
            raise TypeError("unexpected parameter type")

        # Normal path: return mock results.
        await asyncio.sleep(0.01)  # pretend I/O
        return [
            {"title": f"Result {i} for '{query}'", "url": f"https://demo/{i}", "score": 0.95 - i * 0.1}
            for i in range(1, min(max_results + 1, 4))
        ]


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

async def main() -> None:
    conn = await get_connection()
    await init_db(conn)
    tracer = Tracer(conn, batch_size=10)

    retriever = DemoRetriever(tracer=tracer)

    print("=" * 60)
    print("DeepChoice Tracer Integration Demo")
    print("=" * 60)

    # --- Success case ---
    print("\n1. Success path:")
    r = await retriever.search("fastapi best practices", [])
    print(f"   status={r['status']}, results={len(r['results'])}, latency={r['latency_ms']}ms")

    # --- Failure cases ---
    print("\n2. Timeout failure:")
    r = await retriever.search("RAISE_TIMEOUT benchmark", [])
    print(f"   status={r['status']}, error={r['error']}, latency={r['latency_ms']}ms")

    print("\n3. Rate-limit failure:")
    r = await retriever.search("RAISE_RATELIMIT something", [])
    print(f"   status={r['status']}, error={r['error']}")

    print("\n4. Parameter error:")
    r = await retriever.search("RAISE_TYPEERROR bad arg", [])
    print(f"   status={r['status']}, error={r['error']}")

    # Flush all buffered traces to storage.
    await tracer.flush()

    # Report what landed in the database.
    store = tracer.store
    all_traces = await store.get_by_tool(DemoRetriever.__name__, limit=20)
    success_count = sum(1 for t in all_traces if t['success'])
    fail_count = len(all_traces) - success_count

    print("\n" + "-" * 60)
    print(f"Traces in store: {len(all_traces)} total ({success_count} success, {fail_count} failure)")

    for t in all_traces:
        print(f"  [{t['trace_id'][:8]}..] success={bool(t['success'])} "
              f"latency={t['latency_ms']}ms "
              f"error_type={t['error_type'] or '-'}")

    # --- Without tracer (graceful no-op) ---
    print("\n5. Without tracer (noop):")
    untraced = DemoRetriever(tracer=None)
    r = await untraced.search("plain query", [])
    print(f"   status={r['status']}, results={len(r['results'])} — no trace emitted")

    await tracer.close()
    await conn.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
