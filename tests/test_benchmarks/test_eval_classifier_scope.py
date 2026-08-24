import importlib.util
import sys
from pathlib import Path

from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport, TraceType, ErrorType

RUN_EVAL = Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval_scope_mod", RUN_EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestEvalClassifierScope:
    async def test_scopes_to_eval_prefix(self, db_conn):
        # I#1: eval_classifier 只吃 eval-* 失败轨迹，非 eval 前缀失败不得进入训练/测试
        run_eval = _load_run_eval()
        ts = TraceStore(db_conn)
        for i in range(30):
            await ts.insert(TraceReport(
                trace_id=f"eval-x-{i}", agent_id="a", tool_name="t",
                trace_type=TraceType.ATOMIC, success=False, latency_ms=10,
                params={"query": "q"}, error_type=ErrorType.TIMEOUT,
                error_message=f"connection to api.example.com timed out after {i}s",
            ))
        for i in range(10):
            await ts.insert(TraceReport(
                trace_id=f"other-{i}", agent_id="a", tool_name="t",
                trace_type=TraceType.ATOMIC, success=False, latency_ms=10,
                params={"query": "q"}, error_type=ErrorType.PARAM_ERROR,
                error_message="missing required parameter 'query'",
            ))
        result = await run_eval.eval_classifier(db_conn)
        assert result["train_size"] + result["test_size"] == 30
        assert set(result["per_class"].keys()) == {"timeout"}
