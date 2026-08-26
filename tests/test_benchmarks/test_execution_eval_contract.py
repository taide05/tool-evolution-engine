import importlib.util
import json
import sys
from pathlib import Path

from tool_evolution.execution.tool_specs import TOOL_SPECS

_EVAL_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_execution_eval.py"
_TASKS = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_tasks.json"

_EXPECTED_KEYS = {
    "n_tasks",
    "skill_match",
    "skill_plan",
    "llm_plan",
    "comparison",
    "repair_replay",
    "executor_isolation",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("run_execution_eval_mod", _EVAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestExecutionEvalContract:
    def test_script_imports_cleanly(self):
        mod = _load_script()
        assert callable(mod.run_execution_eval)

    def test_benchmark_tasks_tools_subset(self):
        tasks = json.loads(_TASKS.read_text(encoding="utf-8"))
        assert len(tasks) == 50
        for task in tasks:
            for tool in task["tool_chain"]:
                assert tool in TOOL_SPECS, f"{tool} not in TOOL_SPECS"

    async def test_eval_returns_all_keys(self, db_conn):
        mod = _load_script()
        result = await mod.run_execution_eval(db_conn)
        assert _EXPECTED_KEYS <= set(result.keys())
