import json
from pathlib import Path

from scripts.run_eval import (build_gsm_metrics, degradation_sizes,
                               eval_degradation_curve, expand_benchmark_tasks,
                               parse_args)

_TASKS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_tasks.json"

_MAX_RESULTS_VARIANTS = [5, 8, 10, 12, 15, 18, 20, 25]
_LANG_VARIANTS = ["zh", "zh", "zh", "zh", "en", "ja", "zh", "zh"]


def _base_tasks() -> list[dict]:
    return json.loads(_TASKS_PATH.read_text(encoding="utf-8"))


class TestExpandBenchmarkTasks:
    def test_expand_8_matches_legacy_shape(self):
        expanded = expand_benchmark_tasks(_base_tasks(), 8)
        assert len(expanded) == len(_base_tasks()) * 8
        assert expanded[0]["task_id"] == "bench-001-v0"
        assert expanded[7]["task_id"] == "bench-001-v7"
        assert expanded[8]["task_id"] == "bench-002-v0"

    def test_expand_40_gives_2000(self):
        expanded = expand_benchmark_tasks(_base_tasks(), 40)
        assert len(expanded) == len(_base_tasks()) * 40
        assert expanded[0]["task_id"].endswith("-v0")
        assert expanded[-1]["task_id"].endswith("-v39")

    def test_variant_rotation_values(self):
        expanded = expand_benchmark_tasks(_base_tasks(), 40)
        for j in range(8):
            assert expanded[j]["root_params"]["max_results"] == _MAX_RESULTS_VARIANTS[j]
            assert expanded[j]["root_params"]["lang"] == _LANG_VARIANTS[j]
        assert expanded[8]["root_params"]["max_results"] == _MAX_RESULTS_VARIANTS[0]
        assert expanded[8]["root_params"]["lang"] == _LANG_VARIANTS[0]

    def test_expand_deterministic(self):
        assert expand_benchmark_tasks(_base_tasks(), 40) == expand_benchmark_tasks(
            _base_tasks(), 40)

    def test_expand_does_not_mutate_input(self):
        base = _base_tasks()
        before = dict(base[0]["root_params"])
        expand_benchmark_tasks(base, 40)
        assert base[0]["root_params"] == before

    def test_expand_num_variants_one(self):
        expanded = expand_benchmark_tasks(_base_tasks(), 1)
        assert len(expanded) == len(_base_tasks())
        assert expanded[0]["task_id"] == "bench-001-v0"


class TestGsmThreading:
    def test_gsm_seed_and_benchmark_keys(self):
        cls = {"accuracy": 0.9, "macro_f1": 0.9, "per_class_f1": {}}
        kde = {"tools_analyzed": 7, "total_params": 20}
        dag = {"planted_patterns": ["a"], "matched": ["a"]}
        gov = {"skills_scored": 3, "skills": [{"status_after_update": "active"}],
               "ab_test": {"rollback": False}}
        ba = {"baseline": {"failures": 10, "total_calls": 100},
              "optimized": {"failures": 2, "total_calls": 100}}
        gsm = build_gsm_metrics(cls, kde, dag, gov, ba, 12.5,
                                {"synthetic_demo": 100}, 5,
                                seed_tasks=2000, benchmark_tasks=2000)
        assert gsm["seed"]["n_tasks"] == 2000
        assert gsm["benchmark"]["n_tasks"] == 2000


class TestArgparse:
    def test_parse_defaults(self):
        args = parse_args([])
        assert args.seed == 2000
        assert args.num_variants == 40
        assert args.output == "eval_results.json"

    def test_parse_custom(self):
        args = parse_args(["--seed", "400", "--num-variants", "8"])
        assert args.seed == 400
        assert args.num_variants == 8

    def test_parse_num_variants_zero_rejected(self):
        import pytest
        with pytest.raises(SystemExit):
            parse_args(["--num-variants", "0"])


class TestSaveResults:
    def test_save_results_writes_json(self, tmp_path):
        from scripts.run_execution_eval import save_results
        p = tmp_path / "out.json"
        save_results({"n_tasks": 50, "中文键": "值"}, p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data == {"n_tasks": 50, "中文键": "值"}


class TestDegradationSizes:
    def test_degradation_sizes_derived(self):
        assert degradation_sizes(2000) == [("small", 500), ("medium", 1000),
                                           ("large", 2000)]
        assert degradation_sizes(400) == [("small", 100), ("medium", 200),
                                          ("large", 400)]

    async def test_degradation_runs_small_scale(self, db_conn):
        result = await eval_degradation_curve(db_conn, n_tasks=400)
        assert set(result.keys()) == {"small", "medium", "large"}
        for scale in result.values():
            assert scale["n_traces"] > 0
            # > 0 防 error 分支 0 兜底（eval_classifier 失败样本 <20 时返回 error，
            # get("accuracy", 0) 会兜底 0——测试不能把兜底当通过）
            assert scale["classifier_accuracy"] > 0
            for key in ("classifier_macro_f1", "dag_pattern_recall",
                        "dag_discovered"):
                assert key in scale


class TestExecutionEvalNumTasks:
    def test_signature_accepts_n_tasks(self):
        import inspect
        from scripts.run_execution_eval import run_execution_eval
        sig = inspect.signature(run_execution_eval)
        assert "n_tasks" in sig.parameters
        assert sig.parameters["n_tasks"].default == 2000

    async def test_not_divisible_raises(self, db_conn):
        import pytest
        from scripts.run_execution_eval import run_execution_eval
        with pytest.raises(ValueError, match="divisible"):
            await run_execution_eval(db_conn, n_tasks=123)

    def test_shared_expand_consistency(self):
        from scripts.run_execution_eval import load_eval_tasks
        tasks = load_eval_tasks(len(_base_tasks()) * 40)
        assert len(tasks) == len(_base_tasks()) * 40
        expected = {t["task_id"] for t in expand_benchmark_tasks(_base_tasks(), 40)}
        assert {t["task_id"] for t in tasks} == expected

    def test_n_tasks_50_single_variant(self):
        from scripts.run_execution_eval import load_eval_tasks
        tasks = load_eval_tasks(len(_base_tasks()))
        assert len(tasks) == len(_base_tasks())
        assert all(t["task_id"].endswith("-v0") for t in tasks)


class TestLlmPlanPassIsolation:
    async def test_planner_exception_counts_failed(self, db_conn, monkeypatch):
        from tool_evolution.execution.adapters import MockAdapter
        from tool_evolution.execution.audit import ExecutionAudit
        from tool_evolution.execution.executor import SkillExecutor
        from tool_evolution.execution.planner import LLMPlanner
        from tool_evolution.utils.config import settings
        from scripts.run_execution_eval import _run_llm_plan_pass

        monkeypatch.setattr(settings, "deepseek_api_key", "fake-key")

        async def _boom(self, desc):
            raise RuntimeError("simulated planner failure")

        monkeypatch.setattr(LLMPlanner, "plan", _boom)
        audit = ExecutionAudit(db_conn)
        adapter = MockAdapter()
        executor = SkillExecutor(db_conn, adapter, audit=audit)
        try:
            tasks = [{"task_id": "t1", "tool_chain": ["search_api"], "task_name": "x"},
                     {"task_id": "t2", "tool_chain": ["search_api"], "task_name": "y"}]
            result = await _run_llm_plan_pass(db_conn, tasks, ["t1", "t2"],
                                              executor, audit)
            assert result["mode"] == "live"
            assert result["success"] == 0
            assert result["failed"] == 2
            assert set(result["failed_task_ids"]) == {"t1", "t2"}
        finally:
            await adapter.close()

    async def test_planner_none_counts_failed(self, db_conn, monkeypatch):
        from tool_evolution.execution.adapters import MockAdapter
        from tool_evolution.execution.audit import ExecutionAudit
        from tool_evolution.execution.executor import SkillExecutor
        from tool_evolution.execution.planner import LLMPlanner
        from tool_evolution.utils.config import settings
        from scripts.run_execution_eval import _run_llm_plan_pass

        monkeypatch.setattr(settings, "deepseek_api_key", "fake-key")

        async def _none(self, desc):
            return None

        monkeypatch.setattr(LLMPlanner, "plan", _none)
        audit = ExecutionAudit(db_conn)
        adapter = MockAdapter()
        executor = SkillExecutor(db_conn, adapter, audit=audit)
        try:
            tasks = [{"task_id": "t1", "tool_chain": ["search_api"], "task_name": "x"}]
            result = await _run_llm_plan_pass(db_conn, tasks, ["t1"],
                                              executor, audit)
            assert result["mode"] == "live"
            assert result["success"] == 0
            assert result["failed"] == 1
            assert result["failed_task_ids"] == ["t1"]
            assert result["failed_reasons"] == ["plan_rejected"]
        finally:
            await adapter.close()
