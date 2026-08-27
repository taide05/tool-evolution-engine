import json
from pathlib import Path

from scripts.run_eval import build_gsm_metrics, expand_benchmark_tasks, parse_args

_TASKS_PATH = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_tasks.json"

_MAX_RESULTS_VARIANTS = [5, 8, 10, 12, 15, 18, 20, 25]
_LANG_VARIANTS = ["zh", "zh", "zh", "zh", "en", "ja", "zh", "zh"]


def _base_tasks() -> list[dict]:
    return json.loads(_TASKS_PATH.read_text(encoding="utf-8"))


class TestExpandBenchmarkTasks:
    def test_expand_8_matches_legacy_shape(self):
        expanded = expand_benchmark_tasks(_base_tasks(), 8)
        assert len(expanded) == 400
        assert expanded[0]["task_id"] == "bench-001-v0"
        assert expanded[7]["task_id"] == "bench-001-v7"
        assert expanded[8]["task_id"] == "bench-002-v0"

    def test_expand_40_gives_2000(self):
        expanded = expand_benchmark_tasks(_base_tasks(), 40)
        assert len(expanded) == 2000
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
        assert len(expanded) == 50
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
