import json
import pytest
from pathlib import Path
from benchmarks.run_baseline import load_eval_results, EvalResultError
from benchmarks.metrics import compute_all_metrics

VALID = {
    "schema_version": 1,
    "timestamp": "2026-08-23T00:00:00+00:00",
    "seed": {"n_tasks": 1000},
    "failure_reduction": {"baseline_rate": 0.2, "optimized_rate": 0.08,
                          "baseline_total": 100, "optimized_total": 100},
    "dag_recall": {"planted": 8, "discovered": 8, "names": ["a"]},
    "classifier": {"accuracy": 0.9, "macro_f1": 0.9, "per_class_f1": {}},
    "template_coverage": {"tools_total": 7, "with_templates": 7, "params": 20},
    "rule_precision": {"generated": 5, "valid": 5, "deduplicated": 0},
    "governance": {"canary_total": 3, "promoted": 2, "demoted": 0, "rolled_back": 1},
    "throughput": {"traces": 1000, "elapsed_s": 430.0},
    "data_composition": {"synthetic_demo": 1000},
}


class TestLoadEvalResults:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(EvalResultError):
            load_eval_results(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(EvalResultError):
            load_eval_results(p)

    def test_missing_field_raises(self, tmp_path):
        data = dict(VALID)
        del data["governance"]
        p = tmp_path / "partial.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(EvalResultError):
            load_eval_results(p)

    def test_valid_loads(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text(json.dumps(VALID), encoding="utf-8")
        assert load_eval_results(p)["schema_version"] == 1


class TestCompositionPassthrough:
    def test_report_includes_composition(self):
        report = compute_all_metrics(VALID)
        assert report["data_composition"]["sources"] == {"synthetic_demo": 1000}
        assert report["data_composition"]["total"] == 1000
        assert report["data_composition"]["pct"]["synthetic_demo"] == 1.0
        assert report["schema_version"] == 1


def test_run_baseline_timeout_1800():
    import ast
    tree = ast.parse(Path("benchmarks/run_baseline.py").read_text(encoding="utf-8"))
    timeouts = [n.value.value for n in ast.walk(tree)
                if isinstance(n, ast.keyword) and n.arg == "timeout"
                and isinstance(n.value, ast.Constant)]
    assert timeouts and max(timeouts) >= 1800
