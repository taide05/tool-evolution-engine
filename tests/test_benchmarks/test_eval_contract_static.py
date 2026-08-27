import ast
from pathlib import Path

RUN_EVAL = Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py"
SEED_DEMO = Path(__file__).resolve().parents[2] / "scripts" / "seed_demo_data.py"

GSM_KEYS = ["schema_version", "timestamp", "seed", "benchmark", "failure_reduction",
            "dag_recall", "classifier", "template_coverage", "rule_precision",
            "governance", "throughput", "data_composition"]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


class TestEvalContractStatic:
    def test_build_gsm_metrics_defined(self):
        tree = _parse(RUN_EVAL)
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "build_gsm_metrics" in funcs

    def test_gsm_keys_literal(self):
        tree = _parse(RUN_EVAL)
        found = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                found.append(n.value)
        for key in GSM_KEYS:
            assert key in found, f"eval_results 契约字段 {key} 未出现在 run_eval.py"

    def test_source_labeled_synthetic_demo(self):
        for path in (RUN_EVAL, SEED_DEMO):
            tree = _parse(path)
            sources = {n.value.value for n in ast.walk(tree)
                       if isinstance(n, ast.keyword) and n.arg == "source"
                       and isinstance(n.value, ast.Constant)}
            assert "synthetic_demo" in sources, f"{path.name} 缺少 source='synthetic_demo' 标注"
            assert "canary_measurement" not in sources, "synthetic 脚本不得标注真实来源"

    def test_run_migrations_wired(self):
        for path in (RUN_EVAL, SEED_DEMO):
            tree = _parse(path)
            called = {n.func.id for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            assert "run_migrations" in called, f"{path.name} 未接线 run_migrations"

    def test_eval_repair_advisor_defined(self):
        # 注意：eval_repair_advisor 是 async def → AsyncFunctionDef 节点
        tree = _parse(RUN_EVAL)
        funcs = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "eval_repair_advisor" in funcs
        assert "_seed_repair_cases" in funcs

    def test_gsm_has_repair_key(self):
        tree = _parse(RUN_EVAL)
        found = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert "repair_advisor" in found
