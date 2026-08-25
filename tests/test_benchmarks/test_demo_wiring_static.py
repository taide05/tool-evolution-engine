import ast
from pathlib import Path

RUN_DEMO = Path(__file__).resolve().parents[2] / "scripts" / "run_demo.py"


class TestDemoRepairWiring:
    def test_demo_imports_repair_advisor(self):
        tree = ast.parse(RUN_DEMO.read_text(encoding="utf-8"))
        assert "RepairAdvisor" in {a.name for n in ast.walk(tree)
                                  if isinstance(n, ast.ImportFrom)
                                  for a in n.names}
        assert any(n.module == "tool_evolution.analysis.repair_advisor"
                   for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))

    def test_demo_generates_hints_after_rules(self):
        tree = ast.parse(RUN_DEMO.read_text(encoding="utf-8"))
        calls = [n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        assert "add_rule" in calls and "generate_for_rules" in calls
