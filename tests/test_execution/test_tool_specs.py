import importlib.util
import sys
from pathlib import Path

from tool_evolution.execution.tool_specs import (
    TOOL_SPECS,
    describe_tools_for_llm,
    validate_params,
)

_RUN_EVAL = Path(__file__).resolve().parents[2] / "scripts" / "run_eval.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval_tools_mod", _RUN_EVAL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestToolSpecsCoverage:
    def test_matches_run_eval_tools(self):
        run_eval = _load_run_eval()

        assert set(TOOL_SPECS.keys()) == set(run_eval.EVAL_TOOLS)

    def test_all_specs_have_required_fields(self):
        for name, spec in TOOL_SPECS.items():
            assert "params" in spec, name
            assert "mock_result_keys" in spec, name
            assert "mock_token_cost" in spec, name
            assert "mock_latency_ms" in spec, name
            assert isinstance(spec["mock_token_cost"], int), name
            assert isinstance(spec["mock_latency_ms"], int), name


class TestValidateParams:
    def test_valid_params_pass(self):
        assert validate_params("search_api", {"query": "hello", "max_results": 5}) is None

    def test_missing_required_rejected(self):
        reason = validate_params("search_api", {"max_results": 5})
        assert reason is not None
        assert "query" in reason

    def test_unknown_tool_rejected(self):
        reason = validate_params("nonexistent_tool", {"query": "x"})
        assert reason is not None

    def test_wrong_type_rejected(self):
        reason = validate_params("search_api", {"query": "x", "max_results": "many"})
        assert reason is not None


class TestDescribeToolsForLlm:
    def test_contains_all_tools_and_params(self):
        text = describe_tools_for_llm()
        for name in TOOL_SPECS:
            assert name in text
        assert "query" in text
        assert "max_results" in text
