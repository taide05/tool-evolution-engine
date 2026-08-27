import json
from tool_evolution.analysis.distiller import CounterfactualDistiller


class TestCounterfactualDistiller:
    def test_distill_param_error_generates_range_rule(self):
        d = CounterfactualDistiller()
        trace = {
            "error_type": "param_error",
            "tool_name": "search", "tool_version": "1.0.0",
            "params": json.dumps({"max_results": 999}),
            "error_message": "max_results must be between 1 and 100"
        }
        rule = d.distill(trace)
        assert rule["rule_type"] == "range_rule"
        assert rule["tool_name"] == "search"
        assert rule["tool_version"] == "1.0.0"
        assert "max_results" in str(rule["condition"])

    def test_distill_permission_denied_generates_auth_rule(self):
        d = CounterfactualDistiller()
        trace = {
            "error_type": "permission_denied",
            "tool_name": "github_api", "tool_version": "1.0.0",
            "params": json.dumps({}),
            "error_message": "401 Unauthorized - missing API token"
        }
        rule = d.distill(trace)
        assert rule["rule_type"] == "auth_rule"
        assert "token" in str(rule["action"]).lower()

    def test_distill_quota_exhausted_generates_retry_rule(self):
        d = CounterfactualDistiller()
        trace = {
            "error_type": "quota_exhausted",
            "tool_name": "github_api", "tool_version": "1.0.0",
            "params": json.dumps({}),
            "error_message": "API rate limit exceeded"
        }
        rule = d.distill(trace)
        assert rule["rule_type"] == "retry_rule"
        action = rule["action"]
        assert action["delay_seconds"] == 60
        assert action["max_retries"] == 3

    def test_distill_timeout_generates_timeout_rule(self):
        d = CounterfactualDistiller()
        trace = {
            "error_type": "timeout",
            "tool_name": "arxiv_api", "tool_version": "2.0.0",
            "params": json.dumps({"timeout_ms": 30000}),
            "error_message": "Request timed out after 30s"
        }
        rule = d.distill(trace)
        assert rule["rule_type"] == "timeout_rule"
        assert rule["action"]["max_wait_ms"] == 20000

    def test_distill_service_unavailable_generates_circuit_breaker(self):
        d = CounterfactualDistiller()
        trace = {
            "error_type": "service_unavailable",
            "tool_name": "official_docs", "tool_version": "1.0.0",
            "params": json.dumps({}),
            "error_message": "503 Service Unavailable"
        }
        rule = d.distill(trace)
        assert rule["rule_type"] == "circuit_breaker_rule"
        assert rule["action"]["failure_threshold"] == 3
        assert rule["action"]["cooldown_seconds"] == 30
