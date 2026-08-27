import json
import hashlib


class CounterfactualDistiller:
    """Converts failed traces into fix rules via 5 error-type → rule-type mappings.

    Batch deduplication uses MD5 hash of (rule_type, tool_name, tool_version).
    Pure function -- no dependencies on other project modules.
    """

    def distill(self, failed_trace: dict) -> dict:
        error_type = failed_trace["error_type"]
        tool_name = failed_trace["tool_name"]
        tool_version = failed_trace.get("tool_version", "1.0.0")
        params = json.loads(failed_trace.get("params", "{}"))
        error_msg = failed_trace.get("error_message", "")

        rule = {
            "tool_name": tool_name,
            "tool_version": tool_version,
            "status": "active",
            "condition": {},
            "action": {},
            "rule_type": "",
        }

        if error_type == "param_error":
            rule["rule_type"] = "range_rule"
            rule["condition"] = {"param_names": list(params.keys())}
            rule["action"] = {"validate_before_call": True, "error_msg_template": error_msg}
        elif error_type == "permission_denied":
            rule["rule_type"] = "auth_rule"
            rule["condition"] = {"requires_auth": True}
            rule["action"] = {"check_token_before_call": True, "error_msg_template": error_msg}
        elif error_type == "quota_exhausted":
            rule["rule_type"] = "retry_rule"
            rule["condition"] = {"on_error": "quota_exhausted"}
            rule["action"] = {"delay_seconds": 60, "max_retries": 3}
        elif error_type == "timeout":
            rule["rule_type"] = "timeout_rule"
            rule["condition"] = {"on_error": "timeout"}
            rule["action"] = {"max_wait_ms": 20000}
        elif error_type == "service_unavailable":
            rule["rule_type"] = "circuit_breaker_rule"
            rule["condition"] = {"on_error": "service_unavailable"}
            rule["action"] = {"failure_threshold": 3, "cooldown_seconds": 30}

        rule["_hash"] = hashlib.md5(
            f"{rule['rule_type']}:{tool_name}:{tool_version}".encode()
        ).hexdigest()[:12]
        return rule
