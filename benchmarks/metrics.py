"""Tool Evolution Engine Benchmark Metrics Calculator.

GSM Framework: Goals -> Signals -> Metrics

Goal 1: Reduces tool call failures
  Signal 1a: Failure rate drops after optimization → Failure Reduction Rate
  Signal 1b: Retries decrease → Retry Reduction Rate

Goal 2: Discovers reusable skill patterns
  Signal 2a: Planted DAG patterns are found → DAG Pattern Recall
  Signal 2b: Parameter templates are generated → Template Coverage

Goal 3: Analysis pipeline is accurate
  Signal 3a: Classifier identifies error types → Classifier F1
  Signal 3b: Rules distill correctly → Rule Precision

Goal 4: Safely deploys optimizations
  Signal 4a: Canary promotes correctly → Promotion Accuracy
  Signal 4b: A/B rollback triggers correctly → Rollback Accuracy
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Quality Metric 1: Failure Reduction Rate
# ---------------------------------------------------------------------------

def compute_failure_reduction(
    baseline_failure_rate: float,
    optimized_failure_rate: float,
    baseline_total: int = 0,
    optimized_total: int = 0,
) -> dict[str, Any]:
    """Compute failure rate reduction from baseline to optimized."""
    if baseline_failure_rate == 0:
        return {"metric": "failure_reduction", "value": 0.0,
                "note": "No baseline failures to reduce"}

    absolute_reduction = baseline_failure_rate - optimized_failure_rate
    relative_reduction = absolute_reduction / baseline_failure_rate

    return {
        "metric": "failure_reduction",
        "absolute_reduction_pct": round(absolute_reduction * 100, 1),
        "relative_reduction_pct": round(relative_reduction * 100, 1),
        "baseline_rate": round(baseline_failure_rate, 3),
        "optimized_rate": round(optimized_failure_rate, 3),
        "baseline_total": baseline_total,
        "optimized_total": optimized_total,
    }


# ---------------------------------------------------------------------------
# Quality Metric 2: DAG Pattern Recall
# ---------------------------------------------------------------------------

def compute_dag_pattern_recall(
    planted_patterns: int,
    discovered_patterns: int,
    pattern_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute DAG pattern mining recall."""
    if planted_patterns == 0:
        return {"metric": "dag_pattern_recall", "value": 0.0, "note": "No patterns planted"}

    recall = discovered_patterns / planted_patterns

    return {
        "metric": "dag_pattern_recall",
        "value": round(recall, 3),
        "planted": planted_patterns,
        "discovered": discovered_patterns,
        "names": pattern_names or [],
    }


# ---------------------------------------------------------------------------
# Quality Metric 3: Classifier F1
# ---------------------------------------------------------------------------

def compute_classifier_metrics(
    accuracy: float,
    macro_f1: float,
    per_class_f1: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute classifier quality metrics."""
    return {
        "metric": "classifier_f1",
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class_f1": per_class_f1 or {},
    }


# ---------------------------------------------------------------------------
# Quality Metric 4: Parameter Template Coverage
# ---------------------------------------------------------------------------

def compute_template_coverage(
    tools_total: int,
    tools_with_templates: int,
    params_discovered: int,
) -> dict[str, Any]:
    """Compute KDE parameter template coverage."""
    if tools_total == 0:
        return {"metric": "template_coverage", "value": 0.0, "note": "No tools analyzed"}

    coverage = tools_with_templates / tools_total

    return {
        "metric": "template_coverage",
        "value": round(coverage, 3),
        "tools_total": tools_total,
        "tools_with_templates": tools_with_templates,
        "params_discovered": params_discovered,
    }


# ---------------------------------------------------------------------------
# Efficiency Metric 5: Throughput (traces/second)
# ---------------------------------------------------------------------------

def compute_throughput(total_traces: int, elapsed_s: float) -> dict[str, Any]:
    """Compute trace ingestion throughput."""
    if elapsed_s <= 0:
        return {"metric": "throughput", "traces_per_second": 0, "total_traces": total_traces}
    return {
        "metric": "throughput",
        "traces_per_second": round(total_traces / elapsed_s, 1),
        "total_traces": total_traces,
        "elapsed_s": round(elapsed_s, 1),
    }


# ---------------------------------------------------------------------------
# Reliability Metric 6: Rule Precision
# ---------------------------------------------------------------------------

def compute_rule_precision(
    rules_generated: int,
    rules_valid: int,
    deduplicated: int = 0,
) -> dict[str, Any]:
    """Compute rule generation precision (valid vs generated)."""
    if rules_generated == 0:
        return {"metric": "rule_precision", "value": 0.0, "note": "No rules generated"}

    precision = rules_valid / rules_generated

    return {
        "metric": "rule_precision",
        "value": round(precision, 3),
        "rules_generated": rules_generated,
        "rules_valid": rules_valid,
        "deduplicated": deduplicated,
    }


# ---------------------------------------------------------------------------
# Governance Metric 7: Promotion Accuracy
# ---------------------------------------------------------------------------

def compute_promotion_accuracy(
    canary_total: int,
    promoted: int,
    demoted: int,
    rolled_back: int,
) -> dict[str, Any]:
    """Compute canary promotion/demotion/rollback stats."""
    return {
        "metric": "governance_actions",
        "canary_total": canary_total,
        "promoted_to_active": promoted,
        "demoted_to_offline": demoted,
        "a_b_rolled_back": rolled_back,
    }


# ---------------------------------------------------------------------------
# Aggregate Report
# ---------------------------------------------------------------------------

def compute_all_metrics(
    eval_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute all TEE metrics from eval data dict.

    Args:
        eval_data: Dict with keys matching the individual metric functions.
            Expected keys:
            - failure_reduction: {baseline_rate, optimized_rate, ...}
            - dag_recall: {planted, discovered, names}
            - classifier: {accuracy, macro_f1, per_class_f1}
            - template_coverage: {tools_total, with_templates, params}
            - rule_precision: {generated, valid, deduplicated}
            - governance: {canary_total, promoted, demoted, rolled_back}
            - throughput: {traces, elapsed_s}
    """
    eval_data = eval_data or {}
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {},
        "quality": {},
        "efficiency": {},
        "reliability": {},
    }

    # Quality
    if "failure_reduction" in eval_data:
        fd = eval_data["failure_reduction"]
        report["quality"]["failure_reduction"] = compute_failure_reduction(
            fd.get("baseline_rate", 0), fd.get("optimized_rate", 0),
            fd.get("baseline_total", 0), fd.get("optimized_total", 0),
        )
    if "dag_recall" in eval_data:
        dr = eval_data["dag_recall"]
        report["quality"]["dag_pattern_recall"] = compute_dag_pattern_recall(
            dr.get("planted", 0), dr.get("discovered", 0), dr.get("names"),
        )
    if "classifier" in eval_data:
        cf = eval_data["classifier"]
        report["quality"]["classifier"] = compute_classifier_metrics(
            cf.get("accuracy", 0), cf.get("macro_f1", 0), cf.get("per_class_f1"),
        )
    if "template_coverage" in eval_data:
        tc = eval_data["template_coverage"]
        report["quality"]["template_coverage"] = compute_template_coverage(
            tc.get("tools_total", 0), tc.get("with_templates", 0), tc.get("params", 0),
        )

    # Reliability
    if "rule_precision" in eval_data:
        rp = eval_data["rule_precision"]
        report["reliability"]["rule_precision"] = compute_rule_precision(
            rp.get("generated", 0), rp.get("valid", 0), rp.get("deduplicated", 0),
        )

    # Efficiency
    if "throughput" in eval_data:
        tp = eval_data["throughput"]
        report["efficiency"]["throughput"] = compute_throughput(
            tp.get("traces", 0), tp.get("elapsed_s", 0),
        )

    # Governance
    if "governance" in eval_data:
        gv = eval_data["governance"]
        report["reliability"]["governance"] = compute_promotion_accuracy(
            gv.get("canary_total", 0), gv.get("promoted", 0),
            gv.get("demoted", 0), gv.get("rolled_back", 0),
        )

    if "data_composition" in eval_data:
        comp = eval_data["data_composition"]
        total = sum(comp.values())
        report["data_composition"] = {
            "sources": comp,
            "total": total,
            "pct": {k: round(v / max(total, 1), 4) for k, v in comp.items()},
        }
    if "schema_version" in eval_data:
        report["schema_version"] = eval_data["schema_version"]

    # Summary
    s: dict[str, Any] = {}
    q = report["quality"]
    r = report["reliability"]

    if "failure_reduction" in q:
        s["failure_reduction_pct"] = q["failure_reduction"]["relative_reduction_pct"]
    if "dag_pattern_recall" in q:
        s["dag_pattern_recall"] = q["dag_pattern_recall"]["value"]
    if "classifier" in q:
        s["classifier_macro_f1"] = q["classifier"]["macro_f1"]
    if "template_coverage" in q:
        s["template_coverage"] = q["template_coverage"]["value"]
    if "rule_precision" in r:
        s["rule_precision"] = r["rule_precision"]["value"]
    if "governance" in r:
        s["promoted_skills"] = r["governance"]["promoted_to_active"]

    report["summary"] = s
    return report


def save_benchmark(report: dict[str, Any], runs_dir: Path, label: str = "") -> Path:
    """Save benchmark report to timestamped JSON."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = runs_dir / f"benchmark{label}-{ts}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_benchmarks(runs_dir: Path) -> list[dict[str, Any]]:
    """Load all benchmark reports sorted by time."""
    reports = []
    for f in sorted(runs_dir.glob("benchmark*.json")):
        try:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return reports
