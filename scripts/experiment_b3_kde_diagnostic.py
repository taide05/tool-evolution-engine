"""B3 diagnostic: are KDE-derived parameter defaults better than simple statistics?

Scope: **diagnosis only**, no production change, no headline change.

Held-out protocol (same LOMO/cross-model split as the headline):
  for every model family ``f`` in the five recorded families,
      train successes = recorded successes of the fixed-three training models that are
                        not ``f`` (and with ``valid_data``),
      test successes  = recorded successes of ``f``.
  A ``(tool, parameter)`` is analysed only when the *training* side has at least 30
  values carrying that parameter (the old ``KDEAnalyzer.min_samples``).  Every metric is
  computed on the held-out family's values -- never in-sample.

Methods
  * numeric parameters: ``KDEAnalyzer``-style Gaussian KDE peak (the "KDE mode") vs
    mean vs median; MAE / RMSE against the held-out values, plus 95% interval coverage
    of the training KDE interval [p2.5, p97.5] and of mean +/- 2 sd;
  * string parameters: the old ``KDEAnalyzer`` TF-IDF+KMeans dominant-cluster default
    (explicitly labelled ``tfidf_kmeans``, not a true KDE) vs the most-frequent value;
    exact-match rate on the held-out values;
  * boolean parameters: majority vote only (no KDE forced).

Downstream measurement on the 969 evaluable failures: for each failure whose schema
still misses required parameters, three default-completion candidates are compared with
the same schema gate and the evaluator's exact-recorded-success (value-verified) tier:

  * ``schema_default_completion``  -- the current engine rule (schema example/default);
  * ``memory_param_mode``          -- per-(function, parameter) most-frequent value from
                                      training successes (what the memory rule uses);
  * ``kde_default_completion``     -- the held-out-fitted KDE/TF-IDF+KMeans/majority
                                      defaults (rounded to the declared scalar type).

Usage:
    python scripts/experiment_b3_kde_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

# Single-threaded BLAS/OpenMP: the old KDEAnalyzer uses KMeans(random_state=42); with
# multi-threaded reductions the cluster assignment can flip on numeric ties, which made
# the held-out exact-match numbers jitter in the 4th decimal.  Fixed threads + the fixed
# seed keep this diagnostic reproducible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from tool_evolution.analysis.kde_analyzer import KDEAnalyzer  # noqa: E402
from tool_evolution.toolguard.engine import _schema_default_completion, repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error  # noqa: E402
from tool_evolution.toolguard.schema_check import validate_arguments  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

# KMeans on near-duplicate string values legitimately finds fewer distinct clusters than
# requested; that is expected and does not affect the diagnostic, so keep the log clean.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}
_MIN_SAMPLES = 30


# ---------------------------------------------------------------------------
# fitting / prediction helpers
# ---------------------------------------------------------------------------
def _json_type(value) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def _dominant_type(values: list) -> str | None:
    kinds = Counter(_json_type(value) for value in values)
    kinds.pop(None, None)
    if not kinds:
        return None
    return kinds.most_common(1)[0][0]


def _kde_numeric_default(values: list[float]) -> tuple[float | None, float, float]:
    """KDE peak (the old analyzer's approach) + the training interval bounds."""
    array = np.asarray(values, dtype=float)
    lower, upper = float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))
    if array.size < 2 or float(array.std()) == 0.0:
        return float(array.mean()), lower, upper
    try:
        kde = gaussian_kde(array)
        grid = np.linspace(float(array.min()), float(array.max()), 200)
        density = kde(grid)
        return float(grid[int(np.argmax(density))]), lower, upper
    except Exception:  # singular covariance etc.: fall back to the mean, like the analyzer
        return float(array.mean()), lower, upper


def _string_kde_default(values: list[str], tool: str, param: str) -> tuple[str, str]:
    """Old KDEAnalyzer string default: frequency (<=3 unique) or TF-IDF+KMeans."""
    analyzer = KDEAnalyzer(min_samples=_MIN_SAMPLES)
    distribution = analyzer.analyze(tool, "", [{param: value} for value in values])
    result = distribution.get(param) or {}
    return str(result.get("default_value")), str(result.get("method", "frequency_fallback"))


def _coerce_candidate(value, declared: str | None):
    """Round an inferred default to the schema-declared scalar type (or None if not usable)."""
    if declared == "integer":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        return None
    if declared == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)
    if declared == "boolean":
        return bool(value) if isinstance(value, (int, bool)) and not isinstance(value, float) else None
    if declared == "string":
        return str(value) if isinstance(value, str) else None
    return value  # no declared type: keep the observed scalar as-is


# ---------------------------------------------------------------------------
# held-out parameter-default evaluation
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=".scratch/toolguard/tool_calls.jsonl")
    parser.add_argument("--queries", default=(
        ".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/"
        "solvable_queries/test_instruction"))
    parser.add_argument("--output-dir", default=".scratch/toolguard/b3")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(repo_root / args.queries)
    schemas, schema_fidelity = refine_schema_index(schemas, calls)
    families = sorted({call.model for call in calls})
    print(f"calls: {len(calls)} | schemas: {len(schemas)} | families: {families}")

    def is_success(call) -> bool:
        return call.recorded_ok and call.valid_data

    successes_by_model: dict[str, list] = defaultdict(list)
    for call in calls:
        if is_success(call):
            successes_by_model[call.model].append(call)

    # ------------------------------------------------------------------
    # 1. held-out method comparison per (tool, parameter)
    # ------------------------------------------------------------------
    numeric_rows: list[dict] = []
    string_rows: list[dict] = []
    boolean_rows: list[dict] = []
    skipped: Counter[str] = Counter()
    for family in families:
        train_calls = [call for model in sorted(TRAIN_MODELS) if model != family
                       for call in successes_by_model.get(model, [])]
        test_calls = successes_by_model.get(family, [])
        if not train_calls or not test_calls:
            continue
        train_values: dict[tuple[str, str], list] = defaultdict(list)
        for call in train_calls:
            for name, value in call.arguments.items():
                if _json_type(value) is not None:
                    train_values[(call.tool_name, name)].append(value)
        for (tool, param), values in sorted(train_values.items()):
            if len(values) < _MIN_SAMPLES:
                skipped["fewer_than_min_samples"] += 1
                continue
            kind = _dominant_type(values)
            if kind is None:
                skipped["non_scalar"] += 1
                continue
            test_values = [call.arguments[param] for call in test_calls
                           if call.tool_name == tool and param in call.arguments
                           and _json_type(call.arguments[param]) == kind]
            if not test_values:
                skipped["no_held_out_value"] += 1
                continue
            if kind in ("integer", "number"):
                train_numeric = [float(value) for value in values
                                 if _json_type(value) == kind]
                test_numeric = [float(value) for value in test_values]
                kde_default, lower, upper = _kde_numeric_default(train_numeric)
                array = np.asarray(train_numeric, dtype=float)
                mean = float(array.mean())
                median = float(np.median(array))
                mean_lower, mean_upper = mean - 2 * float(array.std()), mean + 2 * float(array.std())
                errors = {
                    "kde": [abs(value - kde_default) for value in test_numeric],
                    "mean": [abs(value - mean) for value in test_numeric],
                    "median": [abs(value - median) for value in test_numeric],
                }
                squared = {
                    "kde": [(value - kde_default) ** 2 for value in test_numeric],
                    "mean": [(value - mean) ** 2 for value in test_numeric],
                    "median": [(value - median) ** 2 for value in test_numeric],
                }
                scale = max(float(np.mean(np.abs(test_numeric))),
                            float(np.mean(np.abs(train_numeric))), 1e-9)
                mae_values = {method: float(np.mean(value)) for method, value in errors.items()}
                strict_best = [method for method, value in mae_values.items()
                               if value == min(mae_values.values())]
                numeric_rows.append({
                    "fold": family,
                    "tool": tool,
                    "param": param,
                    "param_type": kind,
                    "train_samples": len(values),
                    "test_samples": len(test_values),
                    "test_scale": round(scale, 4),
                    "predictions": {"kde_mode": kde_default, "mean": mean, "median": median},
                    "mae": {method: round(value, 4)
                            for method, value in mae_values.items()},
                    "relative_mae": {method: round(value / scale, 4)
                                     for method, value in mae_values.items()},
                    "rmse": {method: round(float(np.sqrt(np.mean(squared[method]))), 4)
                             for method in squared},
                    "best_method": strict_best[0] if len(strict_best) == 1 else "tie",
                    "kde_interval_coverage": round(
                        float(np.mean([lower <= value <= upper for value in test_numeric])), 4),
                    "mean_2sd_coverage": round(
                        float(np.mean([mean_lower <= value <= mean_upper
                                       for value in test_numeric])), 4),
                })
            elif kind == "string":
                train_strings = [str(value) for value in values if _json_type(value) == "string"]
                test_strings = [str(value) for value in test_values]
                kde_default, kde_method = _string_kde_default(train_strings, tool, param)
                mode_default = Counter(train_strings).most_common(1)[0][0]
                exact = {
                    "kde_default": float(np.mean([value == kde_default
                                                  for value in test_strings])),
                    "mode": float(np.mean([value == mode_default
                                           for value in test_strings])),
                }
                normalized = {
                    "kde_default": float(np.mean(
                        [value.strip().lower() == kde_default.strip().lower()
                         for value in test_strings])),
                    "mode": float(np.mean(
                        [value.strip().lower() == mode_default.strip().lower()
                         for value in test_strings])),
                }
                exact_values = {method: round(value, 4) for method, value in exact.items()}
                strict_best = [method for method, value in exact_values.items()
                               if value == max(exact_values.values())]
                string_rows.append({
                    "fold": family,
                    "tool": tool,
                    "param": param,
                    "param_type": "string",
                    "train_samples": len(train_strings),
                    "test_samples": len(test_strings),
                    "unique_train_values": len(set(train_strings)),
                    "kde_method": kde_method,
                    "predictions": {"kde_default": kde_default, "mode": mode_default},
                    "exact_match": exact_values,
                    "normalized_match": {method: round(value, 4)
                                         for method, value in normalized.items()},
                    "best_method": strict_best[0] if len(strict_best) == 1 else "tie",
                })
            elif kind == "boolean":
                train_bools = [bool(value) for value in values if _json_type(value) == "boolean"]
                test_bools = [bool(value) for value in test_values]
                majority = sum(train_bools) > len(train_bools) / 2
                boolean_rows.append({
                    "fold": family,
                    "tool": tool,
                    "param": param,
                    "param_type": "boolean",
                    "train_samples": len(train_bools),
                    "test_samples": len(test_bools),
                    "majority_default": majority,
                    "accuracy": round(
                        float(np.mean([value == majority for value in test_bools])), 4),
                })
            else:
                skipped["other_type"] += 1

    # macro aggregation: one weight per parameter (fold-parameter rows are separate)
    numeric_mae = {method: [row["mae"][method] for row in numeric_rows]
                   for method in ("kde", "mean", "median")}
    numeric_coverage = {
        "kde_interval": [row["kde_interval_coverage"] for row in numeric_rows],
        "mean_2sd": [row["mean_2sd_coverage"] for row in numeric_rows],
    }
    numeric_summary = {
        "parameters": len(numeric_rows),
        "test_values": sum(row["test_samples"] for row in numeric_rows),
        "macro_mae": {method: round(float(np.mean(values)), 4)
                      for method, values in numeric_mae.items()},
        "macro_relative_mae": {
            method: round(float(np.median([row["relative_mae"][method]
                                           for row in numeric_rows])), 4)
            for method in ("kde", "mean", "median")},
        "mean_relative_mae": {
            method: round(float(np.mean([row["relative_mae"][method]
                                         for row in numeric_rows])), 4)
            for method in ("kde", "mean", "median")},
        "micro_relative_mae": {
            method: round(float(np.average(
                [row["relative_mae"][method] for row in numeric_rows],
                weights=[row["test_samples"] for row in numeric_rows])), 4)
            for method in ("kde", "mean", "median")},
        "macro_rmse": {method: round(float(np.mean([row["rmse"][method]
                                                    for row in numeric_rows])), 4)
                       for method in ("kde", "mean", "median")},
        "micro_mae": {
            method: round(float(np.average(
                [row["mae"][method] for row in numeric_rows],
                weights=[row["test_samples"] for row in numeric_rows])), 4)
            for method in ("kde", "mean", "median")},
        "wins": dict(Counter(row["best_method"] for row in numeric_rows)),
        "macro_coverage": {name: round(float(np.mean(values)), 4)
                           for name, values in numeric_coverage.items()},
    }
    string_summary = {
        "parameters": len(string_rows),
        "test_values": sum(row["test_samples"] for row in string_rows),
        "macro_exact_match": {
            method: round(float(np.mean([row["exact_match"][method] for row in string_rows])), 4)
            for method in ("kde_default", "mode")},
        "micro_exact_match": {
            method: round(float(np.average(
                [row["exact_match"][method] for row in string_rows],
                weights=[row["test_samples"] for row in string_rows])), 4)
            for method in ("kde_default", "mode")},
        "macro_normalized_match": {
            method: round(float(np.mean([row["normalized_match"][method]
                                         for row in string_rows])), 4)
            for method in ("kde_default", "mode")},
        "wins": dict(Counter(row["best_method"] for row in string_rows)),
        "kde_methods": dict(Counter(row["kde_method"] for row in string_rows)),
    }
    boolean_summary = {
        "parameters": len(boolean_rows),
        "test_values": sum(row["test_samples"] for row in boolean_rows),
        "macro_accuracy": round(float(np.mean([row["accuracy"] for row in boolean_rows])), 4)
        if boolean_rows else None,
    }

    # ------------------------------------------------------------------
    # 2. downstream: default completion on the 969 evaluable failures
    # ------------------------------------------------------------------
    evaluable_by_family: dict[str, list] = defaultdict(list)
    for call in calls:
        if call.recorded_ok:
            continue
        if call.tool_name not in schemas or "__raw__" in call.arguments:
            continue
        evaluable_by_family[call.model].append(call)

    downstream_rows: list[dict] = []
    current_counts = Counter()
    current_rule_counts: Counter[str] = Counter()
    downstream_diagnostics: Counter[str] = Counter()
    configs: dict[str, Counter] = {
        "schema_default_completion": Counter(),
        "memory_param_mode": Counter(),
        "kde_default_completion": Counter(),
    }
    for family in families:
        train_calls = [call for model in sorted(TRAIN_MODELS) if model != family
                       for call in successes_by_model.get(model, [])]
        test_failures = evaluable_by_family.get(family, [])
        if not train_calls or not test_failures:
            continue
        memory = SuccessMemory.build(train_calls, schemas)
        extractor_map = ExtractorMap.learn(train_calls, schemas)
        test_successes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for call in successes_by_model.get(family, []):
            test_successes[(call.group, call.query_id, call.tool_name)].append(
                dict(call.arguments))
        # fitted defaults from this fold's training side
        train_values: dict[tuple[str, str], list] = defaultdict(list)
        for call in train_calls:
            for name, value in call.arguments.items():
                if _json_type(value) is not None:
                    train_values[(call.tool_name, name)].append(value)
        fitted: dict[tuple[str, str], dict] = {}
        for (tool, param), values in train_values.items():
            if len(values) < _MIN_SAMPLES:
                continue
            kind = _dominant_type(values)
            if kind in ("integer", "number"):
                numeric = [float(value) for value in values if _json_type(value) == kind]
                default, _lower, _upper = _kde_numeric_default(numeric)
                fitted[(tool, param)] = {"value": default, "type": kind,
                                         "method": "gaussian_kde_peak"}
            elif kind == "string":
                strings = [str(value) for value in values if _json_type(value) == "string"]
                default, method = _string_kde_default(strings, tool, param)
                fitted[(tool, param)] = {"value": default, "type": kind, "method": method}
            elif kind == "boolean":
                numeric = [bool(value) for value in values if _json_type(value) == "boolean"]
                fitted[(tool, param)] = {"value": sum(numeric) > len(numeric) / 2,
                                         "type": kind, "method": "majority"}
        for call in test_failures:
            schema = schemas[call.tool_name]
            missing = [name for name in schema.required
                       if call.arguments.get(name) in (None, "")]
            if not missing:
                continue
            reference = test_successes.get(
                (call.group, call.query_id, call.tool_name)) or []
            downstream_diagnostics["targeted_param_pairs"] += len(missing)
            for name in missing:
                if (call.tool_name, name) in fitted:
                    downstream_diagnostics["fitted_kde_pairs"] += 1
                if schema.default_for(name) is not None:
                    downstream_diagnostics["schema_default_pairs"] += 1
                if memory.param_values(call.tool_name, name):
                    downstream_diagnostics["memory_mode_pairs"] += 1
            row = {
                "fold": family,
                "tool": call.tool_name,
                "kind": classify_error(call.error),
                "missing_required": missing,
                "reference_count": len(reference),
            }
            # current engine: full ladder, then attribute the accepted rule category
            current = repair_failure(call, schemas, memory, extractor_map,
                                     successful_arguments=reference)
            current_counts["evaluable_with_missing_required"] += 1
            row["current_status"] = (
                "accepted_schema_valid" if current.repairable and current.schema_valid else
                "accepted_value_verified" if current.repairable else
                "proposed_schema_rejected" if current.proposed else "no_proposal")
            if current.repairable:
                category = (current.rules[0].split(":")[0] if current.rules else "unknown")
                current_rule_counts[category] += 1
                row["current_rule"] = category
            # candidate completions
            for config_name, candidate in (
                ("schema_default_completion",
                 _schema_default_completion(call, schema, classify_error(call.error))),
                ("memory_param_mode", _memory_mode_completion(call, schema, memory)),
                ("kde_default_completion", _kde_completion(
                    call, schema, fitted, extractor_map=None)),
            ):
                outcome = _judge_candidate(candidate, schema, reference)
                configs[config_name][outcome["status"]] += 1
                if (outcome["status"] == "accepted_schema_valid"
                        and str(row.get("current_status", "")).startswith("accepted")):
                    configs[config_name]["schema_valid_already_accepted"] += 1
                if candidate is None:
                    continue
                row[config_name] = {
                    "status": outcome["status"],
                    "arguments": candidate.arguments,
                    "rules": list(candidate.rules),
                }
            downstream_rows.append(row)

    downstream = {}
    targeted_by_kind = Counter(row["kind"] for row in downstream_rows)
    for config_name, counter in configs.items():
        downstream[config_name] = {
            "targeted_failures": current_counts["evaluable_with_missing_required"],
            "proposed": counter.get("accepted_schema_valid", 0)
            + counter.get("accepted_value_verified", 0)
            + counter.get("proposed_schema_rejected", 0),
            "schema_valid": counter.get("accepted_schema_valid", 0),
            "value_verified": counter.get("accepted_value_verified", 0),
            "schema_rejected": counter.get("proposed_schema_rejected", 0),
            "no_proposal": counter.get("no_proposal", 0),
            "schema_valid_already_accepted_by_current_ladder": counter.get(
                "schema_valid_already_accepted", 0),
            "schema_valid_by_kind": dict(Counter(
                row["kind"] for row in downstream_rows
                if (row.get(config_name) or {}).get("status") == "accepted_schema_valid")),
        }
    downstream["current_ladder"] = {
        "targeted_failures": current_counts["evaluable_with_missing_required"],
        "accepted_by_rule": dict(current_rule_counts),
        "accepted_by_kind": dict(Counter(
            row["kind"] for row in downstream_rows
            if str(row.get("current_status", "")).startswith("accepted"))),
    }
    downstream["targeted_by_kind"] = dict(targeted_by_kind)
    downstream["diagnostics"] = dict(downstream_diagnostics)

    # ------------------------------------------------------------------
    # 3. answer + output
    # ------------------------------------------------------------------
    kde_numeric_better = (numeric_summary["macro_relative_mae"]["kde"]
                          < min(numeric_summary["macro_relative_mae"]["mean"],
                                numeric_summary["macro_relative_mae"]["median"]))
    kde_string_better = (string_summary["macro_exact_match"]["kde_default"]
                         > string_summary["macro_exact_match"]["mode"])
    result = {
        "experiment": "ToolGuard B3 KDE parameter-default diagnostic",
        "protocol": {
            "scope": "diagnosis only; no production code and no headline change",
            "held_out": ("for each model family, fit on the fixed-three training "
                         "successes excluding that family, evaluate on that family's "
                         "successes (LOMO/cross-model split)"),
            "min_train_samples": _MIN_SAMPLES,
            "success_definition": "recorded_ok and valid_data (same as SuccessMemory)",
            "methods": {
                "numeric": "Gaussian KDE peak (KDEAnalyzer-style) vs mean vs median; "
                           "MAE/RMSE on held-out values; 95% interval coverage",
                "string": "KDEAnalyzer TF-IDF+KMeans dominant-cluster default vs "
                          "most-frequent value; exact-match on held-out values",
                "boolean": "majority vote accuracy",
            },
            "folds": families,
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
            "skipped_pairs": dict(skipped),
        },
        "numeric": {
            "summary": numeric_summary,
            "parameters": numeric_rows,
        },
        "string": {
            "summary": string_summary,
            "parameters": string_rows,
        },
        "boolean": {
            "summary": boolean_summary,
            "parameters": boolean_rows,
        },
        "downstream": {
            "configurations": downstream,
            "rows": downstream_rows,
            "note": ("schema-valid is structural only; value-verified is an exact "
                     "test-fold recorded success (diagnostic tier, not independent)"),
        },
        "answer": {
            "kde_numeric_beats_simple_stats": bool(kde_numeric_better),
            "kde_string_beats_mode": bool(kde_string_better),
            "verdict": (
                "KDE defaults are NOT better than simple statistics on the held-out "
                "families: "
                f"numeric macro relative-MAE kde={numeric_summary['macro_relative_mae']['kde']} "
                f"vs mean={numeric_summary['macro_relative_mae']['mean']} "
                f"vs median={numeric_summary['macro_relative_mae']['median']} "
                f"(strict per-parameter wins: {numeric_summary['wins']}); string exact-match "
                f"kde={string_summary['macro_exact_match']['kde_default']} vs "
                f"mode={string_summary['macro_exact_match']['mode']} "
                f"(strict per-parameter wins: {string_summary['wins']}). The "
                "KDE/TF-IDF+KMeans machinery adds complexity without held-out predictive "
                "gain, so the current simple schema/memory defaults stay the production "
                "choice.")
            if not (kde_numeric_better or kde_string_better) else
            ("mixed result: see the per-method tables; no production change is made from "
             "this diagnostic"),
            "downstream_note": ("the downstream funnel compares each default source with "
                                "the same schema gate and the same test-fold recorded-"
                                "success tier, so the numbers are comparable but not "
                                "independent verification"),
        },
        "limitations": [
            "the 30-sample bar keeps only heavily repeated (tool, parameter) pairs; the "
            "conclusion applies to those pairs, not to rare parameters",
            "string defaults are mostly the mode of near-constant values, so both methods "
            "score high trivially; the exact-match metric is dominated by the mode",
            "the test values are real recorded successes of the held-out family; a default "
            "that matches them is predictive for this dataset, not proof of semantic "
            "correctness",
            "downstream value-verified uses the test-fold recorded success (diagnostic, "
            "identical to the evaluator's tier)",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kde_diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "kde_diagnostic.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'kde_diagnostic.json'}")
    print(f"wrote {output_dir / 'kde_diagnostic.md'}")
    return 0


def _memory_mode_completion(call, schema, memory: SuccessMemory):
    """Candidate repair: fill missing required params with their training-success mode."""
    missing = [name for name in schema.required if call.arguments.get(name) in (None, "")]
    if not missing:
        return None
    from tool_evolution.toolguard.rules import RepairResult

    arguments = dict(call.arguments)
    rules: list[str] = []
    for name in missing:
        values = memory.param_values(call.tool_name, name)
        if not values:
            continue
        arguments[name] = values.most_common(1)[0][0]
        rules.append(f"memory_param_mode:{name}")
    if not rules:
        return None
    errors = validate_arguments(arguments, schema)
    return RepairResult(kind=classify_error(call.error), repairable=not errors, proposed=True,
                        rules=rules, function_name=call.tool_name, arguments=arguments,
                        schema_valid=not errors, schema_errors=errors,
                        reason="filled missing required parameters with the training-success mode")


def _kde_completion(call, schema, fitted: dict, extractor_map=None):
    """Candidate repair: fill missing required params with fitted KDE/TFIDF/majority defaults."""
    missing = [name for name in schema.required if call.arguments.get(name) in (None, "")]
    if not missing:
        return None
    from tool_evolution.toolguard.rules import RepairResult

    arguments = dict(call.arguments)
    rules: list[str] = []
    for name in missing:
        entry = fitted.get((call.tool_name, name))
        if entry is None:
            continue
        declared = str((schema.properties.get(name) or {}).get("type") or "") or None
        value = _coerce_candidate(entry["value"], declared)
        if value is None:
            continue
        arguments[name] = value
        rules.append(f"kde_default:{name}")
    if not rules:
        return None
    errors = validate_arguments(arguments, schema)
    return RepairResult(kind=classify_error(call.error), repairable=not errors, proposed=True,
                        rules=rules, function_name=call.tool_name, arguments=arguments,
                        schema_valid=not errors, schema_errors=errors,
                        reason="filled missing required parameters with held-out-fitted KDE defaults")


def _judge_candidate(candidate, schema, reference: list[dict]) -> dict:
    if candidate is None:
        return {"status": "no_proposal"}
    if candidate.repairable and candidate.schema_valid:
        return {"status": "accepted_schema_valid"}
    if candidate.repairable:
        return {"status": "accepted_value_verified"}
    if (reference and candidate.arguments is not None
            and any(candidate.arguments == recorded for recorded in reference)):
        return {"status": "accepted_value_verified"}
    return {"status": "proposed_schema_rejected"}


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _render_report(result: dict) -> str:
    lines = ["# B3 KDE parameter-default diagnostic (held-out; diagnosis only)\n"]
    protocol = result["protocol"]
    lines.append(f"Held-out protocol: {protocol['held_out']}; minimum training samples per "
                 f"parameter: {protocol['min_train_samples']}; success = "
                 f"{protocol['success_definition']}.\n")
    numeric = result["numeric"]["summary"]
    string = result["string"]["summary"]
    boolean = result["boolean"]["summary"]
    lines.append("## Numeric parameters: KDE peak vs mean vs median (held-out)\n")
    lines.append(_md_table(
        ["method", "median relative MAE", "mean relative MAE", "micro relative MAE",
         "strict per-param wins"],
        [[method, numeric["macro_relative_mae"][method],
          numeric["mean_relative_mae"][method], numeric["micro_relative_mae"][method],
          numeric["wins"].get(method, 0)]
         for method in ("kde", "mean", "median")]))
    lines.append(f"\nparameters: {numeric['parameters']}; held-out values: "
                 f"{numeric['test_values']}; relative MAE = MAE / mean(|held-out value|), "
                 "so it is comparable across parameter scales; ties are reported as "
                 f"`tie` ({numeric['wins'].get('tie', 0)} params); 95% interval coverage: "
                 f"KDE interval {numeric['macro_coverage']['kde_interval']}, "
                 f"mean+/-2sd {numeric['macro_coverage']['mean_2sd']}\n")
    lines.append("## String parameters: TF-IDF+KMeans default vs mode (held-out)\n")
    lines.append(_md_table(
        ["method", "macro exact match", "micro exact match", "macro normalized", "wins"],
        [[method, string["macro_exact_match"][method], string["micro_exact_match"][method],
          string["macro_normalized_match"][method], string["wins"].get(method, 0)]
         for method in ("kde_default", "mode")]))
    lines.append(f"\nparameters: {string['parameters']}; held-out values: "
                 f"{string['test_values']}; KMeans methods: "
                 f"{json.dumps(string['kde_methods'], ensure_ascii=False)}\n")
    lines.append("## Boolean parameters: majority vote (held-out)\n")
    lines.append(_md_table(
        ["parameters", "held-out values", "macro accuracy"],
        [[boolean["parameters"], boolean["test_values"], boolean["macro_accuracy"]]]))
    lines.append("\n## Downstream: default completion on the 969 evaluable failures\n")
    downstream = result["downstream"]["configurations"]
    lines.append(f"failures that still miss a schema-declared required parameter: "
                 f"**{downstream.get('current_ladder', {}).get('targeted_failures', 0)}** "
                 f"of 969 (by kind: "
                 f"{json.dumps(downstream.get('targeted_by_kind', {}), ensure_ascii=False)}; "
                 "the A1.8.1 finding is that most error-named parameters are not declared "
                 "in the schema index at all, so they cannot be targeted this way)\n")
    lines.append(f"missing-parameter pair diagnostics: "
                 f"{json.dumps(downstream.get('diagnostics', {}), ensure_ascii=False)} "
                 "(targeted = missing (function, parameter) pairs; fitted_kde = pairs with "
                 ">=30 training-side successes of the same function; schema_default = pairs "
                 "with a schema example/default; memory_mode = pairs with any training-side "
                 "success value)\n")
    lines.append(_md_table(
        ["configuration", "targeted failures", "proposed", "schema-valid",
         "value-verified", "schema rejected", "no proposal", "schema-valid already "
         "accepted by current ladder"],
        [[name, counts.get("targeted_failures", 0), counts.get("proposed", 0),
          counts.get("schema_valid", 0), counts.get("value_verified", 0),
          counts.get("schema_rejected", 0), counts.get("no_proposal", 0),
          counts.get("schema_valid_already_accepted_by_current_ladder", 0)]
         for name, counts in downstream.items() if name not in
         ("current_ladder", "targeted_by_kind", "diagnostics")]))
    current = downstream.get("current_ladder", {})
    lines.append(f"\ncurrent ladder accepted rules on the same targeted failures: "
                 f"{json.dumps(current.get('accepted_by_rule', {}), ensure_ascii=False)} "
                 f"(by kind: {json.dumps(current.get('accepted_by_kind', {}), ensure_ascii=False)})\n")
    lines.append("\nschema-valid by kind, per default source: "
                 + json.dumps({name: counts.get("schema_valid_by_kind", {})
                               for name, counts in downstream.items()
                               if name not in ("current_ladder", "targeted_by_kind",
                                               "diagnostics")},
                              ensure_ascii=False) + "\n")
    lines.append("\n## Per-parameter method table (numeric)\n")
    lines.append(_md_table(
        ["fold", "tool", "param", "type", "train", "test", "relMAE kde", "relMAE mean",
         "relMAE median", "best", "KDE cover", "mean+-2sd cover"],
        [[row["fold"], row["tool"], row["param"], row["param_type"], row["train_samples"],
          row["test_samples"], row["relative_mae"]["kde"], row["relative_mae"]["mean"],
          row["relative_mae"]["median"], row["best_method"],
          row["kde_interval_coverage"], row["mean_2sd_coverage"]]
         for row in result["numeric"]["parameters"]]))
    lines.append("\n## Per-parameter method table (string)\n")
    lines.append(_md_table(
        ["fold", "tool", "param", "train", "test", "unique train", "kde method",
         "kde default", "mode", "exact kde", "exact mode", "best"],
        [[row["fold"], row["tool"], row["param"], row["train_samples"], row["test_samples"],
          row["unique_train_values"], row["kde_method"],
          row["predictions"]["kde_default"], row["predictions"]["mode"],
          row["exact_match"]["kde_default"], row["exact_match"]["mode"], row["best_method"]]
         for row in result["string"]["parameters"]]))
    lines.append("\n## Per-parameter method table (boolean)\n")
    lines.append(_md_table(
        ["fold", "tool", "param", "train", "test", "majority", "accuracy"],
        [[row["fold"], row["tool"], row["param"], row["train_samples"], row["test_samples"],
          row["majority_default"], row["accuracy"]]
         for row in result["boolean"]["parameters"]]))
    lines.append("\n## Answer\n")
    lines.append(result["answer"]["verdict"] + "\n")
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

