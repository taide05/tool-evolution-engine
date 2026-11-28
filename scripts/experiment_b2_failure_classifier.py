"""B2 diagnostic: is a learned failure classifier worth replacing the rule classifier?

Scope: **diagnosis only**, no production change, no headline change.

Labels
------
The only labels available in the repository are the deterministic rule labels produced by
``toolguard.rules.classify_error`` (missing_required / invalid_value / server_error / other;
the taxonomy also defines no_such_function / unexpected_kwarg / input_parse, which do not
occur among the 969 evaluable failures).  There is **no independent human ground-truth
``error_type``** field in the recorded calls, so nothing is fabricated: every number below
is an *agreement with the rule labels*, never a "true accuracy".

Split (same LOMO/cross-model protocol as the headline):
    train = failures of the fixed three training models (chatgpt / toolllama /
            gpt-3.5-turbo-1106), 692 rows
    test  = failures of the gpt-4 family (gpt-4-0613 / gpt-4-turbo-preview), 277 rows

The feature set mirrors the old ``tool_evolution.analysis.classifier.FailureClassifier``
(tool_name + error_message char_wb TF-IDF 2-4 grams, param_count, has_auth_header, has_cjk)
with the label taxonomy re-targeted at the rule classes.  TF-IDF and the tool-name encoder
are fit on the training split only.

Three configurations are compared:
    full           text + metadata + tool id (the old feature idea)
    text_only      error-message TF-IDF only
    no_error_text  metadata + tool id only (what remains when the wording is unavailable)

Usage:
    python scripts/experiment_b2_failure_classifier.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from tool_evolution.toolguard.records import load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
TEST_MODELS = {"gpt-4-0613", "gpt-4-turbo-preview"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}
_RULES_TAXONOMY = ["missing_required", "invalid_value", "server_error", "other",
                   "no_such_function", "unexpected_kwarg", "input_parse"]


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff" for char in text)


def _has_auth_header(arguments: dict) -> bool:
    return any(str(name).lower() in ("auth", "authorization", "token", "api_key")
               for name in arguments)


def _row(call) -> dict:
    return {
        "tool": call.tool_name,
        "error": call.error or "",
        "kind": classify_error(call.error),
        "param_count": len(call.arguments),
        "has_auth_header": int(_has_auth_header(call.arguments)),
        "has_cjk": int(_has_cjk(call.error or "")),
        "model": call.model,
        "group": call.group,
        "query_id": call.query_id,
    }


def _fit_predict(train: list[dict], test: list[dict], config: str,
                 seed: int = 42) -> dict:
    labels = sorted({row["kind"] for row in train + test})
    y_train = np.array([row["kind"] for row in train])
    y_test = np.array([row["kind"] for row in test])

    tfidf = None
    if config in ("full", "text_only"):
        texts_train = [f"{row['tool']} {row['error']}" if config == "full" else row["error"]
                       for row in train]
        texts_test = [f"{row['tool']} {row['error']}" if config == "full" else row["error"]
                      for row in test]
        tfidf = TfidfVectorizer(max_features=500, analyzer="char_wb", ngram_range=(2, 4))
        train_matrix = tfidf.fit_transform(texts_train).toarray()
        test_matrix = tfidf.transform(texts_test).toarray()

    tool_encoder: dict[str, int] = {}
    if config == "full":
        train_tools = [row["tool"] for row in train]
        tool_encoder = {name: index + 1 for index, name in enumerate(sorted(set(train_tools)))}
        train_tool = np.array([[tool_encoder.get(row["tool"], 0)] for row in train])
        test_tool = np.array([[tool_encoder.get(row["tool"], 0)] for row in test])

    metadata_train = np.array([[row["param_count"], row["has_auth_header"], row["has_cjk"]]
                               for row in train])
    metadata_test = np.array([[row["param_count"], row["has_auth_header"], row["has_cjk"]]
                              for row in test])
    if config == "full":
        x_train = np.hstack([train_matrix, metadata_train, train_tool])
        x_test = np.hstack([test_matrix, metadata_test, test_tool])
    else:  # text_only
        x_train, x_test = train_matrix, test_matrix

    classifier = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    majority_label = Counter(y_train).most_common(1)[0][0]
    majority_agreement = float(np.mean(y_test == majority_label))

    feature_names: list[str] = []
    importances: dict[str, float] = {}
    if config == "full":
        feature_names = (list(tfidf.get_feature_names_out())
                         + ["param_count", "has_auth_header", "has_cjk", "tool_name"])
        importances = dict(sorted(
            zip(feature_names, classifier.feature_importances_.tolist()),
            key=lambda item: -item[1])[:20])

    disagreements = []
    for index, row in enumerate(test):
        if predictions[index] == y_test[index]:
            continue
        disagreements.append({
            "tool": row["tool"],
            "error_head": row["error"][:180],
            "rule_label": str(y_test[index]),
            "predicted_label": str(predictions[index]),
            "confidence": round(float(probabilities[index].max()), 4),
        })
    disagreements.sort(key=lambda item: -item["confidence"])

    return {
        "config": config,
        "labels": labels,
        "train_rows": len(train),
        "test_rows": len(test),
        "agreement_with_rule_labels": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1_vs_rule_labels": round(float(f1_score(y_test, predictions, average="macro",
                                                        labels=labels, zero_division=0)), 4),
        "weighted_f1_vs_rule_labels": round(float(f1_score(
            y_test, predictions, average="weighted", labels=labels, zero_division=0)), 4),
        "majority_baseline_agreement": round(majority_agreement, 4),
        "majority_baseline_label": majority_label,
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.astype(int).tolist(),
        },
        "disagreement_count": len(disagreements),
        "disagreement_examples": disagreements[:20],
        "top_feature_importances": importances,
    }


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=".scratch/toolguard/tool_calls.jsonl")
    parser.add_argument("--queries", default=(
        ".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/"
        "solvable_queries/test_instruction"))
    parser.add_argument("--output-dir", default=".scratch/toolguard/b2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(repo_root / args.queries)
    schemas, schema_fidelity = refine_schema_index(schemas, calls)
    failures = [call for call in calls if not call.recorded_ok]
    evaluable = [call for call in failures
                 if call.tool_name in schemas and "__raw__" not in call.arguments]
    rows = [_row(call) for call in evaluable]
    train = [row for row in rows if row["model"] in TRAIN_MODELS]
    test = [row for row in rows if row["model"] in TEST_MODELS]
    print(f"failures: {len(failures)} | evaluable: {len(evaluable)} | "
          f"train: {len(train)} | test: {len(test)}")
    print(f"train labels: {dict(Counter(row['kind'] for row in train))}")
    print(f"test labels: {dict(Counter(row['kind'] for row in test))}")

    configs = {}
    for config in ("full", "text_only", "no_error_text"):
        if config == "no_error_text":
            # metadata + tool id only; build the tool column explicitly
            train_meta = [row for row in train]
            test_meta = [row for row in test]
            tools = sorted({row["tool"] for row in train_meta})
            tool_index = {name: index + 1 for index, name in enumerate(tools)}
            y_train = np.array([row["kind"] for row in train_meta])
            y_test = np.array([row["kind"] for row in test_meta])
            labels = sorted(set(y_train) | set(y_test))
            x_train = np.array([[row["param_count"], row["has_auth_header"], row["has_cjk"],
                                 tool_index.get(row["tool"], 0)] for row in train_meta])
            x_test = np.array([[row["param_count"], row["has_auth_header"], row["has_cjk"],
                                tool_index.get(row["tool"], 0)] for row in test_meta])
            classifier = RandomForestClassifier(n_estimators=200, random_state=args.seed)
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, predictions, labels=labels, zero_division=0)
            majority_label = Counter(y_train).most_common(1)[0][0]
            configs[config] = {
                "config": config,
                "labels": labels,
                "train_rows": len(train_meta),
                "test_rows": len(test_meta),
                "agreement_with_rule_labels": round(float(accuracy_score(y_test,
                                                                         predictions)), 4),
                "macro_f1_vs_rule_labels": round(float(f1_score(
                    y_test, predictions, average="macro", labels=labels, zero_division=0)), 4),
                "majority_baseline_agreement": round(float(np.mean(y_test == majority_label)), 4),
                "majority_baseline_label": majority_label,
                "per_class": {
                    label: {"precision": round(float(precision[index]), 4),
                            "recall": round(float(recall[index]), 4),
                            "f1": round(float(f1[index]), 4),
                            "support": int(support[index])}
                    for index, label in enumerate(labels)},
                "confusion_matrix": {"labels": labels,
                                     "matrix": confusion_matrix(
                                         y_test, predictions, labels=labels).astype(int).tolist()},
            }
        else:
            configs[config] = _fit_predict(train, test, config, seed=args.seed)

    result = {
        "experiment": "ToolGuard B2 failure-classifier diagnostic",
        "protocol": {
            "scope": "diagnosis only; no production code and no headline change",
            "label_source": (
                "toolguard.rules.classify_error deterministic rule labels; there is no "
                "independent human-ground-truth error_type in the recorded calls, so every "
                "metric is 'agreement with rule labels', never true accuracy"),
            "rule_taxonomy": _RULES_TAXONOMY,
            "labels_present_in_data": sorted({row["kind"] for row in rows}),
            "split": {
                "train_models": sorted(TRAIN_MODELS),
                "test_models": sorted(TEST_MODELS),
                "train_rows": len(train),
                "test_rows": len(test),
            },
            "data": {
                "all_calls": len(calls),
                "real_failures": len(failures),
                "evaluable_failures": len(evaluable),
                "label_distribution_all_evaluable": dict(
                    Counter(row["kind"] for row in rows)),
                "label_distribution_train": dict(Counter(row["kind"] for row in train)),
                "label_distribution_test": dict(Counter(row["kind"] for row in test)),
            },
            "features": {
                "full": ("char_wb 2-4 gram TF-IDF (max 500) on tool_name + error_message, "
                         "fit on train only; param_count, has_auth_header, has_cjk; "
                         "tool-name categorical id"),
                "text_only": "char_wb 2-4 gram TF-IDF of error_message only",
                "no_error_text": "metadata + tool-name id only (no wording)",
                "classifier": "RandomForestClassifier(n_estimators=200, seed=42)",
            },
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
        },
        "configurations": configs,
        "answer": {
            "question": ("is a learned failure classifier worth replacing the rule "
                         "classifier?"),
            "verdict": ("no / not supported. With the error message available the learned "
                        "model can reproduce the hand-written rule labels almost perfectly, "
                        "but that is circular: the labels are a deterministic function of the "
                        "same error text, so agreement is a consistency check, not evidence of "
                        "additional diagnostic value. Without the wording the agreement drops "
                        "to the majority baseline, so the model adds no independent signal. "
                        "The rule classifier stays the production choice: deterministic, "
                        "auditable, zero training, and it defines the labels. The learnable part "
                        "would only matter for unseen error phrasing, which has no ground truth "
                        "in this dataset and therefore cannot be validated."),
            "usable_evidence": ("the per-class precision/recall and the confusion matrix show "
                                "which rule classes a learned model would confuse if it were "
                                "adopted; they do not establish real accuracy"),
        },
        "limitations": [
            "no independent ground truth: only 4 of the 7 rule classes occur among the 969 "
            "evaluable failures (missing_required/invalid_value/server_error/other)",
            "the test split is small for rare classes (server_error 23, invalid_value 22, "
            "other 16), so per-class recall is high-variance",
            "the classifier is trained and evaluated on the same generated error wording; it "
            "cannot be validated against unseen phrasings without human labels",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classifier_diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "classifier_diagnostic.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'classifier_diagnostic.json'}")
    print(f"wrote {output_dir / 'classifier_diagnostic.md'}")
    return 0


def _render_report(result: dict) -> str:
    lines = ["# B2 failure-classifier diagnostic (diagnosis only)\n"]
    protocol = result["protocol"]
    lines.append(f"Labels: **{protocol['label_source']}**\n")
    data = protocol["data"]
    split = protocol["split"]
    lines.append(_md_table(
        ["item", "value"],
        [["all calls", data["all_calls"]],
         ["real failures", data["real_failures"]],
         ["evaluable failures", data["evaluable_failures"]],
         ["train rows (fixed-three)", split["train_rows"]],
         ["test rows (gpt-4 family)", split["test_rows"]],
         ["labels present", ", ".join(protocol["labels_present_in_data"])],
         ["train label distribution",
          json.dumps(data["label_distribution_train"], ensure_ascii=False)],
         ["test label distribution",
          json.dumps(data["label_distribution_test"], ensure_ascii=False)]]))

    lines.append("\n## Agreement with rule labels (not accuracy; no ground truth)\n")
    lines.append(_md_table(
        ["configuration", "test rows", "agreement", "macro-F1", "weighted-F1",
         "majority agreement"],
        [[name, config["test_rows"], config["agreement_with_rule_labels"],
          config["macro_f1_vs_rule_labels"],
          config.get("weighted_f1_vs_rule_labels", "-"),
          config["majority_baseline_agreement"]]
         for name, config in result["configurations"].items()]))

    for name, config in result["configurations"].items():
        lines.append(f"\n### Per-class metrics — {name}\n")
        lines.append(_md_table(
            ["label", "precision", "recall", "f1", "support"],
            [[label, metrics["precision"], metrics["recall"], metrics["f1"],
              metrics["support"]]
             for label, metrics in config["per_class"].items()]))
        lines.append(f"\nmacro-F1: {config['macro_f1_vs_rule_labels']}; "
                     f"majority baseline: {config['majority_baseline_agreement']} "
                     f"({config['majority_baseline_label']})\n")
        matrix = config["confusion_matrix"]
        lines.append("\nConfusion matrix (rows = rule label, columns = predicted; "
                     f"labels {matrix['labels']}):\n")
        lines.append(_md_table(
            ["true \\ predicted"] + matrix["labels"],
            [[label] + row for label, row in zip(matrix["labels"], matrix["matrix"])]))
        if config.get("disagreement_examples"):
            lines.append("\nTop disagreements:\n")
            lines.append(_md_table(
                ["tool", "error head", "rule label", "predicted", "confidence"],
                [[example["tool"], example["error_head"], example["rule_label"],
                  example["predicted_label"], example["confidence"]]
                 for example in config["disagreement_examples"][:10]]))

    full = result["configurations"]["full"]
    if full.get("top_feature_importances"):
        lines.append("\n## Full-model top feature importances\n")
        lines.append(_md_table(["feature", "importance"],
                               [[feature, round(value, 5)]
                                for feature, value in full["top_feature_importances"].items()]))
    lines.append("\n## Answer\n")
    lines.append(result["answer"]["verdict"] + "\n")
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
