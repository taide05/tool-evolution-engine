"""A1.8.3 replay-safety experiment: training-only replay gain vs the test-fold oracle.

Reads the three A1.7/A1.8.3 experiment runs and reports, for the 228 missing_required
no-proposal subset:

* **training-only replay** (``--replay-source training``; the honest configuration):
  replay candidates come from the *training-side* ``SuccessMemory`` only.  Because the
  baseline ladder already proposes every training-side recorded success, the expected
  gain is 0 -- this script reports it rather than assuming it;
* **test-fold oracle replay under the A1.8.3 safety gates**
  (``--replay-source test-fold-oracle``): the same A1.7 candidate source, but now gated
  by the explicit read-only allowlist, the side-effect-name denial and the non-empty
  declared-schema requirement;
* **historical A1.7 unsafe run** (kept at ``a17_results_unsafe_oracle_ff0d5ce.json``):
  the 77 accepted repairs, of which 13 replay admissions relied on the test-fold oracle
  (6 of them also on a completely empty schema and placeholder values).  Used only as a
  before/after comparison, never as a result.

The output also checks that the default configuration (no replay) still reproduces the
headline aggregate.

Usage:
    python scripts/experiment_a17_value_recovery.py --replay-source training \
        --output-dir .scratch/toolguard/a18/a17_training
    python scripts/experiment_a17_value_recovery.py --replay-source test-fold-oracle \
        --output-dir .scratch/toolguard/a18/a17_oracle_gated
    python scripts/experiment_a18_replay_safety.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.engine import is_side_effect_tool  # noqa: E402

_CASE_KEY_FIELDS = ("fold", "tool", "group", "query_id", "trace_id")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_key(case: dict) -> tuple:
    return tuple(case.get(field) for field in _CASE_KEY_FIELDS)


def _generic_funnel(result: dict) -> dict:
    return result["subset_tables"]["generic"]["a17_replay"]


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-results",
                        default=".scratch/toolguard/a18/a17_training/a17_results.json")
    parser.add_argument("--oracle-results",
                        default=".scratch/toolguard/a18/a17_oracle_gated/a17_results.json")
    parser.add_argument("--historical-unsafe-results",
                        default=".scratch/toolguard/a17/a17_results_unsafe_oracle_ff0d5ce.json")
    parser.add_argument("--output-dir", default=".scratch/toolguard/a18")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    training = _load(repo_root / args.training_results)
    oracle = _load(repo_root / args.oracle_results)
    historical = _load(repo_root / args.historical_unsafe_results)

    # ------------------------------------------------------------------
    # 1. funnel comparison
    # ------------------------------------------------------------------
    training_generic = _generic_funnel(training)
    oracle_generic = _generic_funnel(oracle)
    historical_generic = _generic_funnel(historical)
    training_total = training["subset_tables"]["total"]["a17_after"]
    oracle_total = oracle["subset_tables"]["total"]["a17_after"]
    historical_total = historical["subset_tables"]["total"]["a17_after"]

    # training-side replay candidates available for the 133 generic cases
    training_candidates = Counter()
    for case in training["cases"]:
        if case.get("named"):
            continue
        count = int((case.get("a17") or {}).get("replay_candidate_count") or 0)
        training_candidates["cases_with_candidate"] += int(count > 0)
        training_candidates["total_candidate_argument_sets"] += count

    # ------------------------------------------------------------------
    # 2. before/after on the historical unsafe replay admissions
    # ------------------------------------------------------------------
    historical_cases = historical["cases"]
    oracle_cases = oracle["cases"]
    training_cases = training["cases"]
    assert len(historical_cases) == len(oracle_cases) == len(training_cases), \
        "the three runs must cover the same 228-case subset"
    blocked_reasons = Counter()
    historical_replay: list[dict] = []
    oracle_replay_accepted: list[dict] = []
    for index, (hist, orc, trn) in enumerate(
            zip(historical_cases, oracle_cases, training_cases)):
        assert _case_key(hist) == _case_key(orc) == _case_key(trn), \
            f"case order mismatch at index {index}"
        hist_result = hist.get("a17") or {}
        if "generic_success_replay" not in (hist_result.get("rules") or []):
            continue
        entry = {
            "index": index,
            "fold": hist.get("fold"),
            "tool": hist.get("tool"),
            "arguments": hist_result.get("arguments"),
            "historical_status": hist_result.get("status"),
            "schema_properties": hist.get("schema_props_before"),
            "is_side_effect_tool": is_side_effect_tool(hist.get("tool") or ""),
            "oracle_gated_status": (orc.get("a17") or {}).get("status"),
            "training_only_status": (trn.get("a17") or {}).get("status"),
        }
        if str(entry["oracle_gated_status"]).startswith("accepted"):
            oracle_replay_accepted.append(entry)
        else:
            if not entry["schema_properties"]:
                reason = "empty declared schema (replay refused by A1.8.3 gate)"
            elif entry["is_side_effect_tool"]:
                reason = "side-effect tool name (replay refused)"
            else:
                reason = "not accepted under the safety gates"
            entry["blocked_reason"] = reason
            blocked_reasons[reason] += 1
        historical_replay.append(entry)

    # ------------------------------------------------------------------
    # 3. cache verification of the (small) oracle gated subset + historical 0/77
    # ------------------------------------------------------------------
    historical_cache = historical["cache_verification"]
    oracle_cache = oracle["cache_verification"]
    cache_check = {
        "historical_unsafe": {
            "accepted_checked": historical_cache.get("accepted_checked", 0),
            "cache_ok": historical_cache.get("verified_recovered", 0),
            "cache_error": historical_cache.get("verified_still_error", 0),
            "cache_miss": historical_cache.get("cache_miss", 0),
        },
        "oracle_gated": {
            "accepted_checked": oracle_cache.get("accepted_checked", 0),
            "cache_ok": oracle_cache.get("verified_recovered", 0),
            "cache_error": oracle_cache.get("verified_still_error", 0),
            "cache_miss": oracle_cache.get("cache_miss", 0),
        },
        "training": {
            "accepted_checked": training["cache_verification"].get("accepted_checked", 0),
            "cache_ok": training["cache_verification"].get("verified_recovered", 0),
            "cache_error": training["cache_verification"].get("verified_still_error", 0),
            "cache_miss": training["cache_verification"].get("cache_miss", 0),
        },
    }

    result = {
        "experiment": "ToolGuard A1.8.3 replay safety",
        "protocol": training["protocol"],
        "default_headline_unchanged": {
            "training_run": training["protocol_verification"],
            "oracle_run": oracle["protocol_verification"],
            "match": bool(training["protocol_verification"]["match"]
                          and oracle["protocol_verification"]["match"]),
        },
        "replay_gates": {
            "allowlist": training["protocol"]["replay"]["allowlist_source"],
            "allowlist_size": training["protocol"]["replay"]["allowlist_size"],
            "gates": training["protocol"]["replay"]["gates"],
            "code": "engine.replay_tool_allowed + engine.is_side_effect_tool",
        },
        "generic_subset_comparison": {
            "training_only": {
                "label": "training-side SuccessMemory replay (honest gain)",
                **training_generic,
                "candidate_cases": training_candidates.get("cases_with_candidate", 0),
                "candidate_argument_sets": training_candidates.get(
                    "total_candidate_argument_sets", 0),
            },
            "test_fold_oracle_gated": {
                "label": "test-fold oracle replay, A1.8.3 gates applied (NOT independent)",
                **oracle_generic,
            },
            "historical_unsafe": {
                "label": "historical A1.7 run (ff0d5ce, no safety gates; for reference only)",
                **historical_generic,
            },
        },
        "total_subset": {
            "historical_unsafe": historical_total,
            "oracle_gated": oracle_total,
            "training_only": training_total,
        },
        "historical_replay_admissions": historical_replay,
        "blocked_by_safety_gates": dict(blocked_reasons),
        "oracle_gated_replay_admissions": [
            {"fold": entry["fold"], "tool": entry["tool"],
             "arguments": entry["arguments"],
             "schema_properties": entry["schema_properties"]}
            for entry in oracle_replay_accepted
        ],
        "cache_verification": cache_check,
        "answer": {
            "training_only_replay_gain": {
                "accepted": training_generic.get("accepted", 0),
                "schema_valid": training_generic.get("schema_valid", 0),
                "value_verified": training_generic.get("value_verified", 0),
                "no_proposal": training_generic.get("no_proposal", 0),
                "verdict": ("negative: there is no training-side recorded success for the "
                            "generic no-proposal failures, so replay cannot add coverage "
                            "honestly") if not training_generic.get("accepted") else
                           "positive (see numbers)",
            },
            "historical_oracle_replay": {
                "accepted": historical_generic.get("accepted", 0),
                "schema_valid": historical_generic.get("schema_valid", 0),
                "value_verified": historical_generic.get("value_verified", 0),
                "note": ("all 13 are test-fold recorded successes; 6 were admitted only "
                         "because the schema index declared no parameters (empty schema) "
                         "and carry placeholder values such as user_id='123456789'"),
            },
            "oracle_under_new_gates": {
                "accepted": oracle_generic.get("accepted", 0),
                "schema_valid": oracle_generic.get("schema_valid", 0),
                "value_verified": oracle_generic.get("value_verified", 0),
                "note": ("even the oracle candidate source now yields at most the "
                         "schema-valid subset; empty-schema/placeholder admissions are "
                         "refused"),
            },
        },
        "limitations": [
            "even the gated oracle row is not independent recovery: the replay candidate "
            "is a recorded success of the evaluated fold",
            "the auto allowlist used by the experiment is an upper bound (every observed "
            "tool name that does not look side-effectful); a production allowlist should be "
            "curated and small",
            "replay safety is name- and schema-based; it cannot detect side effects of a "
            "read-only-looking endpoint (the allowlist remains the primary control)",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "replay_safety.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "replay_safety.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'replay_safety.json'}")
    print(f"wrote {output_dir / 'replay_safety.md'}")
    return 0 if result["default_headline_unchanged"]["match"] else 1


def _render_report(result: dict) -> str:
    lines = ["# A1.8.3 replay safety\n"]
    headline = result["default_headline_unchanged"]
    lines.append(f"Default configuration reproduces the headline aggregate: "
                 f"**{'MATCH' if headline['match'] else 'MISMATCH'}** "
                 f"(training run `{json.dumps(headline['training_run']['computed'])}`)\n")
    gates = result["replay_gates"]
    lines.append("## Replay gates (default off, explicit allowlist)\n")
    lines.append(_md_table(["property", "value"], [
        ["opt-in flag", "`allow_recorded_success_replay=False` (default)"],
        ["allowlist default", "`None` = empty (no tool may be replayed)"],
        ["allowlist used in this experiment", gates["allowlist"]],
        ["allowlist size", gates["allowlist_size"]],
        ["gates", gates["gates"]],
        ["implementation", gates["code"]],
    ]))
    comparison = result["generic_subset_comparison"]
    lines.append("\n## 133 generic missing-parameter failures: replay funnels\n")
    lines.append(_md_table(
        ["replay source", "proposed", "schema-valid", "value-verified", "schema rejected",
         "still no proposal", "candidate cases", "candidate argument sets"],
        [[comparison["training_only"]["label"],
          comparison["training_only"].get("proposed", 0),
          comparison["training_only"].get("schema_valid", 0),
          comparison["training_only"].get("value_verified", 0),
          comparison["training_only"].get("proposed_rejected", 0),
          comparison["training_only"].get("no_proposal", 0),
          comparison["training_only"].get("candidate_cases", 0),
          comparison["training_only"].get("candidate_argument_sets", 0)],
         [comparison["test_fold_oracle_gated"]["label"],
          comparison["test_fold_oracle_gated"].get("proposed", 0),
          comparison["test_fold_oracle_gated"].get("schema_valid", 0),
          comparison["test_fold_oracle_gated"].get("value_verified", 0),
          comparison["test_fold_oracle_gated"].get("proposed_rejected", 0),
          comparison["test_fold_oracle_gated"].get("no_proposal", 0),
          "-", "-"],
         [comparison["historical_unsafe"]["label"],
          comparison["historical_unsafe"].get("proposed", 0),
          comparison["historical_unsafe"].get("schema_valid", 0),
          comparison["historical_unsafe"].get("value_verified", 0),
          comparison["historical_unsafe"].get("proposed_rejected", 0),
          comparison["historical_unsafe"].get("no_proposal", 0),
          "-", "-"]]))
    answer = result["answer"]
    lines.append("\n## Answers\n")
    training = answer["training_only_replay_gain"]
    lines.append(f"- **training-only replay real gain: {training['accepted']} accepted "
                 f"({training['schema_valid']} schema-valid + {training['value_verified']} "
                 f"value-verified, {training['no_proposal']} still no proposal). "
                 f"{training['verdict']}**.")
    historical = answer["historical_oracle_replay"]
    lines.append(f"- historical A1.7 replay admissions: {historical['accepted']} "
                 f"({historical['schema_valid']} schema-valid + "
                 f"{historical['value_verified']} value-verified). {historical['note']}.")
    gated = answer["oracle_under_new_gates"]
    lines.append(f"- the same test-fold oracle candidate source with A1.8.3 gates: "
                 f"{gated['accepted']} accepted ({gated['schema_valid']} schema-valid + "
                 f"{gated['value_verified']} value-verified). {gated['note']}.")
    blocked = result["blocked_by_safety_gates"]
    lines.append("\n## Historical replay admissions blocked by the safety gates\n")
    lines.append(_md_table(["blocked reason", "count"],
                           [[reason, count] for reason, count in blocked.items()]))
    lines.append("\n## Every historical replay admission (13)\n")
    lines.append(_md_table(
        ["fold", "tool", "arguments", "schema properties", "oracle gated status",
         "training-only status", "blocked reason"],
        [[entry["fold"], entry["tool"], json.dumps(entry["arguments"], ensure_ascii=False),
          json.dumps(entry["schema_properties"], ensure_ascii=False),
          entry["oracle_gated_status"], entry["training_only_status"],
          entry.get("blocked_reason", "-")]
         for entry in result["historical_replay_admissions"]]))
    cache = result["cache_verification"]
    lines.append("\n## Cache verification\n")
    lines.append(_md_table(
        ["run", "accepted checked", "cache ok", "cache error", "cache miss"],
        [[name, counts.get("accepted_checked", 0), counts.get("cache_ok", 0),
          counts.get("cache_error", 0), counts.get("cache_miss", 0)]
         for name, counts in cache.items()]))
    lines.append("\n## Subset totals (228 cases, A1.6 schema-only + A1.7 extractors + replay)\n")
    lines.append(_md_table(
        ["run", "accepted", "schema-valid", "value-verified", "no proposal"],
        [[name, counts.get("accepted", 0), counts.get("schema_valid", 0),
          counts.get("value_verified", 0), counts.get("no_proposal", 0)]
         for name, counts in result["total_subset"].items()]))
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
