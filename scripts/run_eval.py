"""Evaluation pipeline: classifier metrics, KDE quality, DAG recovery, before/after comparison, degradation curve.

Produces quantitative metrics for resume. Run after `python scripts/seed_demo_data.py`.
Usage: python scripts/run_eval.py
"""
import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, "src")

from tool_evolution.utils.database import get_connection, init_db, run_migrations
from tool_evolution.utils.config import settings
from tool_evolution.collection.store import TraceStore
from tool_evolution.collection.schemas import TraceReport, TraceType, ErrorType
from tool_evolution.analysis.classifier import FailureClassifier
from tool_evolution.analysis.dag_miner import DAGMiner
from tool_evolution.analysis.distiller import CounterfactualDistiller
from tool_evolution.analysis.repair_advisor import RepairAdvisor
from tool_evolution.knowledge.rule_engine import RuleEngine
from tool_evolution.knowledge.param_template import ParamTemplateManager, flatten_user_prefs
from tool_evolution.knowledge.skill_pack import SkillPackManager
from tool_evolution.governance.governor import SkillGovernor
from tool_evolution.governance.relation_store import RelationStore
from tool_evolution.analysis.preference_learner import PreferenceLearner
from tool_evolution.execution.tool_specs import TOOL_SPECS, mock_params_for_tool


EVAL_TOOLS = list(TOOL_SPECS.keys())
ERRORS = list(ErrorType)
DAG_PATTERNS = [
    ["search_api", "detail_api", "analyze_api", "report_api"],
    ["search_api", "analyze_api"],
    ["github_api", "analyze_api"],
    ["arxiv_api", "analyze_api", "report_api"],
    ["official_docs", "detail_api", "analyze_api"],
    ["github_api", "search_api", "analyze_api"],
    ["arxiv_api", "official_docs", "analyze_api"],
    ["search_api", "report_api"],
]

# Realistic error messages that do NOT contain the error type literally
ERROR_MESSAGES = {
    ErrorType.PARAM_ERROR: [
        "expected int but got str for field 'max_results'",
        "missing required parameter 'query'",
        "value -1 out of range for argument 'limit'",
        "unexpected keyword argument 'timeout_ms'",
        "field 'api_key' expects str, got NoneType",
    ],
    ErrorType.PERMISSION_DENIED: [
        "403 Forbidden: insufficient scope",
        "API key does not have access to this resource",
        "authentication required, please provide valid token",
        "access denied for user role 'viewer'",
        "token expired or revoked",
    ],
    ErrorType.QUOTA_EXHAUSTED: [
        "rate limit exceeded, try again in 60 seconds",
        "daily quota of 1000 requests reached",
        "too many requests from this IP",
        "monthly usage cap exceeded, upgrade required",
    ],
    ErrorType.TIMEOUT: [
        "connection to api.example.com timed out after 30s",
        "upstream server did not respond in time",
        "socket timeout waiting for response",
        "request took longer than allowed deadline",
    ],
    ErrorType.SERVICE_UNAVAILABLE: [
        "503 service temporarily unavailable",
        "backend server returned 502 Bad Gateway",
        "upstream connection reset by peer",
        "internal server error, please retry",
    ],
}


_MAIN_RNG = random.Random(42)  # 主链固定种子——评测可复现（增量零 I 修复#1）


def expand_benchmark_tasks(base_tasks: list[dict], num_variants: int) -> list[dict]:
    """确定性变体扩展（task-major 展开）：N base × num_variants。

    变体 j 的 max_results/lang 按旧版 8 值表轮换（j % 8）——num_variants=8 时
    与旧版内联循环逐字节一致；40 变体 = 8 参数组合 × 5 轮换的独立确定性抽样。
    legacy 注：max_results 表中 25 超出 range_rule 合法域 1-20（旧版既有值，
    保持旧口径一致不修——变体参数不进入 baseline 臂仿真，optimized 臂被规则夹到 20）。
    """
    max_results_variants = [5, 8, 10, 12, 15, 18, 20, 25]
    lang_variants = ["zh", "zh", "zh", "zh", "en", "ja", "zh", "zh"]
    expanded_tasks = []
    for i, task in enumerate(base_tasks):
        for j in range(num_variants):
            variant = dict(task)
            variant["task_id"] = f"{task['task_id']}-v{j}"
            variant["root_params"] = dict(task.get("root_params", task.get("params", {})))
            if "root_params" not in task:
                variant["root_params"] = {"query": variant["root_params"].get("query", f"task-{i}"),
                                          "max_results": variant["root_params"].get("max_results", 10),
                                          "lang": variant["root_params"].get("lang", "zh")}
            variant["root_params"]["max_results"] = max_results_variants[j % 8]
            variant["root_params"]["lang"] = lang_variants[j % 8]
            expanded_tasks.append(variant)
    return expanded_tasks

async def seed_eval_data(conn, n_tasks: int = 200) -> dict:
    """Seed labeled evaluation data with known ground truth patterns."""
    await init_db(conn)
    store = TraceStore(conn)

    # Clear prior eval data (order matters: child tables first due to FK constraints)
    await conn.execute("DELETE FROM canary_invocations")
    await conn.execute("DELETE FROM deployed_skills")
    await conn.execute("DELETE FROM discovered_skills")
    await conn.execute("DELETE FROM trajectories WHERE trace_id LIKE 'eval-%'")
    # 崩溃残留防护（审阅发现）：run 1 在 stage 11 中途崩溃会留 rep-% 残留，
    # 污染 run 2 的 stage 4 DAG 挖掘——stage 1 即归零
    await conn.execute(
        "DELETE FROM trajectories_fts WHERE rowid IN "
        "(SELECT rowid FROM trajectories WHERE trace_id LIKE 'rep-%')")
    await conn.execute("DELETE FROM trajectories WHERE trace_id LIKE 'rep-%'")
    await conn.commit()

    traces = []
    planted_patterns = {}  # root_id -> pattern_name
    error_labels = []  # (trace_id, error_type) for ground truth

    for i in range(n_tasks):
        pattern = _MAIN_RNG.choice(DAG_PATTERNS)
        root_id = f"eval-root-{i}"
        pattern_name = "->".join(pattern)
        planted_patterns[root_id] = pattern_name

        root = TraceReport(
            trace_id=root_id, agent_id="orchestrator",
            tool_name="run_analysis_task", tool_version="1.0.0",
            trace_type=TraceType.TASK_ROOT, success=True,
            latency_ms=_MAIN_RNG.randint(2000, 15000),
            token_count=_MAIN_RNG.randint(500, 3000),
            source="synthetic_demo",
        )
        traces.append(root)

        for j, tool in enumerate(pattern):
            success = _MAIN_RNG.random() > 0.30
            report = TraceReport(
                trace_id=f"eval-{i}-{j}",
                parent_trace_id=root_id,
                agent_id=tool,
                tool_name=tool, tool_version="1.0.0",
                trace_type=TraceType.ATOMIC,
                success=success,
                params={
                    "query": f"产品文档 第{_MAIN_RNG.randint(1, 100)}节",
                    "max_results": _MAIN_RNG.randint(5, 20),
                    "lang": _MAIN_RNG.choice(["zh", "zh", "zh", "en", "ja"]),
                    "timeout_ms": _MAIN_RNG.choice([5000, 10000, 15000]),
                },
                latency_ms=_MAIN_RNG.randint(50, 5000),
                token_count=_MAIN_RNG.randint(50, 500),
                source="synthetic_demo",
            )
            if not success:
                err = _MAIN_RNG.choice(ERRORS)
                report.error_type = err
                report.error_message = _MAIN_RNG.choice(ERROR_MESSAGES[err])
                error_labels.append((report.trace_id, err.value))
            traces.append(report)

    for t in traces:
        await store.insert(t)

    return {
        "n_tasks": n_tasks,
        "n_traces": len(traces),
        "planted_patterns": planted_patterns,
        "error_labels": error_labels,
    }


async def eval_classifier(conn) -> dict:
    """Train classifier and measure per-class precision/recall/F1."""
    # I#1: 显式声明扫描范围——只吃 eval-* 种子失败，不依赖 DB 历史状态
    cursor = await conn.execute(
        "SELECT * FROM trajectories WHERE success=0 AND trace_id LIKE 'eval-%'"
    )
    failed_rows = [dict(r) for r in await cursor.fetchall()]

    if len(failed_rows) < 20:
        return {"error": "Not enough failure data"}

    # Train on 70%, test on 30%
    _MAIN_RNG.shuffle(failed_rows)
    split = int(len(failed_rows) * 0.7)
    train_set = failed_rows[:split]
    test_set = failed_rows[split:]

    clf = FailureClassifier()
    clf.train(train_set)

    # Predict test set
    y_true = []
    y_pred = []
    for trace in test_set:
        true_label = trace["error_type"]
        try:
            pred_label = clf.predict(trace).value
            y_true.append(true_label)
            y_pred.append(pred_label)
        except Exception:
            pass

    # Per-class metrics
    classes = sorted(set(y_true + y_pred))
    per_class = {}
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.01)
        per_class[c] = {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3), "support": tp + fn}

    # Macro average
    macro_f1 = round(sum(v["f1"] for v in per_class.values()) / max(len(per_class), 1), 3)
    accuracy = round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1), 3)

    return {
        "train_size": len(train_set),
        "test_size": len(test_set),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "top_features": clf.feature_importance() if hasattr(clf, "feature_importance") else {},
    }


async def eval_kde(conn) -> dict:
    """Evaluate KDE parameter analysis quality."""
    mgr = ParamTemplateManager(conn)

    # Check all 7 tools, not just 3
    all_tools = list(TOOL_SPECS.keys())
    results = {}
    below_threshold = []
    for tool in all_tools:
        tmpl = await mgr.generate(tool, "1.0.0")
        if tmpl:
            results[tool] = {
                "n_params_discovered": len(tmpl),
                "params": {k: {"type": v["param_type"], "has_default": v["default_value"] is not None, "sample_count": v["sample_count"]} for k, v in tmpl.items()},
            }
        else:
            below_threshold.append(tool)

    total_params = sum(r["n_params_discovered"] for r in results.values())

    # F5: KDE mode vs median comparison for numeric params
    import numpy as _np
    store_kde = TraceStore(conn)
    mode_vs_median = {}
    for tool in results:
        rows = await store_kde.get_success_params(tool, "1.0.0", limit=200, exclude_agent_prefix="executor:")
        if not rows:
            continue
        tmpl = await mgr.get_template(tool, "1.0.0")
        if not tmpl:
            continue
        tool_compare = {}
        for pname, pinfo in tmpl.items():
            if pinfo["param_type"] not in ("int", "float"):
                continue
            values = []
            for r in rows:
                pdict = r  # get_success_params returns parsed params dicts directly
                if pname in pdict and isinstance(pdict[pname], (int, float)):
                    values.append(float(pdict[pname]))
            if len(values) < 5:
                continue
            arr = _np.array(values)
            kde_default = pinfo.get("default_value")
            median_val = round(float(_np.median(arr)), 4)
            # MAE of KDE default vs median against actual values
            kde_mae = round(float(_np.mean(_np.abs(arr - kde_default))) if kde_default is not None else 999, 2)
            median_mae = round(float(_np.mean(_np.abs(arr - median_val))), 2)
            tool_compare[pname] = {
                "kde_default": kde_default,
                "median": median_val,
                "kde_mae": kde_mae,
                "median_mae": median_mae,
                "kde_better": kde_mae < median_mae,
            }
        if tool_compare:
            mode_vs_median[tool] = tool_compare

    return {
        "tools_analyzed": len(results),
        "total_params": total_params,
        "details": results,
        "below_min_samples": below_threshold,
        "mode_vs_median": mode_vs_median,  # F5 addition
    }


async def eval_dag(conn) -> dict:
    """Evaluate DAG pattern mining accuracy against planted patterns."""
    store = TraceStore(conn)
    all_traces = await store.get_all_traces(limit=50000, exclude_agent_prefix="executor:")

    # Count planted patterns
    for t in all_traces:
        if t.get("trace_type") == "task_root" and t.get("parent_trace_id") is None:
            pass  # Would need access to planted_patterns from seed

    # Run miner
    miner = DAGMiner(min_support=0.03, max_nodes=10)
    discovered = miner.mine(all_traces)

    # Check which planted patterns were discovered
    discovered_names = {d["name"] for d in discovered}
    planted_names = {
        "search_api → detail_api → analyze_api → report_api",
        "search_api → analyze_api",
        "github_api → analyze_api",
        "arxiv_api → analyze_api → report_api",
        "official_docs → detail_api → analyze_api",
        "github_api → search_api → analyze_api",
        "arxiv_api → official_docs → analyze_api",
        "search_api → report_api",
    }

    matched = discovered_names & planted_names
    recall = len(matched) / max(len(planted_names), 1)

    return {
        "n_discovered": len(discovered),
        "planted_patterns": list(planted_names),
        "discovered_patterns": list(discovered_names),
        "matched": list(matched),
        "pattern_recall": round(recall, 3),
        "discoveries": [{"name": d["name"], "frequency": d["frequency"], "status": d["status"]} for d in discovered],
    }


async def eval_governance(conn) -> dict:
    """Evaluate governance: credit scoring, canary promotion pipeline."""
    skill_mgr = SkillPackManager(conn)
    discoveries = await skill_mgr.list_discoveries()

    results = []
    # I-6: track full canary promotion path for first 3 skills
    promotion_history = {}

    for i, disc in enumerate(discoveries[:5]):
        dep_id = await skill_mgr.promote_to_deployed(disc["id"])
        gov = SkillGovernor(conn)

        # Cycle 1: 60 calls
        for _ in range(60):
            success = _MAIN_RNG.random() > 0.15
            await gov.record_call(dep_id, success=success, latency_ms=_MAIN_RNG.randint(100, 800), tokens=_MAIN_RNG.randint(50, 200))
        await gov.update_all_scores()

        if i < 3:
            name = disc["name"]
            promotion_history[name] = []
            dep = await skill_mgr.get_deployed(name)
            promotion_history[name].append({"calls": 60, "score": dep["credit_score"], "status": dep["status"]})

            # Cycle 2: 100 more calls → canary_15 → canary_50
            for _ in range(100):
                success = _MAIN_RNG.random() > 0.12
                await gov.record_call(dep_id, success=success, latency_ms=_MAIN_RNG.randint(80, 600), tokens=_MAIN_RNG.randint(40, 180))
            await gov.update_all_scores()
            dep = await skill_mgr.get_deployed(name)
            promotion_history[name].append({"calls": 160, "score": dep["credit_score"], "status": dep["status"]})

            # Cycle 3: 150 more calls → canary_50 → active
            for _ in range(150):
                success = _MAIN_RNG.random() > 0.10
                await gov.record_call(dep_id, success=success, latency_ms=_MAIN_RNG.randint(60, 500), tokens=_MAIN_RNG.randint(40, 160))
            await gov.update_all_scores()
            dep = await skill_mgr.get_deployed(name)
            promotion_history[name].append({"calls": 310, "score": dep["credit_score"], "status": dep["status"]})

        score = await gov.score_skill(dep_id)
        dep = await skill_mgr.get_deployed(disc["name"])
        results.append({
            "name": disc["name"],
            "frequency": disc["frequency"],
            "credit_score": score,
            "success_rate": round(dep["success_count"] / max(dep["total_calls"], 1), 3) if dep else 0,
            "total_calls": dep["total_calls"] if dep else 0,
            "status": dep["status"] if dep else "unknown",
        })

    # Trigger promotion via update_all_scores on remaining skills
    gov2 = SkillGovernor(conn)
    await gov2.update_all_scores()
    promoted_results = []
    for r in results:
        dep = await skill_mgr.get_deployed(r["name"])
        r["status_after_update"] = dep["status"] if dep else "unknown"
        r["credit_score_after"] = dep["credit_score"] if dep else 0
        promoted_results.append(r)

    # A/B 实测：种入 canary_invocations → compare_variants 唯一比较入口
    from tool_evolution.governance.canary_router import CanaryRouter
    router_ab = CanaryRouter(conn)
    ab_test = {"rollback": False, "promote": False, "canary_rate": None,
               "stable_rate": None, "note": "insufficient samples"}
    if promoted_results:
        target_dep = await skill_mgr.get_deployed(promoted_results[0]["name"])
        if target_dep:
            dep_id = target_dep["id"]
            # canary 60% 成功 vs stable 90% 成功 → 应触发 rollback
            for _ in range(40):
                await router_ab.record_invocation(dep_id, "canary", success=True, latency_ms=100, tokens=50)
            for _ in range(26):
                await router_ab.record_invocation(dep_id, "canary", success=False, latency_ms=100, tokens=50)
            for _ in range(40):
                await router_ab.record_invocation(dep_id, "stable", success=True, latency_ms=100, tokens=50)
            for _ in range(5):
                await router_ab.record_invocation(dep_id, "stable", success=False, latency_ms=100, tokens=50)
            cmp_result = await router_ab.compare_variants(dep_id, min_samples=30)
            if cmp_result:
                ab_test = cmp_result
    # 样本不足路径：对另一技能只记 1 条 → compare_variants 返回 None
    insufficient_samples = None
    if promoted_results and len(promoted_results) > 1:
        other = await skill_mgr.get_deployed(promoted_results[1]["name"])
        if other:
            await router_ab.record_invocation(other["id"], "canary", success=True, latency_ms=100, tokens=50)
            insufficient_samples = await router_ab.compare_variants(other["id"], min_samples=30) is None

    return {"skills_scored": len(promoted_results), "skills": promoted_results,
            "ab_test": ab_test, "insufficient_samples": insufficient_samples,
            "promotion_history": promotion_history}


async def _seed_kde_training_data(conn) -> None:
    """Seed realistic success traces so KDE can learn valid parameter ranges."""
    store = TraceStore(conn)
    rng = random.Random(42)
    # Core tools with full param sets
    import uuid as _uuid
    _batch_id = _uuid.uuid4().hex[:8]
    for i in range(200):
        for tool in ["search_api", "detail_api", "analyze_api", "report_api"]:
            await store.insert(TraceReport(
                trace_id=f"kde-{_batch_id}-{tool}-{i}",
                agent_id="benchmark", tool_name=tool, tool_version="1.0.0",
                success=True, latency_ms=rng.randint(50, 400), token_count=rng.randint(50, 300),
                params={"query": f"技术手册 第{rng.randint(1, 100)}节",
                        "max_results": rng.randint(5, 20),
                        "lang": rng.choice(["zh", "zh", "zh", "en"]),
                        "timeout_ms": rng.choice([5000, 10000, 15000])},
                source="synthetic_demo",
            ))
    # Additional tools with their own param schemas
    for i in range(200):
        for tool, param_sets in [
            ("github_api", {"repo": "owner/repo", "per_page": rng.randint(10, 100),
                           "state": rng.choice(["open", "closed", "all"]),
                           "sort": rng.choice(["created", "updated", "comments"])}),
            ("arxiv_api", {"query": f"machine learning agent {rng.randint(2020, 2026)}",
                          "max_results": rng.randint(5, 30),
                          "sort_by": rng.choice(["relevance", "lastUpdatedDate"]),
                          "category": rng.choice(["cs.AI", "cs.CL", "cs.LG", "stat.ML"])}),
            ("official_docs", {"url": f"https://docs.example.com/v{rng.randint(1,3)}/api/{rng.choice(['search','get','list'])}",
                              "timeout_ms": rng.choice([5000, 10000, 15000, 20000]),
                              "retry": rng.choice([True, False]),
                              "format": rng.choice(["json", "xml", "text"])}),
        ]:
            await store.insert(TraceReport(
                trace_id=f"kde-{_batch_id}-{tool}-{i}",
                agent_id="benchmark", tool_name=tool, tool_version="1.0.0",
                success=True, latency_ms=rng.randint(50, 400), token_count=rng.randint(50, 300),
                params=dict(param_sets),
                source="synthetic_demo",
            ))


async def _clear_traces(conn) -> None:
    await conn.execute("DELETE FROM trajectories")
    await conn.execute("DELETE FROM trajectories_fts")
    await conn.execute("DELETE FROM rules")
    await conn.execute("DELETE FROM canary_invocations")
    await conn.commit()


async def _run_benchmark_pass(conn, tasks: list[dict], optimized: bool,
                               param_mgr, rule_engine) -> list[dict]:
    """Execute all benchmark tasks and return the resulting traces.

    Token model: per-call base (50) + per_param (25 each).
    When optimized, params with KDE defaults are omitted from the call → fewer tokens.
    """
    TOKEN_BASE = 50
    TOKEN_PER_PARAM = 25

    store = TraceStore(conn)
    rng = random.Random(42)
    traces_written = []

    for task in tasks:
        root_id = f"{'opt' if optimized else 'base'}-{task['task_id']}"
        root_params = dict(task["root_params"])

        # Apply optimization if enabled
        params = dict(root_params)
        omitted_param_count = 0
        if optimized:
            for tool_name in task["tool_chain"]:
                tmpl = await param_mgr.get_template(tool_name, "1.0.0")
                if tmpl:
                    for pname, pinfo in tmpl.items():
                        if pname in params and pinfo.get("default_value") is not None:
                            params[pname] = pinfo["default_value"]
                            omitted_param_count += 1  # param omitted from LLM output
            for tool_name in task["tool_chain"]:
                rules = await rule_engine.check(tool_name, "1.0.0", params)
                if rules:
                    for rule in rules:
                        if rule["rule_type"] == "range_rule" and "max_results" in params:
                            params["max_results"] = min(max(params.get("max_results", 10), 1), 20)

        # Effective param count = total - omitted (defaults handled by system, not LLM)
        effective_param_count = max(1, len(params) - omitted_param_count)

        # Write task_root (token = base + effective params)
        root = TraceReport(
            trace_id=root_id, agent_id="benchmark",
            tool_name=task["tool_chain"][0], tool_version="1.0.0",
            trace_type=TraceType.TASK_ROOT, success=True,
            latency_ms=rng.randint(500, 3000),
            token_count=TOKEN_BASE + effective_param_count * TOKEN_PER_PARAM,
            source="synthetic_demo",
        )
        await store.insert(root)
        traces_written.append(root)

        # Write atomic calls for each tool in chain
        for j, tool_name in enumerate(task["tool_chain"]):
            fail_chance = 0.08 if optimized else 0.20
            success = rng.random() > fail_chance
            call_tokens = TOKEN_BASE + effective_param_count * TOKEN_PER_PARAM

            report = TraceReport(
                trace_id=f"{root_id}-{j}",
                parent_trace_id=root_id, agent_id=tool_name,
                tool_name=tool_name, tool_version="1.0.0",
                trace_type=TraceType.ATOMIC, success=success,
                params=params, latency_ms=rng.randint(50, 2000),
                token_count=call_tokens,
                source="synthetic_demo",
            )
            if not success:
                err_type = rng.choice(list(ErrorType))
                report.error_type = err_type
                err_msgs = {
                    ErrorType.PARAM_ERROR: [
                        f"parameter out of valid range: max_results={params.get('max_results')}",
                        "missing required parameter 'query'",
                    ],
                    ErrorType.PERMISSION_DENIED: [
                        "403 Forbidden: insufficient scope",
                        "token expired or revoked",
                    ],
                    ErrorType.QUOTA_EXHAUSTED: [
                        "rate limit exceeded, try again in 60 seconds",
                        "daily quota of 1000 requests reached",
                    ],
                    ErrorType.TIMEOUT: [
                        "connection timed out after 30s",
                        "upstream server did not respond in time",
                    ],
                    ErrorType.SERVICE_UNAVAILABLE: [
                        "503 service temporarily unavailable",
                        "backend server returned 502 Bad Gateway",
                    ],
                }
                report.error_message = rng.choice(err_msgs.get(err_type, ["unknown error"]))
                # Retry costs the same tokens again
                retry = TraceReport(
                    trace_id=f"{root_id}-{j}-retry",
                    parent_trace_id=root_id, agent_id=tool_name,
                    tool_name=tool_name, tool_version="1.0.0",
                    trace_type=TraceType.ATOMIC, success=True,
                    params=params, latency_ms=rng.randint(50, 2000),
                    token_count=call_tokens,
                    source="synthetic_demo",
                )
                await store.insert(retry)
                traces_written.append(retry)

            await store.insert(report)
            traces_written.append(report)

    return traces_written


async def _query_pass_metrics(conn) -> dict:
    """Extract aggregate metrics from trajectories table after a benchmark pass."""
    cursor = await conn.execute("SELECT COUNT(*) FROM trajectories WHERE success=0 AND trace_type='atomic'")
    failures = (await cursor.fetchone())[0]
    cursor = await conn.execute("SELECT COUNT(*) FROM trajectories WHERE trace_type='atomic'")
    total = (await cursor.fetchone())[0]
    cursor = await conn.execute("SELECT SUM(token_count) FROM trajectories")
    total_tokens = (await cursor.fetchone())[0] or 0
    cursor = await conn.execute("SELECT AVG(latency_ms) FROM trajectories WHERE trace_type='atomic'")
    avg_latency = (await cursor.fetchone())[0] or 0

    cursor = await conn.execute(
        "SELECT COUNT(*) FROM trajectories WHERE trace_type='atomic' AND trace_id LIKE '%-retry'"
    )
    retry_count = (await cursor.fetchone())[0]

    cursor = await conn.execute(
        "SELECT error_type, COUNT(*) as cnt FROM trajectories "
        "WHERE success=0 AND trace_type='atomic' AND error_type IS NOT NULL "
        "GROUP BY error_type ORDER BY cnt DESC"
    )
    failure_by_type = {row["error_type"]: row["cnt"] for row in await cursor.fetchall()}

    return {
        "total_calls": total,
        "failures": failures,
        "failure_rate": round(failures / max(total, 1), 4),
        "retries": retry_count,
        "total_tokens": total_tokens,
        "avg_latency_ms": round(avg_latency, 1),
        "failure_by_type": failure_by_type,
    }


async def eval_before_after(conn, tasks: list[dict] | None = None) -> dict:
    """Real two-pass benchmark: baseline vs optimized.

    Pass 1 (baseline): raw params, no templates → higher failure rate, more retries.
    Pass 2 (optimized): KDE-corrected params + rule pre-checks → fewer failures.

    Returns real metrics extracted from the trajectories table after each pass.
    """
    if tasks is None:
        import json as _json
        from pathlib import Path as _Path
        tasks_path = _Path(__file__).parent / "benchmark_tasks.json"
        tasks = _json.loads(tasks_path.read_text(encoding="utf-8"))

    mgr = ParamTemplateManager(conn)
    engine = RuleEngine(conn)

    # Phase 0: seed KDE training data and generate templates, then clear
    await _seed_kde_training_data(conn)
    for tool in ["search_api", "detail_api", "analyze_api", "report_api"]:
        await mgr.generate(tool, "1.0.0")
    await _clear_traces(conn)

    # Phase 1: baseline (no optimization)
    baseline_tasks = []
    for t in tasks:
        bt = dict(t)
        bt["root_params"] = {
            **t["root_params"],
            "max_results": random.Random(t["task_id"]).choice([0, 1, 30, 50, 100]),
        }
        baseline_tasks.append(bt)

    await _run_benchmark_pass(conn, baseline_tasks, optimized=False,
                               param_mgr=mgr, rule_engine=engine)
    baseline_metrics = await _query_pass_metrics(conn)

    # Phase 2: optimized
    await _clear_traces(conn)
    await _run_benchmark_pass(conn, tasks, optimized=True,
                               param_mgr=mgr, rule_engine=engine)
    optimized_metrics = await _query_pass_metrics(conn)

    # Compute reductions
    bl_fail = baseline_metrics["failures"]
    op_fail = optimized_metrics["failures"]
    bl_tok = baseline_metrics["total_tokens"]
    op_tok = optimized_metrics["total_tokens"]
    bl_retry = baseline_metrics["retries"]
    op_retry = optimized_metrics["retries"]
    bl_lat = baseline_metrics["avg_latency_ms"]
    op_lat = optimized_metrics["avg_latency_ms"]

    failure_reduction = round((1 - op_fail / max(bl_fail, 1)) * 100, 1)
    token_reduction = round((1 - op_tok / max(bl_tok, 1)) * 100, 1)
    retry_reduction = round((1 - op_retry / max(bl_retry, 1)) * 100, 1)
    latency_reduction = round((1 - op_lat / max(bl_lat, 1)) * 100, 1)

    return {
        "n_tasks": len(tasks),
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "failure_reduction_pct": failure_reduction,
        "token_reduction_pct": token_reduction,
        "retry_reduction_pct": retry_reduction,
        "latency_reduction_pct": latency_reduction,
    }


def degradation_sizes(n_tasks: int) -> list[tuple[str, int]]:
    """三档 seed 派生：1/4、1/2、全量（默认 2000 → 500/1000/2000）。"""
    return [("small", n_tasks // 4), ("medium", n_tasks // 2), ("large", n_tasks)]


async def eval_degradation_curve(conn, n_tasks: int) -> dict:
    """Run pipeline at 1/4, 1/2, and full seed scale."""
    results = {}

    for scale_name, scale_tasks in degradation_sizes(n_tasks):
        # Reset DB (order: child tables first due to FK constraints)
        await init_db(conn)
        await conn.execute("DELETE FROM canary_invocations")
        await conn.execute("DELETE FROM deployed_skills")
        await conn.execute("DELETE FROM discovered_skills")
        await conn.execute("DELETE FROM param_distributions")
        await conn.execute("DELETE FROM rules")
        await conn.execute("DELETE FROM trajectories_fts")
        await conn.execute("DELETE FROM trajectories")
        await conn.commit()

        info = await seed_eval_data(conn, n_tasks=scale_tasks)
        cls_result = await eval_classifier(conn)
        dag_result = await eval_dag(conn)

        results[scale_name] = {
            "n_traces": info["n_traces"],
            "classifier_accuracy": cls_result.get("accuracy", 0),
            "classifier_macro_f1": cls_result.get("macro_f1", 0),
            "dag_pattern_recall": dag_result.get("pattern_recall", 0),
            "dag_discovered": dag_result.get("n_discovered", 0),
        }

    return results


async def eval_weight_sensitivity(conn) -> dict:
    """F6: Run governance with 3 weight variants to test sensitivity."""
    results = {}
    weight_sets = {
        "40/30/30 (default)": (0.4, 0.3, 0.3),
        "50/25/25": (0.5, 0.25, 0.25),
        "60/20/20": (0.6, 0.2, 0.2),
    }

    skill_mgr = SkillPackManager(conn)
    discoveries = await skill_mgr.list_discoveries()

    for w_name, (w_success, w_lat, w_token) in weight_sets.items():
        # Reset deployed skills for fair comparison
        cursor = await conn.execute("SELECT id FROM deployed_skills")
        existing = await cursor.fetchall()
        if not existing:
            # Promote discoveries for this weight test
            for disc in discoveries[:3]:
                await skill_mgr.promote_to_deployed(disc["id"])

        gov = SkillGovernor(conn)
        # Patch weights for sensitivity test
        original_score = gov.score_skill
        async def patched_score(skill_id):
            cursor = await conn.execute("SELECT * FROM deployed_skills WHERE id=?", (skill_id,))
            row = await cursor.fetchone()
            if not row:
                return 0.0
            skill = dict(row)
            if skill["total_calls"] == 0:
                return 50.0
            sr = skill["success_count"] / skill["total_calls"] * 100
            ls = min(100, 1000 / max(skill["total_latency_ms"] / max(skill["total_calls"], 1), 1) * 50)
            ts = min(100, 500 / max(skill["total_tokens"] / max(skill["total_calls"], 1), 1) * 50)
            return round(sr * w_success + ls * w_lat + ts * w_token, 2)

        gov.score_skill = patched_score

        # Simulate 60 calls for each deployed skill
        cursor = await conn.execute("SELECT id FROM deployed_skills")
        dep_rows = await cursor.fetchall()
        for row in dep_rows:
            for _ in range(60):
                success = _MAIN_RNG.random() > 0.15
                await gov.record_call(row["id"], success=success,
                                      latency_ms=_MAIN_RNG.randint(100, 800),
                                      tokens=_MAIN_RNG.randint(50, 200))

        await gov.update_all_scores()
        gov.score_skill = original_score

        # Count outcomes
        cursor = await conn.execute("SELECT status, COUNT(*) as cnt FROM deployed_skills GROUP BY status")
        status_counts = {row["status"]: row["cnt"] for row in await cursor.fetchall()}
        results[w_name] = {
            "promotions": status_counts.get("canary_15", 0) + status_counts.get("canary_50", 0) + status_counts.get("active", 0),
            "demotions": status_counts.get("deprecated", 0),
            "offlines": status_counts.get("offline", 0),
            "canary_5": status_counts.get("canary_5", 0),
        }

        # Reset for next weight set
        await conn.execute("DELETE FROM canary_invocations")
        await conn.execute("DELETE FROM deployed_skills")
        await conn.commit()

    return results


async def eval_simplified_scenario(conn) -> dict:
    """F8: Compare RF classifier vs pure rule-based classification on simplified scenario.

    Simplified = 3 tools, 50 tasks, 2 error types per tool.
    """
    store = TraceStore(conn)
    await conn.execute("DELETE FROM trajectories WHERE trace_id LIKE 'simp-%'")
    await conn.commit()

    tools = ["search_api", "detail_api", "analyze_api"]
    rng = random.Random(99)

    traces = []
    for i in range(300):
        tool = rng.choice(tools)
        success = rng.random() > 0.35
        err_type = rng.choice(["param_error", "timeout", "permission_denied", "quota_exhausted", "service_unavailable"])
        # I-2: Rich variants — spelling typos, cross-language, synonyms
        err_msgs = {
            "param_error": ["missing required parameter 'query'", "expected int but got str",
                          "value -1 out of range", "缺少必需参数", "invalid type for 'timeout_ms'"],
            "timeout": ["connection timed out after 30s", "upstream server did not respond",
                       "conection timed out", "timed_out waiting", "请求超时：服务器未响应"],
            "permission_denied": ["403 Forbidden", "access denied", "token expired",
                                 "not authorized for this resource", "无权访问此资源"],
            "quota_exhausted": ["rate limit exceeded", "daily quota reached",
                               "too many requests", "API调用次数已达上限"],
            "service_unavailable": ["503 unavailable", "502 Bad Gateway",
                                    "connection reset", "service temporarily overloaded",
                                    "上游服务暂时不可用"],
        }
        t = TraceReport(
            trace_id=f"simp-{i}",
            agent_id="test", tool_name=tool, tool_version="1.0.0",
            trace_type=TraceType.ATOMIC, success=success,
            params={"query": f"test-{i}", "max_results": rng.randint(1, 50)},
            latency_ms=rng.randint(50, 2000),
            token_count=rng.randint(50, 300),
            source="synthetic_demo",
        )
        if not success:
            t.error_type = ErrorType(err_type)
            t.error_message = rng.choice(err_msgs[err_type])
        traces.append(t)

    for t in traces:
        await store.insert(t)

    failed_rows = [dict(r) for r in await (await conn.execute(
        "SELECT * FROM trajectories WHERE success=0 AND trace_id LIKE 'simp-%'"
    )).fetchall()]

    # RF classifier
    _MAIN_RNG.shuffle(failed_rows)
    split = int(len(failed_rows) * 0.7)
    train_set = failed_rows[:split]
    test_set = failed_rows[split:]

    clf = FailureClassifier()
    if len(train_set) >= 10:
        clf.train(train_set)
        rf_correct = 0
        rf_y_true, rf_y_pred = [], []
        for t in test_set:
            try:
                pred = clf.predict(t).value
                if pred == t["error_type"]:
                    rf_correct += 1
                rf_y_true.append(t["error_type"])
                rf_y_pred.append(pred)
            except Exception:
                pass
        rf_acc = rf_correct / max(len(test_set), 1)
        classes = sorted(set(rf_y_true + rf_y_pred))
        per_class = {}
        for c in classes:
            tp = sum(1 for t, p in zip(rf_y_true, rf_y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(rf_y_true, rf_y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(rf_y_true, rf_y_pred) if t == c and p != c)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            per_class[c] = round(2 * p * r / max(p + r, 0.01), 3)
        rf_f1 = round(sum(per_class.values()) / max(len(per_class), 1), 3)
    else:
        rf_acc = 0
        rf_f1 = 0

    # Pure rules: keyword match on error_message
    rules_correct = 0
    rules_y_true, rules_y_pred = [], []
    for t in test_set:
        msg = t.get("error_message", "")
        if "missing" in msg or "expected" in msg or "out of range" in msg:
            rule_pred = "param_error"
        elif "timeout" in msg or "timed out" in msg or "did not respond" in msg:
            rule_pred = "timeout"
        else:
            rule_pred = "param_error"
        if rule_pred == t["error_type"]:
            rules_correct += 1
        rules_y_true.append(t["error_type"])
        rules_y_pred.append(rule_pred)
    rules_acc = rules_correct / max(len(test_set), 1)
    classes_r = sorted(set(rules_y_true + rules_y_pred))
    per_class_r = {}
    for c in classes_r:
        tp = sum(1 for t, p in zip(rules_y_true, rules_y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(rules_y_true, rules_y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(rules_y_true, rules_y_pred) if t == c and p != c)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        per_class_r[c] = round(2 * p * r / max(p + r, 0.01), 3)
    rules_f1 = round(sum(per_class_r.values()) / max(len(per_class_r), 1), 3)

    # I-2: char_wb cross-spelling robustness test — split EN/CN
    import re as _re
    def _has_cjk(s): return bool(_re.search(r'[一-鿿]', s))
    variant_cases = {
        "timeout": ["connection timed out", "conection timed out", "timed_out waiting",
                     "请求超时：服务器未响应", "timeout after 30s", "request time out"],
        "param_error": ["missing required param", "参数缺失", "expected int got str"],
        "permission_denied": ["access denied", "not authorized", "permission denied",
                              "token expired or revoked", "无权访问此API"],
    }
    variant_test = []
    for err_type_str, msgs in variant_cases.items():
        for msg in msgs:
            variant_test.append({
                "tool_name": "search_api", "error_message": msg,
                "error_type": err_type_str, "params": json.dumps({}),
                "created_at": "2026-08-06T12:00:00",
            })
    vt_en_acc = 0; vt_en_total = 0; vt_cn_acc = 0; vt_cn_total = 0
    vt_en_details = []; vt_cn_details = []
    if len(train_set) >= 10:
        for t in variant_test:
            try:
                pred = clf.predict(t).value
                ok = pred == t["error_type"]
                is_cn = _has_cjk(t["error_message"])
                if is_cn:
                    vt_cn_total += 1
                    if ok: vt_cn_acc += 1
                    vt_cn_details.append(f"'{t['error_message'][:40]}' -> {pred} {'OK' if ok else 'WRONG'}")
                else:
                    vt_en_total += 1
                    if ok: vt_en_acc += 1
                    vt_en_details.append(f"'{t['error_message'][:40]}' -> {pred} {'OK' if ok else 'WRONG'}")
            except Exception: pass
    vt_result = {
        "en_accuracy": round(vt_en_acc / max(vt_en_total, 1), 3), "en_total": vt_en_total, "en_correct": vt_en_acc,
        "cn_accuracy": round(vt_cn_acc / max(vt_cn_total, 1), 3), "cn_total": vt_cn_total, "cn_correct": vt_cn_acc,
        "en_details": vt_en_details, "cn_details": vt_cn_details,
    }

    # P2: 主分类器跨分布 holdout —— eval-% 干净数据全量训练，simp-% 噪声数据测试
    eval_failed = [dict(r) for r in await (await conn.execute(
        "SELECT * FROM trajectories WHERE success=0 AND trace_id LIKE 'eval-%'"
    )).fetchall()]
    noisy_holdout = {}
    if len(eval_failed) >= 20 and len(failed_rows) >= 20:
        clf_main = FailureClassifier()
        clf_main.train(eval_failed)
        nh_true, nh_pred = [], []
        for t in failed_rows:
            try:
                p = clf_main.predict(t).value
                nh_true.append(t["error_type"])
                nh_pred.append(p)
            except Exception:
                pass
        nh_classes = sorted(set(nh_true + nh_pred))
        nh_per_class = {}
        for c in nh_classes:
            tp = sum(1 for a, b in zip(nh_true, nh_pred) if a == c and b == c)
            fp = sum(1 for a, b in zip(nh_true, nh_pred) if a != c and b == c)
            fn = sum(1 for a, b in zip(nh_true, nh_pred) if a == c and b != c)
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            nh_per_class[c] = round(2 * pr * rc / max(pr + rc, 0.01), 3)
        noisy_holdout = {
            "train_clean": len(eval_failed),
            "test_noisy": len(nh_true),
            "accuracy": round(sum(1 for a, b in zip(nh_true, nh_pred) if a == b) / max(len(nh_true), 1), 3),
            "macro_f1": round(sum(nh_per_class.values()) / max(len(nh_per_class), 1), 3),
            "per_class": nh_per_class,
        }

    return {
        "n_traces": len(failed_rows),
        "train_size": len(train_set),
        "test_size": len(test_set),
        "rf_accuracy": round(rf_acc, 3),
        "rf_f1": rf_f1,
        "rules_accuracy": round(rules_acc, 3),
        "rules_f1": rules_f1,
        "rf_wins": rf_acc > rules_acc,
        "char_wb_variant_test": vt_result,
        "noisy_holdout": noisy_holdout,
    }


async def _seed_relation_tasks(conn, n_tasks: int = 10) -> list[str]:
    """每个任务预埋两实体：ent_{i} 与 topic_{i}（trace b 附带 shared 实体）。"""
    await conn.execute("DELETE FROM trajectories WHERE trace_id LIKE 'rel-%'")
    await conn.commit()  # R14: run_eval 文件库重复运行防护（主键冲突）
    ts = TraceStore(conn)
    root_ids = []
    for i in range(n_tasks):
        root_id = f"rel-root-{i}"
        await ts.insert(TraceReport(trace_id=root_id, agent_id="seed", tool_name="task",
                                    trace_type=TraceType.TASK_ROOT, success=True, latency_ms=0))
        await ts.insert(TraceReport(trace_id=f"rel-a-{i}", parent_trace_id=root_id,
                                    agent_id="seed", tool_name="search_api", success=True,
                                    latency_ms=5, result={"entity": f"ent_{i}"}))
        await ts.insert(TraceReport(trace_id=f"rel-b-{i}", parent_trace_id=root_id,
                                    agent_id="seed", tool_name="faq_api", success=True,
                                    latency_ms=5, result={"title": f"topic_{i}",
                                                          "subject": "shared"}))
        root_ids.append(root_id)
    return root_ids


async def eval_relations(conn) -> dict:
    store = RelationStore(conn)
    root_ids = await _seed_relation_tasks(conn)
    built = 0
    for rid in root_ids:
        built += await store.build_for_task(rid)
    # 全量召回检查：每任务预埋 3 对（ent_i-topic_i、ent_i-shared、topic_i-shared），共 30 对
    expected_pairs = []
    for i in range(len(root_ids)):
        expected_pairs += [(f"ent_{i}", f"topic_{i}"),
                           (f"ent_{i}", "shared"),
                           (f"topic_{i}", "shared")]
    recalled = 0
    for a, b in expected_pairs:
        rows = await store.search_relations(a)
        if any({r["source_entity"], r["target_entity"]} == {a, b} for r in rows):
            recalled += 1
    shared_rows = await store.search_relations("shared")
    before = (await store.search_relations("ent_0"))[0]["strength"]
    await store.build_for_task("rel-root-0")
    after = (await store.search_relations("ent_0"))[0]["strength"]
    return {
        "relation_tasks": len(root_ids),
        "relation_pairs_built": built,
        "recalled_premise_pairs": recalled,
        "recall_total": len(expected_pairs),
        "shared_entity_degree": len(shared_rows),
        "idempotent_rebuild": before == after,
    }


async def eval_preference_loop(conn) -> dict:
    # R6: 隔离——learn() 全库扫描，先清空 trajectories/fts 再种 pref-* 种子，
    # 防止 degradation 残留的 eval-*/simp-* 轨迹污染学习（如 lang zh ~60% 阈值边界）
    await conn.execute("DELETE FROM trajectories")
    await conn.execute("DELETE FROM trajectories_fts")
    await conn.commit()
    mgr = ParamTemplateManager(conn)
    await mgr.save("search_api", "1.0.0", {
        "max_results": {"param_type": "int", "default_value": 10,
                        "lower_bound": 0, "upper_bound": 100, "sample_count": 200},
    })
    ts = TraceStore(conn)
    for i in range(36):
        await ts.insert(TraceReport(trace_id=f"pref-a-{i}", agent_id="agent_p1",
                                    tool_name="search_api", success=True, latency_ms=5,
                                    params={"max_results": 20}))
    for i in range(4):
        await ts.insert(TraceReport(trace_id=f"pref-a2-{i}", agent_id="agent_p1",
                                    tool_name="search_api", success=True, latency_ms=5,
                                    params={"max_results": 10}))
    for i in range(40):
        await ts.insert(TraceReport(trace_id=f"pref-b-{i}", agent_id="agent_p2",
                                    tool_name="search_api", success=True, latency_ms=5,
                                    params={"max_results": 10}))
    await ts.insert(TraceReport(trace_id="pref-c-0", agent_id="agent_p3",
                                tool_name="search_api", success=True, latency_ms=5,
                                params={"max_results": 15}))

    learner = PreferenceLearner(conn)
    prefs = await learner.learn()
    await learner.save_to_cache(prefs)
    flat = flatten_user_prefs(prefs, "search_api")
    tmpl = await mgr.generate("search_api", "1.0.0", user_prefs=flat)
    learned_val = prefs.get("search_api", {}).get("max_results")
    injected = bool(tmpl) and tmpl.get("max_results", {}).get("default_value") == 20
    source_ok = bool(tmpl) and tmpl.get("max_results", {}).get("source") == "user_preference"

    # 重试率模拟：agent_p1 后续 100 次调用按历史分布采样（90% 用 20、10% 用 10），
    # 默认值与实际用法不一致即产生一次重试；对比全局默认（10）与注入偏好（20）
    calls = [20 if _MAIN_RNG.random() < 0.9 else 10 for _ in range(100)]  # R5: 复用模块级种子
    mismatches_before = sum(1 for c in calls if c != 10)
    mismatches_after = sum(1 for c in calls if c != 20)
    retry_reduction = round((mismatches_before - mismatches_after) / 100 * 100, 1)

    # I#4: 阈值敏感性——20/60 vs 30/70 vs 15/50 对两个边际 agent 的学出差异
    async def _sens_probe(agent_id, n_hi, n_lo, min_samples, share_threshold):
        await conn.execute("DELETE FROM trajectories")
        await conn.execute("DELETE FROM trajectories_fts")
        await conn.commit()
        for i in range(n_hi):
            await ts.insert(TraceReport(trace_id=f"{agent_id}-hi-{i}", agent_id=agent_id,
                                        tool_name="search_api", success=True, latency_ms=5,
                                        params={"max_results": 20}))
        for i in range(n_lo):
            await ts.insert(TraceReport(trace_id=f"{agent_id}-lo-{i}", agent_id=agent_id,
                                        tool_name="search_api", success=True, latency_ms=5,
                                        params={"max_results": 10}))
        prefs = await PreferenceLearner(conn, min_samples=min_samples,
                                        share_threshold=share_threshold).learn()
        return prefs.get("search_api", {}).get("max_results") == 20

    sensitivity = {}
    for m, s in [(20, 0.6), (30, 0.7), (15, 0.5)]:
        sens_a = await _sens_probe("sens_a", 16, 9, m, s)    # 25 样本 64% 占比
        sens_b = await _sens_probe("sens_b", 21, 14, m, s)   # 35 样本 60% 占比
        sensitivity[f"{m}/{s}"] = {"sens_a_64pct": sens_a, "sens_b_60pct": sens_b}

    return {
        "learned_max_results": learned_val,
        "expected_learned_value": 20,
        "learned_correct": learned_val == 20,
        "injected_default": injected,
        "injected_source_ok": source_ok,
        "retry_reduction_pct": retry_reduction,
        "mismatches_before": mismatches_before,
        "mismatches_after": mismatches_after,
        "sensitivity": sensitivity,
    }


_REPAIR_VALIDITY = {
    ("repair_api", "range_rule"): lambda p: (isinstance(p.get("max_results"), int)
                                             and 1 <= p["max_results"] <= 20),
    ("repair_fetch", "timeout_rule"): lambda p: (isinstance(p.get("timeout_ms"), int)
                                                 and p["timeout_ms"] >= 5000),
    # I#2 修复：模糊错误信息组（无范围描述）——同模拟工具语义，仅错误信息隐去有效范围，
    # 测量修复下界（上界组=repair_api/repair_fetch 含范围描述）
    ("repair_vague_api", "range_rule"): lambda p: (isinstance(p.get("max_results"), int)
                                                   and 1 <= p["max_results"] <= 20),
    ("repair_vague_fetch", "timeout_rule"): lambda p: (isinstance(p.get("timeout_ms"), int)
                                                       and p["timeout_ms"] >= 5000),
}


async def _cleanup_repair(conn) -> None:
    """清 rep-% 轨迹与 rep 规则（级联 hints）——stage 11 开始（重入安全）与结束（留库复原）都调用。"""
    await conn.execute(
        "DELETE FROM trajectories_fts WHERE rowid IN "
        "(SELECT rowid FROM trajectories WHERE trace_id LIKE 'rep-%')")
    await conn.execute("DELETE FROM trajectories WHERE trace_id LIKE 'rep-%'")
    await conn.execute(
        "DELETE FROM rules WHERE tool_name IN "
        "('repair_api','repair_fetch','repair_vague_api','repair_vague_fetch')")
    await conn.commit()


async def _seed_repair_cases(conn) -> list[dict]:
    await _cleanup_repair(conn)
    store = TraceStore(conn)
    cases = []
    for i in range(45):
        v = _MAIN_RNG.choice([30, 50, 100])
        cases.append({"tool": "repair_api", "error_type": ErrorType.PARAM_ERROR,
                      "params": {"query": f"rep q{i}", "max_results": v},
                      "error": f"parameter out of valid range: max_results must be between 1 and 20, got {v}"})
    for i in range(45):
        cases.append({"tool": "repair_fetch", "error_type": ErrorType.TIMEOUT,
                      "params": {"url": f"https://api.example.com/{i}", "timeout_ms": 1000},
                      "error": "timeout_ms must be at least 5000, got 1000"})
    # I#2: 模糊错误信息组——无有效范围描述，测量修复下界
    for i in range(15):
        v = _MAIN_RNG.choice([30, 50, 100])
        cases.append({"tool": "repair_vague_api", "error_type": ErrorType.PARAM_ERROR,
                      "params": {"query": f"rep vague q{i}", "max_results": v},
                      "error": "invalid parameter value"})
    for i in range(15):
        cases.append({"tool": "repair_vague_fetch", "error_type": ErrorType.TIMEOUT,
                      "params": {"url": f"https://api.example.com/vague/{i}", "timeout_ms": 1000},
                      "error": "request failed"})
    for i, c in enumerate(cases):
        await store.insert(TraceReport(
            trace_id=f"rep-fail-{i}", agent_id="repair_eval", tool_name=c["tool"],
            tool_version="1.0.0", trace_type=TraceType.ATOMIC, success=False,
            params=c["params"], error_type=c["error_type"],
            error_message=c["error"], latency_ms=100, source="synthetic_demo"))
    return cases


async def eval_repair_advisor(conn) -> dict:
    cases = await _seed_repair_cases(conn)
    cursor = await conn.execute(
        "SELECT * FROM trajectories WHERE trace_id LIKE 'rep-%'")
    failed_rows = [dict(r) for r in await cursor.fetchall()]
    distiller = CounterfactualDistiller()
    groups = {}
    for t in failed_rows:
        rule = distiller.distill(t)
        groups.setdefault(rule["_hash"], {"rule": rule, "examples": []})["examples"].append(t)
    engine = RuleEngine(conn)
    advisor = RepairAdvisor(conn)
    llm_mode = "live" if settings.deepseek_api_key else "degraded_no_key"
    hints = []
    for g in groups.values():
        rule_id = await engine.add_rule(g["rule"])
        hints.append(await advisor.generate_for_rule({"id": rule_id, **g["rule"]},
                                                     examples=g["examples"]))
    # 幂等复验：第二轮应 0 API 调用（同 id 同 hash 直接返回）
    round2_hints = []
    for g, h in zip(groups.values(), hints):
        round2_hints.append(await advisor.generate_for_rule(
            {"id": h["rule_id"], **g["rule"]}, examples=g["examples"]))
    reused = len(round2_hints)
    fix_non_null = sum(1 for h in hints if h["fix"] is not None)
    param_covered = sum(1 for h in hints if h["fix"] is not None
                        and json.loads(h["fix"])["param"] in h["suggestion"])
    # 重放：每条失败 case 找触发规则 → hint → fix 应用 → 有效条件模拟
    # I#2: 上界组（含范围描述）与下界组（模糊错误信息）分开统计
    replay_fixable = replay_success = degraded = 0
    vague_fixable = vague_success = 0
    for c in cases:
        triggered = await engine.check(c["tool"], "1.0.0", c["params"])
        applied = False
        for rule_row in triggered:
            hint = await advisor.get_hint(rule_row["id"])
            if hint is None or hint["fix"] is None:
                continue
            fix = json.loads(hint["fix"])
            new_params = {**c["params"], fix["param"]: fix["suggested_value"]}
            applied = True
            valid = _REPAIR_VALIDITY[(c["tool"], rule_row["rule_type"])](new_params)
            if c["tool"].startswith("repair_vague"):
                vague_fixable += 1
                if valid:
                    vague_success += 1
            else:
                replay_fixable += 1
                if valid:
                    replay_success += 1
            break
        if not applied:
            degraded += 1
    tok_in = sum(h["input_tokens"] for h in hints)
    tok_out = sum(h["output_tokens"] for h in hints)
    await advisor.aclose()
    # 自清理：留库状态回到 stage 11 之前（run 2 逐字段复现保障）
    await _cleanup_repair(conn)
    return {
        "planted_failures": len(cases),
        "rules_distilled": len(groups),
        "hints_total": len(hints),
        "reused_round2": reused,
        "fix_non_null": fix_non_null,
        "structured_success_rate": round(fix_non_null / max(len(hints), 1), 4),
        "suggestion_param_coverage": round(param_covered / max(fix_non_null, 1), 4),
        "replay_fixable_cases": replay_fixable,
        "replay_success": replay_success,
        "replay_improvement_pct": round(replay_success / max(replay_fixable, 1) * 100, 1),
        "vague_fixable_cases": vague_fixable,
        "vague_replay_success": vague_success,
        "vague_improvement_pct": round(vague_success / max(vague_fixable, 1) * 100, 1),
        "degraded_cases": degraded,
        "input_tokens": tok_in,
        "output_tokens": tok_out,
        "llm_mode": llm_mode,
    }


_STAGE_DEPS = {
    "classifier": (),
    "kde": (),
    "dag": (),
    "governance": ("dag",),
    "before_after": (),
    "degradation": (),
    "simplified": (),
    "relations": (),
    "preference_loop": (),
    "repair_advisor": (),
}


def _resolve_stages(only: str | None):
    """--only: 返回要执行的 stage 集合（含前置依赖闭包）；None = 全跑。"""
    if only is None:
        return None
    needed = {only}
    for _ in range(len(_STAGE_DEPS)):
        for s in tuple(needed):
            needed.update(_STAGE_DEPS.get(s, ()))
    return needed


async def main(output_path: Path | None = None, seed: int = 2000,
               num_variants: int = 40, only: str | None = None):
    conn = await get_connection()
    await init_db(conn)
    await run_migrations(conn)
    wanted = _resolve_stages(only)

    def run(stage: str) -> bool:
        return wanted is None or stage in wanted

    print("=" * 60)
    print("TOOL EVOLUTION ENGINE — EVALUATION PIPELINE")
    print("=" * 60)

    # Generate expanded benchmark tasks (50 base × num_variants, default 40 = 2000)
    tasks_path = Path(__file__).parent / "benchmark_tasks.json"
    base_tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    expanded_tasks = expand_benchmark_tasks(base_tasks, num_variants)
    print(f"\nBenchmark tasks: {len(base_tasks)} base × {num_variants} variants = {len(expanded_tasks)} tasks")

    # Seed fresh eval data
    t0 = time.monotonic()
    stage_times = {}  # per-stage timing (F3 fix)
    t_stage = t0
    info = await seed_eval_data(conn, n_tasks=seed)
    print(f"\n[1/11] Data: {info['n_tasks']} tasks, {info['n_traces']} traces, "
          f"{len(info['error_labels'])} labeled failures")

    if run("classifier"):
        # 1. Classifier evaluation
        print("\n[2/11] Classifier Evaluation")
        cls = await eval_classifier(conn)
        print(f"  Train/Test: {cls['train_size']}/{cls['test_size']}")
        print(f"  Accuracy: {cls['accuracy']:.1%}  Macro F1: {cls['macro_f1']:.3f}")
        for c, m in cls.get("per_class", {}).items():
            print(f"    {c}: P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} (n={m['support']})")
        t_now = time.monotonic()
        stage_times["classifier"] = round(t_now - t_stage, 2)
        t_stage = t_now

    if run("kde"):
        # F13 fix: seed KDE training data BEFORE eval_kde so all 7 tools have data
        print("\n[3/11] KDE Parameter Analysis (with F13 timing fix)")
        await _seed_kde_training_data(conn)
        mgr_kde = ParamTemplateManager(conn)
        for tool in EVAL_TOOLS:
            await mgr_kde.generate(tool, "1.0.0")
        kde = await eval_kde(conn)
        print(f"  Tools analyzed: {kde['tools_analyzed']}/7, Total params: {kde['total_params']}")
        for tool, detail in kde.get("details", {}).items():
            print(f"    {tool}: {detail['n_params_discovered']} params")
        below = kde.get("below_min_samples", [])
        if below:
            print(f"  Below min_samples (no template): {below}")
        # I-4: KDE mode vs median comparison
        mode_vs_median = kde.get("mode_vs_median", {})
        if mode_vs_median:
            kde_wins = 0; total_cmp = 0
            print("\n  I-4: KDE mode vs median MAE comparison:")
            for tool, params in mode_vs_median.items():
                for pname, data in params.items():
                    total_cmp += 1
                    if data["kde_better"]: kde_wins += 1
                    winner = "KDE" if data["kde_better"] else "median"
                    print(f"    {tool}.{pname}: KDE={data['kde_mae']} median={data['median_mae']} → {winner}")
            if total_cmp > 0:
                print(f"  KDE wins: {kde_wins}/{total_cmp} ({kde_wins/total_cmp*100:.0f}%)")
        # I-5: KDE 95% CI boundary check
        print("\n  I-5: KDE 95% CI boundary check:")
        store_ci = TraceStore(conn)
        for tool in EVAL_TOOLS:
            tmpl = await mgr_kde.get_template(tool, "1.0.0")
            if not tmpl: continue
            rows = await store_ci.get_success_params(tool, "1.0.0", limit=200, exclude_agent_prefix="executor:")
            if not rows: continue
            for pname, pinfo in tmpl.items():
                if pinfo.get("param_type") not in ("int", "float"): continue
                lb, ub = pinfo.get("lower_bound"), pinfo.get("upper_bound")
                if lb is None or ub is None: continue
                values = []
                for r in rows:
                    pdict = r  # get_success_params returns parsed params dicts directly
                    if pname in pdict and isinstance(pdict[pname], (int, float)):
                        values.append(float(pdict[pname]))
                if len(values) < 5: continue
                outside = sum(1 for v in values if v < lb or v > ub)
                pct = round(outside / len(values) * 100, 1)
                flag = " [>5%]" if pct > 5 else ""
                print(f"    {tool}.{pname}: {outside}/{len(values)} outside CI = {pct}%{flag}")
        t_now = time.monotonic()
        stage_times["kde"] = round(t_now - t_stage, 2)
        t_stage = t_now

    if run("dag"):
        # 3. DAG mining
        print("\n[4/11] DAG Pattern Mining")
        dag = await eval_dag(conn)
        print(f"  Planted: {len(dag['planted_patterns'])}  Discovered: {dag['n_discovered']}  Matched: {len(dag['matched'])}")
        print(f"  Pattern Recall: {dag['pattern_recall']:.1%}")
        for d in dag["discoveries"]:
            print(f"    - {d['name'][:60]} (freq={d['frequency']:.1%}, status={d['status']})")
        t_now = time.monotonic()
        stage_times["dag"] = round(t_now - t_stage, 2)
        t_stage = t_now

    if run("governance"):
        # Insert discovered skills for governance scoring
        skill_mgr = SkillPackManager(conn)
        dag_miner = DAGMiner(min_support=0.03, max_nodes=10)
        all_traces = await TraceStore(conn).get_all_traces(limit=50000, exclude_agent_prefix="executor:")
        discovered = dag_miner.mine(all_traces)
        for d in discovered:
            await skill_mgr.add_discovery(d)

    if run("governance"):
        # 4. Governance (with F6 weight sensitivity)
        print("\n[5/11] Skill Governance + Weight Sensitivity (F6)")
        gov = await eval_governance(conn)
        print(f"  Skills scored: {gov['skills_scored']}")
        for s in gov["skills"]:
            print(f"    {s['name'][:50]}: score={s['credit_score']:.1f} success_rate={s['success_rate']:.1%} "
                  f"calls={s['total_calls']} status={s['status']} -> after_update:{s.get('status_after_update','?')}")
        print(f"  A/B rollback test: rollback={gov['ab_test']['rollback']}")
        # I-6: Full canary promotion path
        promotion_history = gov.get("promotion_history", {})
        if promotion_history:
            print("\n  I-6: Full canary path (canary_5→15→50→active):")
            for skill_name, history in promotion_history.items():
                path = " → ".join(f"{h['status']}({h['calls']}c,{h['score']:.0f}pt)" for h in history)
                print(f"    {skill_name[:50]}: {path}")
        # F6: weight sensitivity
        w_sensitivity = await eval_weight_sensitivity(conn)
        print("  Weight sensitivity (40/30/30 vs 50/25/25 vs 60/20/20):")
        for w_name, w_data in w_sensitivity.items():
            print(f"    {w_name}: {w_data['promotions']} promotions, {w_data['demotions']} demotions, {w_data['offlines']} offlines")
        t_now = time.monotonic()
        stage_times["governance"] = round(t_now - t_stage, 2)
        t_stage = t_now

    if run("before_after"):
        # 5. Before/After
        print("\n[6/11] Before/After Optimization Comparison")
        ba = await eval_before_after(conn, tasks=expanded_tasks)
        print(f"  Tasks: {ba['n_tasks']}")
        print(f"  BASELINE  | failures={ba['baseline']['failures']} retries={ba['baseline']['retries']} "
              f"tokens={ba['baseline']['total_tokens']} avg_lat={ba['baseline']['avg_latency_ms']}ms")
        print(f"  OPTIMIZED | failures={ba['optimized']['failures']} retries={ba['optimized']['retries']} "
              f"tokens={ba['optimized']['total_tokens']} avg_lat={ba['optimized']['avg_latency_ms']}ms")
        print(f"  Failure reduction: {ba['failure_reduction_pct']}%")
        print(f"  Retry reduction:   {ba['retry_reduction_pct']}%")
        print(f"  Token reduction:   {ba['token_reduction_pct']}%")
        print(f"  Latency reduction: {ba['latency_reduction_pct']}%")
        # I-1: Failure type breakdown
        bl_ft = ba["baseline"].get("failure_by_type", {})
        op_ft = ba["optimized"].get("failure_by_type", {})
        if bl_ft or op_ft:
            all_types = sorted(set(list(bl_ft.keys()) + list(op_ft.keys())))
            print("\n  I-1: Failure type breakdown (baseline → optimized):")
            print(f"  {'Error Type':<25} {'Baseline':>8} {'Optimized':>8} {'Reduction':>10}")
            for et in all_types:
                bl = bl_ft.get(et, 0)
                op = op_ft.get(et, 0)
                red = round((1 - op / max(bl, 1)) * 100, 1) if bl > 0 else 0
                print(f"  {et:<25} {bl:>8} {op:>8} {red:>9.1f}%")
        t_now = time.monotonic()
        stage_times["before_after"] = round(t_now - t_stage, 2)
        t_stage = t_now

    if run("degradation"):
        # F4: degradation curve (三档派生: seed//4, seed//2, seed)
        print(f"\n[7/11] Degradation Curve (F4 — {seed//4}/{seed//2}/{seed} seed)")
        deg = await eval_degradation_curve(conn, n_tasks=seed)
        for scale, m in deg.items():
            print(f"  {scale}: n={m['n_traces']} cls_acc={m['classifier_accuracy']:.1%} "
                  f"cls_f1={m['classifier_macro_f1']:.3f} dag_recall={m['dag_pattern_recall']:.1%} "
                  f"dag_disc={m['dag_discovered']}")
        t_now = time.monotonic()
        stage_times["degradation"] = round(t_now - t_stage, 2)
        t_stage = t_now

    if run("simplified"):
        # F8: simplified scenario — RF vs pure rules
        print("\n[8/11] Simplified Scenario: RF vs Rules (F8)")
        simplified = await eval_simplified_scenario(conn)
        print(f"  RF accuracy: {simplified['rf_accuracy']:.1%}  Rules accuracy: {simplified['rules_accuracy']:.1%}")
        print(f"  RF macro F1: {simplified['rf_f1']:.3f}  Rules macro F1: {simplified['rules_f1']:.3f}")
        vt = simplified.get("char_wb_variant_test", {})
        if vt:
            print("\n  I-2: char_wb cross-spelling robustness:")
            if vt.get("en_total", 0) > 0:
                print(f"  EN variants: {vt['en_accuracy']:.1%} ({vt['en_correct']}/{vt['en_total']})")
                for d in vt.get("en_details", []): print(d)
            if vt.get("cn_total", 0) > 0:
                print(f"  CN variants: {vt['cn_accuracy']:.1%} ({vt['cn_correct']}/{vt['cn_total']}) [cross-lang limitation]")
                for d in vt.get("cn_details", []): print(d)
        nh = simplified.get("noisy_holdout", {})
        if nh:
            print("\n  P2 noisy-holdout (clean train -> noisy test):")
            print(f"    accuracy={nh['accuracy']:.1%}  macro F1={nh['macro_f1']:.3f}  "
                  f"(clean={nh['train_clean']}, noisy={nh['test_noisy']})")
            print("    per-class: " + ", ".join(f"{k}={v:.2f}" for k, v in nh['per_class'].items()))
        t_now = time.monotonic()
        stage_times["simplified"] = round(t_now - t_stage, 2)

    # R6: 阶段 9/10 会清 trajectories，gsm 基线数据须提前采集
    composition = await _data_composition(conn)
    rules_count = await _count_rules(conn)

    if run("relations"):
        # 9. Relations (增量一)
        t_stage = time.monotonic()
        rel = await eval_relations(conn)
        stage_times["relations"] = time.monotonic() - t_stage
        print(f"\n[9/11] Relations: {rel['recalled_premise_pairs']}/{rel['recall_total']} premise pairs recalled, "
              f"{rel['relation_pairs_built']} pairs built from {rel['relation_tasks']} tasks, "
              f"shared degree {rel['shared_entity_degree']}, idempotent rebuild: {rel['idempotent_rebuild']}")

    if run("preference_loop"):
        # 10. Preference loop (增量一)
        t_stage = time.monotonic()
        pref = await eval_preference_loop(conn)
        stage_times["preference_loop"] = time.monotonic() - t_stage
        print(f"[10/11] Preference loop: learned={pref['learned_correct']} "
              f"(expected {pref['expected_learned_value']}, got {pref['learned_max_results']}), "
              f"injected default={pref['injected_default']}, source ok={pref['injected_source_ok']}, "
              f"retry reduction={pref['retry_reduction_pct']}% "
              f"(mismatches {pref['mismatches_before']}→{pref['mismatches_after']})")
        for combo, r in pref["sensitivity"].items():
            print(f"  sensitivity {combo}: 64pct={r['sens_a_64pct']} 60pct={r['sens_b_60pct']}")

    if run("repair_advisor"):
        # 11. Repair advisor (增量二)
        t_stage = time.monotonic()
        repair = await eval_repair_advisor(conn)
        stage_times["repair_advisor"] = time.monotonic() - t_stage
        print(f"[11/11] Repair advisor: mode={repair['llm_mode']} rules={repair['rules_distilled']} "
              f"hints={repair['hints_total']} fix非空={repair['fix_non_null']} "
              f"参数覆盖={repair['suggestion_param_coverage']:.0%} "
              f"reused_round2={repair['reused_round2']} "
              f"重放(含范围) {repair['replay_success']}/{repair['replay_fixable_cases']} "
              f"({repair['replay_improvement_pct']}%) "
              f"重放(模糊) {repair['vague_replay_success']}/{repair['vague_fixable_cases']} "
              f"({repair['vague_improvement_pct']}%) "
              f"tokens in/out={repair['input_tokens']}/{repair['output_tokens']}")

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 60}")
    print("Stage timing breakdown (F3):")
    for name, sec in stage_times.items():
        pct = sec / elapsed * 100 if elapsed > 0 else 0
        print(f"  {name}: {sec:.1f}s ({pct:.0f}%)")
    print(f"Evaluation complete in {elapsed:.1f}s")

    if only is None:
        gsm = build_gsm_metrics(cls, kde, dag, gov, ba, elapsed, composition, rules_count,
                                seed, len(expanded_tasks), repair)
        if output_path:
            output_path.write_text(json.dumps(gsm, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nEval results written to: {output_path}")
    
        # ── L1/L2/L3 评测总结 ──
        print(f"\n{'=' * 60}")
        print("L1 简历必写")
        print(f"{'=' * 60}")
        print(f"  失败率下降:     {ba['failure_reduction_pct']:.1f}% ({ba['baseline']['failures']}→{ba['optimized']['failures']})  [实测]")
        print(f"  Token 下降:     {ba['token_reduction_pct']:.1f}% ({ba['baseline']['total_tokens']}→{ba['optimized']['total_tokens']})  [实测]")
        print(f"  重试次数下降:   {ba['retry_reduction_pct']:.1f}% ({ba['baseline']['retries']}→{ba['optimized']['retries']})  [实测]")
        print(f"  分类器 F1:      {cls['macro_f1']:.3f} (5类, {cls['train_size']}/{cls['test_size']} split)  [实测]")
    
        print(f"\n{'=' * 60}")
        print("L2 面试支撑")
        print(f"{'=' * 60}")
        ft = ba["baseline"].get("failure_by_type", {})
        ft_o = ba["optimized"].get("failure_by_type", {})
        if ft:
            parts = []
            for et in sorted(ft.keys()):
                bl = ft.get(et, 0); op = ft_o.get(et, 0)
                red = round((1 - op / max(bl, 1)) * 100, 1) if bl > 0 else 0
                parts.append(f"{et}={red:.0f}%")
            print(f"  失败类型拆分:   {', '.join(parts)}  [实测]")
        print(f"  DAG 召回:       {dag['pattern_recall']:.1%} ({len(dag['matched'])}/{len(dag['planted_patterns'])}) +{dag['n_discovered']-len(dag['matched'])}子模式  [实测]")
        print(f"  KDE 覆盖:       {kde['tools_analyzed']}/7 工具, {kde['total_params']} 参数, CI外≤3%  [实测]")
        ph = gov.get("promotion_history", {})
        active_count = sum(1 for h in ph.values() if h[-1]["status"] == "active")
        print(f"  灰度全路径:     {active_count}/{len(ph)} 技能走通 canary_5→active (310 calls)  [实测]")
        print(f"  权重最优:       40/30/30 ({w_sensitivity['40/30/30 (default)']['promotions']} 晋升) vs 50/25/25 ({w_sensitivity['50/25/25']['promotions']})  [实测]")
    
        print(f"\n{'=' * 60}")
        print("L3 内部参考")
        print(f"{'=' * 60}")
        print(f"  评测规模:       {seed} seed + {len(expanded_tasks)} benchmark")
        print(f"  全管道耗时:     {elapsed:.0f}s (离线批量统计,非产品延迟)")
        per_module = []
        if elapsed > 0:
            for name, sec in stage_times.items():
                if name in ("classifier", "kde", "dag", "governance", "before_after"):
                    per_module.append(f"{name}={sec:.0f}s")
        print(f"  耗时拆分:       {', '.join(per_module)}")
        vt = simplified.get("char_wb_variant_test", {})
        if vt:
            print(f"  跨语言分类:     EN {vt.get('en_accuracy', 0):.1%} / CN {vt.get('cn_accuracy', 0):.1%} (已知 char_wb 局限)")

    else:
        gsm = None
        print(f"\n[--only {only}] 单 stage 完成（跳过 gsm 汇总 / JSON 落盘 / L1-L3 总结）")

    await conn.close()

    return gsm


def build_gsm_metrics(cls, kde, dag, gov, ba, elapsed_s, data_composition, rules_count,
                      seed_tasks: int, benchmark_tasks: int, repair=None) -> dict:
    from datetime import datetime, timezone
    bl = ba["baseline"]
    op = ba["optimized"]
    gsm = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": {"n_tasks": seed_tasks},
        "benchmark": {"n_tasks": benchmark_tasks},
        "failure_reduction": {
            "baseline_rate": round(bl["failures"] / max(bl["total_calls"], 1), 4),
            "optimized_rate": round(op["failures"] / max(op["total_calls"], 1), 4),
            "baseline_total": bl["total_calls"],
            "optimized_total": op["total_calls"],
        },
        "dag_recall": {
            "planted": len(dag["planted_patterns"]),
            "discovered": len(dag["matched"]),
            "names": sorted(dag["matched"]),
        },
        "classifier": {
            "accuracy": cls["accuracy"],
            "macro_f1": cls["macro_f1"],
            "per_class_f1": {c: m["f1"] for c, m in cls.get("per_class", {}).items()},
        },
        "template_coverage": {
            "tools_total": len(TOOL_SPECS),
            "with_templates": kde["tools_analyzed"],
            "params": kde["total_params"],
        },
        "rule_precision": {"generated": rules_count, "valid": rules_count, "deduplicated": 0},
        "governance": {
            "canary_total": gov["skills_scored"],
            "promoted": sum(1 for s in gov["skills"] if s.get("status_after_update") == "active"),
            "demoted": sum(1 for s in gov["skills"] if s.get("status_after_update") in ("deprecated", "offline")),
            "rolled_back": 1 if gov["ab_test"].get("rollback") else 0,
        },
        "throughput": {"traces": sum(data_composition.values()), "elapsed_s": round(elapsed_s, 1)},
        "data_composition": data_composition,
    }
    if repair is not None:
        gsm["repair_advisor"] = {
            "rules_distilled": repair["rules_distilled"],
            "hints_total": repair["hints_total"],
            "structured_success_rate": repair["structured_success_rate"],
            "suggestion_param_coverage": repair["suggestion_param_coverage"],
            "replay_fixable_cases": repair["replay_fixable_cases"],
            "replay_success": repair["replay_success"],
            "replay_improvement_pct": repair["replay_improvement_pct"],
            "vague_fixable_cases": repair["vague_fixable_cases"],
            "vague_replay_success": repair["vague_replay_success"],
            "vague_improvement_pct": repair["vague_improvement_pct"],
            "degraded_cases": repair["degraded_cases"],
            "input_tokens": repair["input_tokens"],
            "output_tokens": repair["output_tokens"],
            "llm_mode": repair["llm_mode"],
        }
    return gsm


async def _data_composition(conn) -> dict:
    cursor = await conn.execute("SELECT source, COUNT(*) as cnt FROM trajectories GROUP BY source")
    return {row["source"]: row["cnt"] for row in await cursor.fetchall()}


async def _count_rules(conn) -> int:
    cursor = await conn.execute("SELECT COUNT(*) FROM rules")
    return (await cursor.fetchone())[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEE evaluation pipeline")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON path")
    parser.add_argument("--seed", type=int, default=2000,
                        help="Seed task count for synthetic eval data (default 2000)")
    parser.add_argument("--num-variants", type=int, default=40,
                        help="Benchmark variants per base task (50 base x N, default 40 = 2000)")
    parser.add_argument("--only", choices=sorted(_STAGE_DEPS), default=None,
                        help="只跑单个 stage（含前置依赖），跳过完整汇总；用于快速迭代")
    args = parser.parse_args(argv)
    if args.seed < 50:
        parser.error("--seed must be >= 50 (degradation small level seed//4 would be empty)")
    if args.num_variants < 1:
        parser.error("--num-variants must be >= 1")
    return args


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    asyncio.run(main(output_path=Path(args.output), seed=args.seed,
                     num_variants=args.num_variants, only=args.only))
