"""确定性工具规范——MockAdapter 的模拟语义与 llm_plan 提示词的工具清单。

参数 schema 是「agent 执行任务（非复杂）中会调用的工具」的真实参数集（异构——
不同工具不同参数名/类型），不再是全工具通用的 query/max_results 同质集。
`mock` 字段给 string 参数的示例值；`min`/`max` 给 int/float 参数的采样范围，
供 `mock_params_for_tool()` 生成 seed/benchmark 的模拟参数。
"""

import random


def _p(t: str, required: bool = False, default=None, mock=None,
       lo: int | None = None, hi: int | None = None):
    """参数定义的紧凑构造器。"""
    d: dict = {"type": t}
    if required:
        d["required"] = True
    if default is not None:
        d["default"] = default
    if mock is not None:
        d["mock"] = mock
    if lo is not None:
        d["min"] = lo
    if hi is not None:
        d["max"] = hi
    return d


def _spec(params, result_keys, token_cost, latency_ms):
    return {
        "params": params,
        "mock_result_keys": result_keys,
        "mock_token_cost": token_cost,
        "mock_latency_ms": latency_ms,
    }


TOOL_SPECS: dict[str, dict] = {
    # ── 检索与知识 ──────────────────────────────────────────────
    "search_api": _spec(
        {"query": _p("string", required=True, mock="数据接口 使用说明"),
         "max_results": _p("int", default=10, lo=1, hi=50),
         "lang": _p("string", default="zh", mock="zh"),
         "timeout_ms": _p("int", default=10000, lo=1000, hi=30000)},
        ["documents", "total"], 120, 175),
    "detail_api": _spec(
        {"query": _p("string", required=True, mock="doc-1024 详情"),
         "max_results": _p("int", default=10, lo=1, hi=50),
         "lang": _p("string", default="zh", mock="zh"),
         "timeout_ms": _p("int", default=10000, lo=1000, hi=30000)},
        ["item", "source"], 90, 140),
    "web_search": _spec(
        {"query": _p("string", required=True, mock="大模型 agent 工具调用"),
         "num_pages": _p("int", default=3, lo=1, hi=10),
         "safe_search": _p("bool", default=True),
         "region": _p("string", mock="zh-CN")},
        ["pages", "total"], 110, 260),
    "vector_search": _spec(
        {"query": _p("string", required=True, mock="工具调用失败重试策略"),
         "collection": _p("string", required=True, mock="tool_docs"),
         "top_k": _p("int", default=5, lo=1, hi=20),
         "score_threshold": _p("float", lo=0.0, hi=1.0)},
        ["hits", "distances"], 100, 95),
    "knowledge_base": _spec(
        {"query": _p("string", required=True, mock="如何配置 canary 灰度"),
         "section": _p("string", mock="ops"),
         "format": _p("string", default="markdown", mock="markdown")},
        ["articles", "count"], 95, 130),
    "document_qa": _spec(
        {"question": _p("string", required=True, mock="这个参数默认值是多少"),
         "doc_id": _p("string", required=True, mock="kb-42"),
         "context_window": _p("int", default=2000, lo=1000, hi=8000)},
        ["answer", "citations"], 130, 300),
    "wikipedia": _spec(
        {"title": _p("string", required=True, mock="Tool calling"),
         "language": _p("string", default="zh", mock="zh"),
         "section": _p("string", mock="summary")},
        ["text", "url"], 80, 210),

    # ── 数据与存储 ──────────────────────────────────────────────
    "database_query": _spec(
        {"sql": _p("string", required=True, mock="SELECT id, name FROM users WHERE active=1"),
         "db_name": _p("string", default="main", mock="main"),
         "limit": _p("int", default=100, lo=1, hi=500)},
        ["rows", "count"], 150, 180),
    "table_lookup": _spec(
        {"table": _p("string", required=True, mock="orders"),
         "columns": _p("string", mock="id,status,amount"),
         "filters": _p("string", mock="status=paid"),
         "sort_by": _p("string", mock="created_at desc")},
        ["rows", "count"], 120, 120),
    "cache_get": _spec(
        {"key": _p("string", required=True, mock="user:42:profile"),
         "namespace": _p("string", default="default", mock="default")},
        ["value", "hit"], 60, 45),
    "cache_set": _spec(
        {"key": _p("string", required=True, mock="user:42:profile"),
         "value": _p("string", required=True, mock="{\"name\":\"alice\"}"),
         "ttl_seconds": _p("int", default=3600, lo=1, hi=86400)},
        ["ok", "expires_at"], 70, 50),
    "data_export": _spec(
        {"format": _p("string", required=True, mock="csv"),
         "columns": _p("string", mock="id,name,date"),
         "file_path": _p("string", mock="exports/result.csv")},
        ["file_path", "rows_written"], 140, 240),

    # ── 文件操作 ────────────────────────────────────────────────
    "file_read": _spec(
        {"path": _p("string", required=True, mock="/data/report.txt"),
         "encoding": _p("string", default="utf-8", mock="utf-8"),
         "offset": _p("int", default=0, lo=0, hi=1000)},
        ["content", "size"], 80, 70),
    "file_write": _spec(
        {"path": _p("string", required=True, mock="/data/out.txt"),
         "content": _p("string", required=True, mock="generated report"),
         "mode": _p("string", default="w", mock="w")},
        ["ok", "bytes_written"], 90, 90),
    "file_list": _spec(
        {"directory": _p("string", required=True, mock="/data"),
         "pattern": _p("string", mock="*.txt"),
         "recursive": _p("bool", default=False)},
        ["files", "count"], 70, 80),
    "file_delete": _spec(
        {"path": _p("string", required=True, mock="/data/tmp.txt"),
         "force": _p("bool", default=False)},
        ["ok"], 50, 60),

    # ── 代码与开发 ──────────────────────────────────────────────
    "code_execute": _spec(
        {"code": _p("string", required=True, mock="print(sum(range(10)))"),
         "language": _p("string", default="python", mock="python"),
         "timeout_ms": _p("int", default=10000, lo=1000, hi=60000)},
        ["stdout", "exit_code"], 180, 400),
    "code_review": _spec(
        {"file_path": _p("string", required=True, mock="src/app.py"),
         "severity_threshold": _p("string", default="warning", mock="warning")},
        ["issues", "summary"], 170, 500),
    "git_commit": _spec(
        {"message": _p("string", required=True, mock="fix: retry backoff"),
         "files": _p("string", mock="src/retry.py"),
         "branch": _p("string", mock="main")},
        ["commit_sha", "ok"], 110, 160),
    "git_diff": _spec(
        {"base_branch": _p("string", required=True, mock="main"),
         "head_branch": _p("string", mock="feature/x"),
         "path": _p("string", mock="src/")},
        ["diff", "changed_files"], 100, 150),
    "run_tests": _spec(
        {"test_path": _p("string", required=True, mock="tests/"),
         "coverage": _p("bool", default=False),
         "parallel": _p("int", default=1, lo=1, hi=8)},
        ["passed", "failed", "coverage_pct"], 200, 2000),

    # ── 网络与通信 ──────────────────────────────────────────────
    "http_get": _spec(
        {"url": _p("string", required=True, mock="https://api.example.com/v1/items"),
         "headers": _p("string", mock="Authorization: Bearer ..."),
         "timeout_ms": _p("int", default=10000, lo=1000, hi=30000)},
        ["status", "body"], 90, 210),
    "http_post": _spec(
        {"url": _p("string", required=True, mock="https://api.example.com/v1/items"),
         "body": _p("string", mock="{\"name\":\"x\"}"),
         "content_type": _p("string", default="application/json", mock="application/json")},
        ["status", "body"], 100, 230),
    "send_email": _spec(
        {"to": _p("string", required=True, mock="user@example.com"),
         "subject": _p("string", required=True, mock="任务完成通知"),
         "body": _p("string", mock="任务已完成，详见附件")},
        ["ok", "message_id"], 120, 350),
    "send_message": _spec(
        {"channel": _p("string", required=True, mock="slack"),
         "recipient": _p("string", required=True, mock="#ops"),
         "text": _p("string", required=True, mock="deploy finished")},
        ["ok", "ts"], 80, 180),

    # ── 分析与平台 ──────────────────────────────────────────────
    "analyze_api": _spec(
        {"query": _p("string", required=True, mock="待分析的技术文档摘要"),
         "max_results": _p("int", default=10, lo=1, hi=50),
         "lang": _p("string", default="zh", mock="zh"),
         "timeout_ms": _p("int", default=10000, lo=1000, hi=30000)},
        ["summary", "keywords"], 160, 220),
    "report_api": _spec(
        {"query": _p("string", required=True, mock="周度分析报告"),
         "max_results": _p("int", default=10, lo=1, hi=50),
         "lang": _p("string", default="zh", mock="zh"),
         "timeout_ms": _p("int", default=10000, lo=1000, hi=30000)},
        ["report", "sections"], 200, 190),
    "github_api": _spec(
        {"repo": _p("string", required=True, mock="owner/repo"),
         "per_page": _p("int", default=30, lo=1, hi=100),
         "state": _p("string", default="open", mock="open"),
         "sort": _p("string", default="created", mock="created")},
        ["repos", "count"], 110, 250),
    "arxiv_api": _spec(
        {"query": _p("string", required=True, mock="machine learning agent"),
         "max_results": _p("int", default=10, lo=1, hi=30),
         "category": _p("string", mock="cs.AI"),
         "sort_by": _p("string", default="relevance", mock="relevance")},
        ["papers", "total"], 130, 230),
    "official_docs": _spec(
        {"url": _p("string", required=True, mock="https://docs.example.com/v1/api"),
         "format": _p("string", default="json", mock="json"),
         "retry": _p("bool", default=False),
         "timeout_ms": _p("int", default=10000, lo=1000, hi=30000)},
        ["sections", "version"], 100, 150),
}


def describe_tools_for_llm() -> str:
    """生成 llm_plan 提示词的工具清单段——全部工具名+参数 schema。"""
    lines = []
    for name, spec in TOOL_SPECS.items():
        params = []
        for pname, pdef in spec["params"].items():
            req = "required" if pdef.get("required") else f"default={pdef.get('default')}"
            params.append(f"{pname}({pdef['type']}, {req})")
        lines.append(f"- {name}: " + ", ".join(params))
    return "Available tools:\n" + "\n".join(lines)


def mock_params_for_tool(tool_name: str, rng: random.Random) -> dict:
    """按参数 schema 生成 mock 参数：int/float 用 min/max 采样，bool 随机，string 用 mock 示例值。

    可选参数以 ~30% 概率省略（模拟 LLM 未显式传参）；required 恒生成。
    """
    spec = TOOL_SPECS.get(tool_name)
    if spec is None:
        return {}
    out: dict = {}
    for pname, pdef in spec["params"].items():
        if not pdef.get("required") and rng.random() > 0.7:
            continue
        out[pname] = _mock_value(pname, pdef, rng)
    return out


def _mock_value(pname: str, pdef: dict, rng: random.Random):
    t = pdef["type"]
    if t == "int":
        return rng.randint(pdef.get("min", 1), pdef.get("max", 100))
    if t == "float":
        return round(rng.uniform(pdef.get("min", 0.0), pdef.get("max", 1.0)), 3)
    if t == "bool":
        return rng.choice([True, False])
    return pdef.get("mock", pname)


def _coerce_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def validate_params(tool_name: str, params: dict) -> str | None:
    """返回参数非法原因或 None。unknown 工具 / 缺 required / 类型不符即非法。"""
    spec = TOOL_SPECS.get(tool_name)
    if spec is None:
        return f"unknown tool '{tool_name}'"
    for pname, pdef in spec["params"].items():
        if pname not in params:
            if pdef.get("required"):
                return f"missing required parameter '{pname}'"
            continue
        value = params[pname]
        if pdef["type"] == "int":
            if _coerce_int(value) is None:
                return f"parameter '{pname}' expects int, got {type(value).__name__}"
        elif pdef["type"] == "string":
            if not isinstance(value, str):
                return f"parameter '{pname}' expects string, got {type(value).__name__}"
        elif pdef["type"] == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"parameter '{pname}' expects float, got {type(value).__name__}"
        elif pdef["type"] == "bool":
            if not isinstance(value, bool):
                return f"parameter '{pname}' expects bool, got {type(value).__name__}"
    return None
