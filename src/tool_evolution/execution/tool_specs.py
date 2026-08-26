"""确定性工具规范——MockAdapter 的模拟语义与 llm_plan 提示词的工具清单。

参数 schema 与 run_eval seed 数据同构（query/max_results/lang/timeout_ms 为
全工具通用参数集），保证执行层评测与管道评测语义一致。
"""

_COMMON_PARAMS = {
    "query": {"type": "string", "required": True},
    "max_results": {"type": "int", "required": False, "default": 10},
    "lang": {"type": "string", "required": False, "default": "zh"},
    "timeout_ms": {"type": "int", "required": False, "default": 10000},
}

TOOL_SPECS: dict[str, dict] = {
    "search_api": {
        "params": _COMMON_PARAMS,
        "mock_result_keys": ["documents", "total"],
        "mock_token_cost": 120,
        "mock_latency_ms": 175,
    },
    "detail_api": {
        "params": _COMMON_PARAMS,
        "mock_result_keys": ["item", "source"],
        "mock_token_cost": 90,
        "mock_latency_ms": 140,
    },
    "analyze_api": {
        "params": _COMMON_PARAMS,
        "mock_result_keys": ["summary", "keywords"],
        "mock_token_cost": 160,
        "mock_latency_ms": 220,
    },
    "report_api": {
        "params": _COMMON_PARAMS,
        "mock_result_keys": ["report", "sections"],
        "mock_token_cost": 200,
        "mock_latency_ms": 190,
    },
    "github_api": {
        "params": _COMMON_PARAMS,
        "mock_result_keys": ["repos", "count"],
        "mock_token_cost": 110,
        "mock_latency_ms": 250,
    },
    "arxiv_api": {
        "params": _COMMON_PARAMS,
        "mock_result_keys": ["papers", "total"],
        "mock_token_cost": 130,
        "mock_latency_ms": 230,
    },
    "official_docs": {
        "params": _COMMON_PARAMS,
        "mock_result_keys": ["sections", "version"],
        "mock_token_cost": 100,
        "mock_latency_ms": 150,
    },
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
    return None
