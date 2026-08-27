"""计划装配器——技能包 DAG + KDE 默认值 + 偏好注入 + 规则前置校验 → 可执行计划。"""

import json

import aiosqlite

from ..governance.mcp_bridge import MCPBridge
from ..knowledge.param_template import ParamTemplateManager, flatten_user_prefs
from ..knowledge.rule_engine import RuleEngine


class PlanAssembler:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def assemble(self, skill: dict, task_params: dict | None = None,
                       optimized: bool = True) -> dict | None:
        """返回可执行计划：

        {'skill_id','skill_name','nodes':[{tool_name,params}],'edges':[{'from':idx,'to':idx}],
         'precheck_rules':[dict], 'blocked': bool, 'block_reason': str|None}

        装配顺序（D3 覆盖口径）：dag_definition 解析 nodes/edges（R5 端点→索引映射）
        → 每节点参数 = KDE 默认值（get_template 的 default_value，跳过 None）
        → 偏好覆盖（flatten_user_prefs）→ task_params 覆盖（最高优先级）
        → RuleEngine.check 前置拦截（触发非空→blocked，不抛）。

        optimized=False（灰度 stable 基线）：跳过 KDE 默认值/偏好注入/前置拦截，
        仅 task_params 直装配——与 run_eval before/after 的 baseline 口径对齐。
        """
        try:
            dag = json.loads(skill["dag_definition"])
        except (json.JSONDecodeError, TypeError):
            return {
                "skill_id": skill["id"], "skill_name": skill["name"],
                "nodes": [], "edges": [],
                "precheck_rules": [], "blocked": True,
                "block_reason": "dag_definition 不是合法 JSON",
            }
        raw_nodes = dag.get("nodes", [])
        raw_edges = dag.get("edges", [])

        node_ref = _build_node_ref(raw_nodes)
        edges = []
        for edge in raw_edges:
            src = node_ref.get(edge.get("from"))
            dst = node_ref.get(edge.get("to"))
            if src is None or dst is None:
                missing = edge.get("from") if src is None else edge.get("to")
                return {
                    "skill_id": skill["id"], "skill_name": skill["name"],
                    "nodes": [], "edges": [],
                    "precheck_rules": [], "blocked": True,
                    "block_reason": f"edges 端点 '{missing}' 无法映射到节点",
                }
            edges.append({"from": src, "to": dst})

        tmpl_mgr = ParamTemplateManager(self.conn)
        bridge = MCPBridge(self.conn)
        prefs = await bridge.get_user_preferences() if optimized else {}
        engine = RuleEngine(self.conn)

        nodes = []
        precheck_rules = []
        for node in raw_nodes:
            tool_name = node.get("tool_name")
            params: dict = {}
            if optimized:
                template = await tmpl_mgr.get_template(tool_name, "1.0.0")
                if template:
                    for pname, pdef in template.items():
                        if pdef.get("default_value") is not None:
                            params[pname] = pdef["default_value"]
                params.update(flatten_user_prefs(prefs, tool_name))
            if task_params:
                params.update(task_params)
            nodes.append({"tool_name": tool_name, "params": params})
            if not optimized:
                continue
            triggered = await engine.check(tool_name, "1.0.0", params)
            for rule in triggered:
                precheck_rules.append({
                    "rule_id": rule["id"], "rule_type": rule["rule_type"],
                    "tool_name": tool_name,
                })

        blocked = bool(precheck_rules)
        return {
            "skill_id": skill["id"], "skill_name": skill["name"],
            "nodes": nodes, "edges": edges,
            "precheck_rules": precheck_rules,
            "blocked": blocked,
            "block_reason": (
                f"前置拦截触发 {len(precheck_rules)} 条规则" if blocked else None
            ),
        }


def _build_node_ref(raw_nodes: list[dict]) -> dict[str, int]:
    """R5 解析规则：'tool_name'→首个匹配节点、'tool_name_N'→第 N+1 次出现的节点。"""
    ref: dict[str, int] = {}
    counts: dict[str, int] = {}
    for idx, node in enumerate(raw_nodes):
        tool = node.get("tool_name")
        if tool is None:
            continue
        count = counts.get(tool, 0)
        counts[tool] = count + 1
        endpoint = tool if count == 0 else f"{tool}_{count}"
        ref[endpoint] = idx
    return ref
