import json
import aiosqlite


def _range_violated(param_name: str, params: dict, template: dict | None) -> bool:
    """range_rule 值校验：参数存在且（无界可查 → 保守拦截；有界 → 值越界/非数值拦截）。"""
    if param_name not in params:
        return False
    bounds = (template or {}).get(param_name, {})
    lo = bounds.get("lower_bound")
    hi = bounds.get("upper_bound")
    if lo is None or hi is None:
        return True
    try:
        value = float(params[param_name])
    except (TypeError, ValueError):
        return True
    return not (lo <= value <= hi)


class RuleEngine:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def add_rule(self, rule: dict) -> int:
        cursor = await self.conn.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (rule["tool_name"], rule["tool_version"], rule["rule_type"],
             json.dumps(rule["condition"]), json.dumps(rule["action"]),
             rule.get("status", "active"))
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def check(self, tool_name: str, tool_version: str, params: dict) -> list[dict]:
        cursor = await self.conn.execute(
            """SELECT * FROM rules
               WHERE tool_name=? AND tool_version=? AND status='active'""",
            (tool_name, tool_version)
        )
        rows = await cursor.fetchall()
        from .param_template import ParamTemplateManager
        template = await ParamTemplateManager(self.conn).get_template(tool_name, tool_version)
        triggered = []
        for row in rows:
            rule = dict(row)
            try:
                condition = json.loads(rule["condition"])
            except (json.JSONDecodeError, TypeError):
                continue
            param_names = condition.get("param_names", [])
            if param_names and rule["rule_type"] == "range_rule":
                # 值校验（D2-6 修复）：以 KDE 界判定真越界，无界可查则保守触发——
                # 修复前"参数名命中即拦"导致合法值调用也被前置拦截
                if any(_range_violated(p, params, template)
                       for p in param_names):
                    triggered.append(rule)
            elif param_names and any(p in params for p in param_names):
                triggered.append(rule)
            elif not param_names:
                triggered.append(rule)
        return triggered

    async def get_runtime_rules(self, tool_name: str, tool_version: str) -> list[dict]:
        """执行层运行时控制规则——只取 3 类运行时规则，不复用 check（check 对无
        param_names 规则一律触发，会把运行时规则误当前置拦截）。"""
        cursor = await self.conn.execute(
            """SELECT * FROM rules
               WHERE tool_name=? AND tool_version=? AND status='active'
                 AND rule_type IN ('timeout_rule','retry_rule','circuit_breaker_rule')
               ORDER BY id ASC""",
            (tool_name, tool_version)
        )
        rules = []
        for row in await cursor.fetchall():
            rule = dict(row)
            rule["condition"] = json.loads(rule["condition"])
            rule["action"] = json.loads(rule["action"])
            rules.append(rule)
        return rules

    async def deprecate_version(self, tool_name: str, old_version: str) -> None:
        await self.conn.execute(
            "UPDATE rules SET status='deprecated' WHERE tool_name=? AND tool_version=?",
            (tool_name, old_version)
        )
        await self.conn.commit()
