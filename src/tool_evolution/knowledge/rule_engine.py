import json
import aiosqlite


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
        triggered = []
        for row in rows:
            rule = dict(row)
            condition = json.loads(rule["condition"])
            param_names = condition.get("param_names", [])
            if param_names and any(p in params for p in param_names):
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
