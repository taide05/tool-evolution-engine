import asyncio
import hashlib
import json
import httpx
import aiosqlite
from ..utils.config import settings

_ERROR_TYPE_BY_RULE = {
    "range_rule": "param_error",
    "auth_rule": "permission_denied",
    "retry_rule": "quota_exhausted",
    "timeout_rule": "timeout",
    "circuit_breaker_rule": "service_unavailable",
}

_SYSTEM_PROMPT = (
    "You are a repair advisor for an agent tool-calling system. "
    "Given a failed tool call's error information, produce a structured repair suggestion. "
    'Respond ONLY with a JSON object: {"suggestion": string, '
    '"fix": {"param": string, "suggested_value": any} | null, "reason": string}. '
    "fix.param must name a parameter that appears in the call parameters. "
    "suggested_value must keep the original parameter type: output a JSON number for "
    "numeric parameters, not a string. "
    "If the error cannot be fixed by changing a parameter value, set fix to null."
)


def _coerce_params(params) -> dict | None:
    """params 可能是 dict（内存轨迹）或 JSON 字符串（trajectories 表行）——统一解析为 dict。"""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            return None
    return params if isinstance(params, dict) else None


class RepairAdvisor:
    """Offline batch repair-suggestion generator backed by a DeepSeek-compatible LLM.

    Idempotency: content_hash = md5(canonical(condition) + canonical(action)).
    Four-branch generate flow: same rule_id + same hash -> return existing row;
    same rule_id + different hash -> regenerate and UPDATE (content evolution);
    no row + hash seen elsewhere -> copy-on-hit (fix must be non-NULL);
    otherwise -> call LLM and INSERT. Fail-open: no API key / transport failure /
    parse failure / fix-null all degrade to a template suggestion with fix=NULL.
    """

    def __init__(self, conn: aiosqlite.Connection,
                 client: httpx.AsyncClient | None = None,
                 concurrency: int | None = None,
                 timeout_s: float | None = None,
                 retries: int | None = None):
        self.conn = conn
        self._client = client
        self._owns_client = client is None
        self._sem = asyncio.Semaphore(
            concurrency if concurrency is not None else settings.repair_concurrency)
        self._timeout = timeout_s if timeout_s is not None else settings.repair_timeout_s
        self._retries = retries if retries is not None else settings.repair_retries
        self._write_lock = asyncio.Lock()

    def _content_hash(self, rule: dict) -> str:
        condition = rule["condition"]
        action = rule["action"]
        if not isinstance(condition, str):
            condition = json.dumps(condition, sort_keys=True, ensure_ascii=False)
        else:
            condition = json.dumps(json.loads(condition), sort_keys=True, ensure_ascii=False)
        if not isinstance(action, str):
            action = json.dumps(action, sort_keys=True, ensure_ascii=False)
        else:
            action = json.dumps(json.loads(action), sort_keys=True, ensure_ascii=False)
        canonical = f"{condition}|{action}"
        return hashlib.md5(canonical.encode()).hexdigest()

    def _param_names(self, rule: dict, examples: list[dict] | None) -> list[str]:
        condition = rule["condition"]
        if not isinstance(condition, dict):
            condition = json.loads(condition)
        names = condition.get("param_names") or []
        if names:
            return names
        if examples:
            params = _coerce_params(examples[0].get("params"))
            if params:
                return list(params.keys())
        return []

    def _template_hint(self, rule: dict, examples: list[dict] | None, reason: str) -> dict:
        names = self._param_names(rule, examples)
        if names:
            suggestion = (f"调用 {rule['tool_name']} 时参数不合法，"
                          f"请检查 {', '.join(names)} 的取值范围")
        else:
            suggestion = f"调用 {rule['tool_name']} 时参数不合法，请检查调用参数"
        return {
            "suggestion": suggestion,
            "fix": None,
            "reason": reason,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    async def _call_llm(self, rule: dict, examples: list[dict] | None) -> dict | None:
        """Returns {suggestion, fix, reason, input_tokens, output_tokens} or None (total failure)."""
        condition = rule["condition"]
        if not isinstance(condition, dict):
            condition = json.loads(condition)
        action = rule["action"]
        if not isinstance(action, dict):
            action = json.loads(action)
        error_template = action.get("error_msg_template")
        if not error_template and examples:
            error_template = examples[0].get("error_message", "")
        param_snapshots = []
        if examples:
            for ex in examples[:3]:
                params = _coerce_params(ex.get("params"))
                if params:
                    param_snapshots.append(params)
        error_type = _ERROR_TYPE_BY_RULE.get(rule["rule_type"], rule["rule_type"])
        user_payload = {
            "error_type": error_type,
            "tool_name": rule["tool_name"],
            "error_message": error_template or "",
            "param_names": self._param_names(rule, examples),
            "example_params": param_snapshots,
        }
        body = {
            "model": settings.repair_llm_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        }
        url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.deepseek_api_key}"}

        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            self._client = client
            self._owns_client = True

        last_exc: Exception | None = None
        for attempt in range(1 + self._retries):
            if attempt > 0:
                await asyncio.sleep(2 ** (attempt - 1))
            try:
                resp = await client.post(url, json=body, headers=headers,
                                         timeout=self._timeout)
            except httpx.HTTPError as exc:
                last_exc = exc
                continue
            if resp.status_code != 200:
                last_exc = RuntimeError(f"llm http {resp.status_code}")
                continue
            data = resp.json()
            usage = data.get("usage", {}) or {}
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = self._parse_content(content)
            if parsed is None:
                return None
            parsed["input_tokens"] = usage.get("prompt_tokens", 0)
            parsed["output_tokens"] = usage.get("completion_tokens", 0)
            return parsed
        if last_exc is not None:
            return None
        return None

    def _parse_content(self, content: str) -> dict | None:
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        suggestion = data.get("suggestion")
        if not isinstance(suggestion, str) or not suggestion:
            return None
        fix = data.get("fix")
        if fix is not None:
            if not isinstance(fix, dict) or not isinstance(fix.get("param"), str) \
                    or "suggested_value" not in fix:
                fix = None
        reason = data.get("reason")
        return {
            "suggestion": suggestion,
            "fix": fix,
            "reason": reason if isinstance(reason, str) else "",
        }

    async def _existing_by_hash(self, content_hash: str) -> dict | None:
        cursor = await self.conn.execute(
            """SELECT h.* FROM repair_hints h
               JOIN rules r ON h.rule_id = r.id
               WHERE h.content_hash=? AND h.fix IS NOT NULL
               LIMIT 1""",
            (content_hash,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def _get_by_rule_id(self, rule_id: int) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM repair_hints WHERE rule_id=?", (rule_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    def _row_to_hint(self, row: dict) -> dict:
        return {
            "rule_id": row["rule_id"],
            "content_hash": row["content_hash"],
            "suggestion": row["suggestion"],
            "fix": row["fix"],
            "model": row["model"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def _persist(self, rule_id: int, content_hash: str, payload: dict,
                       exists: bool) -> dict:
        async with self._write_lock:
            if exists:
                await self.conn.execute(
                    """UPDATE repair_hints SET content_hash=?, suggestion=?, fix=?,
                       model=?, input_tokens=?, output_tokens=?,
                       updated_at=datetime('now') WHERE rule_id=?""",
                    (content_hash, payload["suggestion"],
                     json.dumps(payload["fix"]) if payload["fix"] is not None else None,
                     settings.repair_llm_model,
                     payload.get("input_tokens", 0), payload.get("output_tokens", 0),
                     rule_id))
            else:
                await self.conn.execute(
                    """INSERT INTO repair_hints
                       (rule_id, content_hash, suggestion, fix, model, input_tokens, output_tokens)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (rule_id, content_hash, payload["suggestion"],
                     json.dumps(payload["fix"]) if payload["fix"] is not None else None,
                     settings.repair_llm_model,
                     payload.get("input_tokens", 0), payload.get("output_tokens", 0)))
            await self.conn.commit()
        row = await self._get_by_rule_id(rule_id)
        return self._row_to_hint(row)

    async def generate_for_rule(self, rule: dict,
                                examples: list[dict] | None = None) -> dict:
        hint, _ = await self._generate_kind(rule, examples)
        return hint

    async def _generate_kind(self, rule: dict,
                             examples: list[dict] | None) -> tuple[dict, str]:
        """Returns (hint, kind): kind = 'generated'（本调用新产出）| 'reused'（复用既有内容，0 API）。"""
        rule_id = rule["id"]
        content_hash = self._content_hash(rule)

        existing = await self._get_by_rule_id(rule_id)
        if existing is not None:
            if existing["content_hash"] == content_hash:
                return self._row_to_hint(existing), "reused"
            payload = await self._generate_payload(rule, examples)
            return await self._persist(rule_id, content_hash, payload, exists=True), "generated"

        reused = await self._existing_by_hash(content_hash)
        if reused is not None:
            payload = {
                "suggestion": reused["suggestion"],
                "fix": json.loads(reused["fix"]) if reused["fix"] else None,
                "input_tokens": 0,
                "output_tokens": 0,
            }
            return await self._persist(rule_id, content_hash, payload, exists=False), "reused"

        payload = await self._generate_payload(rule, examples)
        return await self._persist(rule_id, content_hash, payload, exists=False), "generated"

    async def _generate_payload(self, rule: dict,
                                examples: list[dict] | None) -> dict:
        if not settings.deepseek_api_key:
            return self._template_hint(rule, examples, "no_api_key")
        llm = await self._call_llm(rule, examples)
        if llm is None:
            return self._template_hint(rule, examples, "llm_unavailable")
        if llm["fix"] is None:
            template = self._template_hint(rule, examples, "llm_fix_null")
            return {
                "suggestion": template["suggestion"],
                "fix": None,
                "reason": "llm_fix_null",
                "input_tokens": llm["input_tokens"],
                "output_tokens": llm["output_tokens"],
            }
        return llm

    async def generate_for_rules(self, rules: list[dict],
                                 examples_by_hash: dict | None = None) -> dict:
        examples_by_hash = examples_by_hash or {}
        counter = {"generated": 0, "reused": 0}

        async def _one(rule: dict) -> dict:
            async with self._sem:
                hint, kind = await self._generate_kind(
                    rule, examples=examples_by_hash.get(rule.get("_hash")))
                counter[kind] += 1
                return hint

        hints = list(await asyncio.gather(*(_one(r) for r in rules)))
        degraded = sum(1 for h in hints if h["fix"] is None)
        return {
            "generated": counter["generated"],
            "reused": counter["reused"],
            "degraded": degraded,
            "hints": hints,
            "total_input_tokens": sum(h["input_tokens"] for h in hints),
            "total_output_tokens": sum(h["output_tokens"] for h in hints),
        }

    async def get_hint(self, rule_id: int) -> dict | None:
        row = await self._get_by_rule_id(rule_id)
        return self._row_to_hint(row) if row else None

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
