"""LLM 规划基线（对照组）——DeepSeek 生成顺序执行计划，fail-closed。"""

import asyncio
import json

import httpx

from ..utils.config import settings
from .tool_specs import TOOL_SPECS, describe_tools_for_llm

_SYSTEM_PROMPT = (
    "You are a task planner for a deterministic tool-execution system. "
    "Given a task description, produce a sequential execution plan using ONLY "
    "the tools listed below. "
    'Respond ONLY with a JSON object: {"steps": [{"tool": string, '
    '"params": object}, ...]}. '
    "Every step's params must include all required parameters of that tool."
    "\n\n"
    + describe_tools_for_llm()
)


class LLMPlanner:
    """对照组基线：LLM 规划 → 顺序 steps。校验失败/不可用 → None（R7 fail-closed）。"""

    def __init__(self, client: httpx.AsyncClient | None = None,
                 timeout_s: float | None = None, retries: int | None = None):
        self._client = client
        self._owns_client = client is None
        self._timeout = timeout_s if timeout_s is not None else settings.repair_timeout_s
        self._retries = retries if retries is not None else settings.repair_retries

    async def plan(self, task_description: str) -> list[dict] | None:
        if not settings.deepseek_api_key:
            return None
        body = {
            "model": settings.repair_llm_model,
            "temperature": 0.1,
            # v4 默认思考模式（实测输出 token ~8x）——显式关闭（同 repair_advisor）
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": task_description},
            ],
        }
        url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.deepseek_api_key}"}

        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=self._timeout)
            self._client = client
            self._owns_client = True

        for attempt in range(1 + self._retries):
            if attempt > 0:
                await asyncio.sleep(2 ** (attempt - 1))
            try:
                resp = await client.post(url, json=body, headers=headers,
                                         timeout=self._timeout)
            except httpx.HTTPError:
                continue
            if resp.status_code != 200:
                continue
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return self._validate(json.loads(content) if _safe_json(content) else None)
        return None

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _validate(payload) -> list[dict] | None:
        """fail-closed 结构校验：steps 非空、tool 合法、缺 required 键 → None；
        未知工具剔除（剔除后空→None）；params 未知键剔除。"""
        if not isinstance(payload, dict):
            return None
        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            return None
        cleaned = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = step.get("tool")
            spec = TOOL_SPECS.get(tool)
            if spec is None:
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                params = {}
            params = {k: v for k, v in params.items() if k in spec["params"]}
            for pname, pdef in spec["params"].items():
                if pdef.get("required") and pname not in params:
                    return None
            cleaned.append({"tool": tool, "params": params})
        return cleaned or None


def _safe_json(content) -> dict | None:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None
