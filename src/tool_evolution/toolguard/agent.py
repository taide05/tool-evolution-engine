"""Minimal OpenAI-compatible tool-calling agent used by the live demos (baseline vs guarded).

The same loop drives every model family in the multi-model experiment; only the endpoint
(``base_url``), model id and auth key differ.  ``DeepSeekClient`` is kept as a
backward-compatible alias of :class:`ChatCompletionsClient`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

SYSTEM_PROMPT = (
    "You are a factual assistant with access to tools. Use the tools to obtain the requested "
    "information; do not guess. When you have the answer, reply with one short sentence that "
    "contains the requested value."
)


@dataclass
class LLMConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    max_turns: int = 8
    temperature: float | None = 0.0
    max_tokens: int = 800
    # Some OpenAI reasoning models only accept ``max_completion_tokens``.
    max_tokens_param: str = "max_tokens"
    timeout_s: float = 60.0
    name: str = "deepseek"  # label used in error messages
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


class ChatCompletionsClient:
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint with tool calling."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = httpx.Client(timeout=config.timeout_s)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        body[self.config.max_tokens_param] = self.config.max_tokens
        body.update(self.config.extra_body)
        headers = {"Authorization": f"Bearer {self.config.api_key}", **self.config.extra_headers}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    headers=headers,
                    json=body,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"{self.config.name} HTTP {response.status_code}: {response.text[:200]}")
            payload = response.json()
            usage = payload.get("usage") or {}
            for key in self.usage:
                self.usage[key] += int(usage.get(key) or 0)
            return payload
        raise RuntimeError(f"{self.config.name} request failed: {last_error}")

    def close(self) -> None:
        self._client.close()


# Backward-compatible alias: the DeepSeek demo scripts import this name.
DeepSeekClient = ChatCompletionsClient


@dataclass
class AgentRun:
    answer: str
    turns: int
    tool_calls: int
    tool_errors: int
    usage: dict[str, int]
    elapsed_ms: float
    exhausted: bool = False
    guard_events: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


def run_task(
    client: DeepSeekClient,
    backend: Any,
    question: str,
    middleware: Any = None,
    max_turns: int | None = None,
) -> AgentRun:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools = backend.schemas()
    tool_calls = tool_errors = turns = 0
    budget = max_turns or client.config.max_turns
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    start = time.perf_counter()
    for turn in range(budget):
        turns += 1
        payload = client.chat(messages, tools)
        payload_usage = payload.get("usage") or {}
        for key in usage:
            usage[key] += int(payload_usage.get(key) or 0)
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        calls = message.get("tool_calls") or []
        if not calls:
            answer = message.get("content") or ""
            return AgentRun(
                answer=answer,
                turns=turns,
                tool_calls=tool_calls,
                tool_errors=tool_errors,
                usage=dict(usage),
                elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
                guard_events=middleware.dump_events() if middleware is not None else [],
                transcript=messages,
            )
        messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": calls})
        for call in calls:
            function = (call.get("function") or {})
            name = function.get("name") or ""
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls += 1
            result = (middleware.call(name, arguments) if middleware is not None
                      else backend.call(name, arguments))
            if not result.ok:
                tool_errors += 1
                content = json.dumps({"error": result.error or "tool error"}, ensure_ascii=False)
            else:
                content = json.dumps({"result": result.response}, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
    return AgentRun(
        answer="",
        turns=turns,
        tool_calls=tool_calls,
        tool_errors=tool_errors,
        usage=dict(usage),
        elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
        exhausted=True,
        guard_events=middleware.dump_events() if middleware is not None else [],
        transcript=messages,
    )
