"""Check the configured LLM providers for the multi-model experiment.

Reads credentials through ``tool_evolution.utils.config.settings`` (i.e. from `.env` or
OS environment variables, prefix ``TOOLEVO_``) and never prints the keys themselves.

Without flags it only reports which providers are configured.  With ``--smoke`` it makes
one real tool-calling round trip per configured provider and reports:
tool-call detected?, latency, token usage, and the returned model id.  This is the
gate before spending money on the multi-model experiment.

Usage
-----
    python scripts/check_providers.py
    python scripts/check_providers.py --smoke
    python scripts/check_providers.py --smoke --provider qwen
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.agent import ChatCompletionsClient, LLMConfig  # noqa: E402
from tool_evolution.utils.config import settings  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo the given text back.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "text to echo"}},
            "required": ["text"],
        },
    },
}]
MESSAGES = [
    {"role": "system", "content": "You must use the provided tool when the user asks for it."},
    {"role": "user", "content": "Call the echo tool with the text 'ping'."},
]


def provider_table() -> dict[str, dict]:
    return {
        "deepseek": {
            "label": "mid / DeepSeek-V4.1-Flash",
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "model": "deepseek-v4-flash",
            "max_tokens_param": "max_tokens",
        },
        "qwen": {
            "label": "cheap / qwen3.8-flash",
            "api_key": settings.qwen_api_key,
            "base_url": settings.qwen_base_url,
            "model": settings.qwen_model,
            "max_tokens_param": "max_tokens",
        },
        "qwen-strong": {
            "label": "strong / qwen4-max",
            "api_key": settings.qwen_api_key,
            "base_url": settings.qwen_base_url,
            "model": settings.qwen_strong_model,
            "max_tokens_param": "max_tokens",
        },
        "openai": {
            "label": "optional / GPT-6.2 Sol",
            "api_key": settings.openai_api_key,
            "base_url": settings.openai_base_url,
            "model": settings.openai_model,
            "max_tokens_param": settings.openai_max_tokens_param,
        },
    }


def smoke(provider: str, config: dict) -> bool:
    if not config["api_key"]:
        print(f"[{provider:8s}] NOT CONFIGURED — no key in .env / environment")
        return False
    client = ChatCompletionsClient(LLMConfig(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        name=provider,
        max_tokens=200,
        max_tokens_param=config["max_tokens_param"],
        timeout_s=90.0,
    ))
    started = time.perf_counter()
    try:
        payload = client.chat(MESSAGES, TOOLS)
    except Exception as exc:  # noqa: BLE001 - report, never leak the key
        elapsed = time.perf_counter() - started
        print(f"[{provider:8s}] FAIL after {elapsed:.1f}s — {type(exc).__name__}: {str(exc)[:300]}")
        return False
    finally:
        client.close()
    elapsed = time.perf_counter() - started
    message = ((payload.get("choices") or [{}])[0].get("message") or {})
    calls = message.get("tool_calls") or []
    usage = payload.get("usage") or {}
    returned_model = payload.get("model", config["model"])
    verdict = "PASS" if calls else "FAIL (no tool call emitted)"
    names = [((call.get("function") or {}).get("name") or "?") for call in calls]
    print(f"[{provider:8s}] {verdict} — {elapsed:.1f}s, model={returned_model}, "
          f"tool_calls={names or '[]'}, usage={json_compact(usage)}")
    return bool(calls)


def json_compact(value: dict) -> str:
    return "{" + ", ".join(f"{key}={value[key]}" for key in sorted(value)) + "}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="make one real tool-calling request")
    parser.add_argument("--provider", choices=("deepseek", "qwen", "openai"), default=None)
    args = parser.parse_args()

    table = provider_table()
    providers = [args.provider] if args.provider else list(table)
    print("configured providers (keys never printed):")
    for provider in providers:
        config = table[provider]
        state = "configured" if config["api_key"] else "missing key"
        print(f"  {provider:8s} {config['label']:26s} model={config['model']:18s} {state} "
              f"base_url={config['base_url']}")

    if not args.smoke:
        print("\nrun with --smoke for one real tool-calling round trip per configured provider")
        return 0

    print("\n=== tool-calling smoke test ===")
    failures = 0
    for provider in providers:
        if not table[provider]["api_key"]:
            print(f"[{provider:8s}] SKIP — no key configured (not counted as a failure)")
            continue
        if not smoke(provider, table[provider]):
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
