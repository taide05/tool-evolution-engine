"""Tool-name normalization compatible with the ToolBench schema convention.

The recorded backend and the released trajectories use ToolBench's naming rules
for tool/api identifiers.  We re-implement the three small helpers here (they are
pure string normalization) so ToolGuard does not depend on the research code.
Semantics follow the public ToolBench/StableToolBench implementation.
"""

from __future__ import annotations

import re

_NON_TOKEN = re.compile(r"[^\u4e00-\u9fa5^a-z^A-Z^0-9^_]")
_REPEATED_UNDERSCORE = re.compile(r"(_)\1+")
_RESERVED = {"from", "class", "return", "false", "true", "id", "and"}


def standardize_category(category: str) -> str:
    save = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save or "," in save:
        save = save.replace(" ", "_").replace(",", "_")
    return save.replace("__", "_")


def standardize(value: str) -> str:
    """Lower-case token form used for tool names: ``Foo Bar!`` -> ``foo_bar``."""
    text = _NON_TOKEN.sub("_", value)
    text = _REPEATED_UNDERSCORE.sub("_", text).lower()
    text = text.strip("_")
    if text and text[0].isdigit():
        text = "get_" + text
    return text


def change_name(name: str) -> str:
    """Escape identifiers that collide with Python keywords in ToolBench."""
    return "is_" + name if name in _RESERVED else name


def function_name(api_name: str, tool_name: str) -> str:
    """OpenAI function name emitted by the ToolBench environment for one API."""
    pure_api = change_name(standardize(api_name))
    full = f"{pure_api}_for_{standardize(tool_name)}"
    # ToolBench truncates generated function names to keep them <= 64 chars.
    return full[-64:]


def cache_tool_dir(tool_name: str, category: str) -> str:
    """Directory name used by the recorded backend for one tool."""
    return f"{standardize(tool_name)}_for_{standardize_category(category)}"
