"""Unit tests for ToolGuard naming helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.naming import (  # noqa: E402
    cache_tool_dir,
    change_name,
    function_name,
    standardize,
    standardize_category,
)


def test_standardize_matches_toolbench_rules():
    assert standardize("Find_User_ID by Email!") == "find_user_id_by_email"
    assert standardize(" 123tool ") == "get_123tool"
    assert standardize("foo__bar") == "foo_bar"


def test_standardize_category_replaces_separators():
    assert standardize_category("Business, Software") == "Business_Software"


def test_change_name_escapes_python_keywords():
    assert change_name("class") == "is_class"
    assert change_name("normal") == "normal"


def test_function_name_is_truncated_to_64_chars():
    long_api = "a" * 120
    name = function_name(long_api, "tool")
    assert len(name) == 64
    assert name.endswith("_for_tool")


def test_cache_tool_dir_joins_standard_tool_and_category():
    assert cache_tool_dir("My Tool", "Business") == "my_tool_for_Business"
