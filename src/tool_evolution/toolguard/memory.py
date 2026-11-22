"""Experience memory: repair a failed tool call by replaying a previously successful one.

This is the core "learn from real traces" component.  For every successful recorded
call we remember ``(task instance, function) -> argument set``.  When a later call
for the same task/function fails with a parameter-level error, the middleware can
re-issue the previously successful invocation instead of guessing.

Only the *training* split is allowed into memory; evaluation always happens on
held-out model families or held-out task groups.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .records import FunctionSchema, ToolCall


@dataclass
class ReplayCandidate:
    arguments: dict[str, Any]
    source_trace: str
    schema_score: int  # number of schema checks passed


@dataclass
class SuccessMemory:
    by_task_function: dict[tuple[str, str, str], list[dict[str, Any]]] = field(default_factory=dict)
    by_function_param: dict[tuple[str, str], Counter] = field(default_factory=dict)
    successful_calls: int = 0

    @classmethod
    def build(cls, calls: list[ToolCall], schemas: dict[str, FunctionSchema],
              only_valid_traces: bool = True) -> "SuccessMemory":
        memory = cls()
        for call in calls:
            if not call.recorded_ok:
                continue
            if only_valid_traces and not call.valid_data:
                continue
            memory.successful_calls += 1
            key = (call.group, call.query_id, call.tool_name)
            memory.by_task_function.setdefault(key, []).append(dict(call.arguments))
            for name, value in call.arguments.items():
                if isinstance(value, (str, int, float, bool)):
                    memory.by_function_param.setdefault((call.tool_name, name), Counter())[value] += 1
        return memory

    def size(self) -> dict[str, int]:
        return {
            "successful_calls": self.successful_calls,
            "task_function_sets": len(self.by_task_function),
            "function_param_modes": len(self.by_function_param),
        }

    def best_replay(self, call: ToolCall, schema: FunctionSchema | None) -> ReplayCandidate | None:
        """Return the most schema-consistent successful argument set for this task/function."""
        candidates = self.by_task_function.get((call.group, call.query_id, call.tool_name))
        if not candidates:
            return None
        best: ReplayCandidate | None = None
        seen: set[tuple[tuple[str, str], ...]] = set()
        for arguments in candidates:
            fingerprint = tuple(sorted((str(k), str(v)) for k, v in arguments.items()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            score = _schema_score(arguments, schema)
            if best is None or score > best.schema_score:
                best = ReplayCandidate(arguments=dict(arguments), source_trace=call.trace_id, schema_score=score)
        return best

    def mode_value(self, function_name: str, param_name: str) -> Any | None:
        counter = self.by_function_param.get((function_name, param_name))
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def param_values(self, function_name: str, param_name: str) -> Counter:
        return self.by_function_param.get((function_name, param_name)) or Counter()

    def observed_params(self, function_name: str) -> set[str]:
        return {param for (function, param) in self.by_function_param if function == function_name}


def _schema_score(arguments: dict[str, Any], schema: FunctionSchema | None) -> int:
    if schema is None:
        return 0
    score = 0
    if not any(name not in schema.properties for name in arguments):
        score += 1
    if all(name in arguments and arguments[name] not in ("", None) for name in schema.required):
        score += 2
    score += sum(1 for name in arguments if name in schema.properties)
    return score


def schema_valid(arguments: dict[str, Any], schema: FunctionSchema | None) -> bool:
    if schema is None:
        return False
    if any(name not in schema.properties for name in arguments):
        return False
    return all(name in arguments and arguments[name] not in ("", None) for name in schema.required)
