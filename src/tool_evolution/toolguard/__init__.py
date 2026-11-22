"""ToolGuard: tool-call reliability middleware for LLM agents.

This package is the product-shaped part of the project: it sits between an agent
and a tool backend, records every call, classifies failures, mines repair rules
from historical failures, and applies repairs at runtime.

The evaluation harness replays *real* failed tool calls from public trajectories
(StableToolBench baseline release, Apache-2.0) against a deterministic recorded
backend, so results are reproducible without any external API key.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
