import json
import pytest
from tool_evolution.analysis.dag_miner import DAGMiner


@pytest.fixture
def sample_task_traces():
    traces = []
    for task_id in range(20):
        root = {"trace_id": f"root-{task_id}", "parent_trace_id": None,
                "tool_name": "run_check", "trace_type": "task_root", "success": 1}
        if task_id >= 18:
            c1 = {"trace_id": f"c1-{task_id}", "parent_trace_id": f"root-{task_id}",
                  "tool_name": "search_law", "trace_type": "atomic", "success": 1}
            c2 = {"trace_id": f"c2-{task_id}", "parent_trace_id": f"root-{task_id}",
                  "tool_name": "export_pdf", "trace_type": "atomic", "success": 1}
            traces.extend([root, c1, c2])
        else:
            c1 = {"trace_id": f"c1-{task_id}", "parent_trace_id": f"root-{task_id}",
                  "tool_name": "search_law", "trace_type": "atomic", "success": 1,
                  "params": {"query": f"劳动法 第{task_id}条", "max_results": task_id + 5}}
            c2 = {"trace_id": f"c2-{task_id}", "parent_trace_id": f"root-{task_id}",
                  "tool_name": "analyze_compliance", "trace_type": "atomic", "success": 1,
                  "params": {"threshold": 0.8, "strict_mode": True}}
            c3 = {"trace_id": f"c3-{task_id}", "parent_trace_id": f"root-{task_id}",
                  "tool_name": "generate_report", "trace_type": "atomic", "success": 1,
                  "params": {"format": "markdown", "lang": "zh"}}
            traces.extend([root, c1, c2, c3])
    return traces


@pytest.fixture
def branching_traces():
    """5 tasks where some branch: search → (analyze | summarize) → report"""
    traces = []
    for task_id in range(5):
        root = {"trace_id": f"broot-{task_id}", "parent_trace_id": None,
                "tool_name": "orchestrator", "trace_type": "task_root", "success": 1}
        c1 = {"trace_id": f"bc1-{task_id}", "parent_trace_id": f"broot-{task_id}",
              "tool_name": "search_law", "trace_type": "atomic", "success": 1}
        traces.extend([root, c1])
        if task_id < 3:
            c2 = {"trace_id": f"bc2-{task_id}", "parent_trace_id": f"broot-{task_id}",
                  "tool_name": "analyze_compliance", "trace_type": "atomic", "success": 1}
        else:
            c2 = {"trace_id": f"bc2-{task_id}", "parent_trace_id": f"broot-{task_id}",
                  "tool_name": "summarize", "trace_type": "atomic", "success": 1}
        c3 = {"trace_id": f"bc3-{task_id}", "parent_trace_id": f"broot-{task_id}",
              "tool_name": "generate_report", "trace_type": "atomic", "success": 1}
        traces.extend([c2, c3])
    return traces


class TestDAGMiner:
    def test_mine_finds_common_pattern(self, sample_task_traces):
        miner = DAGMiner(min_support=0.5, max_nodes=10)
        skills = miner.mine(sample_task_traces)
        assert len(skills) >= 1
        skill = skills[0]
        assert "dag_definition" in skill
        dag = json.loads(skill["dag_definition"])
        assert len(dag["nodes"]) >= 2

    def test_skill_has_canary_status(self, sample_task_traces):
        miner = DAGMiner(min_support=0.5, max_nodes=10)
        skills = miner.mine(sample_task_traces)
        for skill in skills:
            assert skill["status"] == "canary"

    def test_empty_traces_returns_empty(self):
        miner = DAGMiner()
        skills = miner.mine([])
        assert skills == []

    def test_below_support_threshold(self, sample_task_traces):
        miner = DAGMiner(min_support=0.99, max_nodes=10)
        skills = miner.mine(sample_task_traces)
        assert skills == []

    def test_extracts_param_template(self, sample_task_traces):
        miner = DAGMiner(min_support=0.5, max_nodes=10)
        skills = miner.mine(sample_task_traces)
        assert len(skills) >= 1
        # At least one skill should have a non-empty param_template
        templates = [json.loads(s["param_template"]) for s in skills]
        non_empty = [t for t in templates if t]
        assert len(non_empty) >= 1

    def test_branching_dag_detects_common_core(self, branching_traces):
        """The shared search_law prefix across all 5 tasks should be found."""
        miner = DAGMiner(min_support=0.6, max_nodes=10)
        skills = miner.mine(branching_traces)
        assert len(skills) >= 1
        names = {s["name"] for s in skills}
        assert any("search_law" in n for n in names)

    def test_skill_name_uses_topological_order(self, sample_task_traces):
        miner = DAGMiner(min_support=0.5, max_nodes=10)
        skills = miner.mine(sample_task_traces)
        for s in skills:
            assert " → " in s["name"]

    def test_no_tasks_returns_empty(self):
        """Traces with no task_roots should yield no skills."""
        traces = [
            {"trace_id": "a1", "parent_trace_id": None,
             "tool_name": "x", "trace_type": "atomic", "success": 1},
        ]
        miner = DAGMiner(min_support=0.1)
        skills = miner.mine(traces)
        assert skills == []
