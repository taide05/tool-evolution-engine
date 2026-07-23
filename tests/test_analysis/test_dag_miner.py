import json
import pytest
from tool_evolution.analysis.dag_miner import DAGMiner


@pytest.fixture
def sample_task_traces():
    traces = []
    for task_id in range(20):
        root = {"trace_id": f"root-{task_id}", "parent_trace_id": None,
                "tool_name": "run_check", "trace_type": "task_root", "success": 1}
        c1 = {"trace_id": f"c1-{task_id}", "parent_trace_id": f"root-{task_id}",
              "tool_name": "search_law", "trace_type": "atomic", "success": 1}
        c2 = {"trace_id": f"c2-{task_id}", "parent_trace_id": f"root-{task_id}",
              "tool_name": "analyze_compliance", "trace_type": "atomic", "success": 1}
        c3 = {"trace_id": f"c3-{task_id}", "parent_trace_id": f"root-{task_id}",
              "tool_name": "generate_report", "trace_type": "atomic", "success": 1}
        traces.extend([root, c1, c2, c3])
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
