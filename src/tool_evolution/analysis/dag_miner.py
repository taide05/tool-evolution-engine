import json
from collections import Counter, defaultdict
import networkx as nx


class DAGMiner:
    def __init__(self, min_support: float = 0.05, max_nodes: int = 10):
        self.min_support = min_support
        self.max_nodes = max_nodes

    def mine(self, traces: list[dict]) -> list[dict]:
        if not traces:
            return []

        trees = self._group_by_root(traces)
        total_tasks = len(trees)

        dag_counter = Counter()
        dag_params = defaultdict(list)

        for root, children in trees.items():
            dag = self._build_dag(root, children, traces)
            if dag and len(dag["nodes"]) >= 2 and len(dag["nodes"]) <= self.max_nodes:
                key = self._dag_fingerprint(dag)
                dag_counter[key] += 1
                dag_params[key].append(dag)

        min_count = max(1, int(total_tasks * self.min_support))
        skills = []
        for key, count in dag_counter.most_common():
            if count < min_count:
                continue
            sample_dag = dag_params[key][0]
            skills.append({
                "name": self._name_dag(sample_dag),
                "dag_definition": json.dumps(sample_dag),
                "param_template": json.dumps({}),
                "frequency": round(count / total_tasks, 4),
                "status": "canary",
            })

        return skills

    def _group_by_root(self, traces: list[dict]) -> dict[str, list[dict]]:
        roots = [t for t in traces if t.get("trace_type") == "task_root" and t.get("parent_trace_id") is None]
        trees = {}
        for root in roots:
            children = [t for t in traces if t.get("parent_trace_id") == root["trace_id"]]
            trees[root["trace_id"]] = children
        return trees

    def _build_dag(self, root_id: str, children: list[dict], all_traces: list[dict]) -> dict | None:
        if not children:
            return None

        nodes = {}
        edges = []
        for c in children:
            name = c["tool_name"]
            if name not in nodes:
                nodes[name] = {"tool_name": name}

        ordered = sorted(children, key=lambda t: t.get("created_at", ""))
        for i in range(len(ordered) - 1):
            src = ordered[i]["tool_name"]
            dst = ordered[i + 1]["tool_name"]
            if src != dst:
                edges.append({"from": src, "to": dst})

        return {"nodes": list(nodes.values()), "edges": edges}

    def _dag_fingerprint(self, dag: dict) -> str:
        node_names = sorted(n["tool_name"] for n in dag["nodes"])
        edge_pairs = tuple(sorted((e["from"], e["to"]) for e in dag["edges"]))
        return f"{'|'.join(node_names)}::{edge_pairs}"

    def _name_dag(self, dag: dict) -> str:
        node_names = [n["tool_name"] for n in dag["nodes"]]
        return " → ".join(node_names[:5])
