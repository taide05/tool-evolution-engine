import json
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx


class DAGMiner:
    """Frequent subgraph miner for tool-call DAG patterns.

    Builds a nx.DiGraph per task from trace data, enumerates connected
    induced subgraphs, and uses Weisfeiler-Lehman graph hashing for
    canonical labeling to find recurring tool-call patterns.
    """

    def __init__(self, min_support: float = 0.05, max_nodes: int = 10):
        self.min_support = min_support
        self.max_nodes = max_nodes

    def mine(self, traces: list[dict]) -> list[dict]:
        if not traces:
            return []

        trees = self._group_by_root(traces)
        total_tasks = len(trees)

        task_graphs: dict[str, nx.DiGraph] = {}
        for root_id, children in trees.items():
            g = self._build_dag(root_id, children)
            if g is not None and 2 <= len(g) <= self.max_nodes:
                task_graphs[root_id] = g

        if not task_graphs:
            return []

        # Enumerate all connected subgraphs and group by WL hash
        wl_counter: Counter[str] = Counter()
        wl_instances: dict[str, list[tuple[str, nx.DiGraph, nx.DiGraph]]] = defaultdict(list)

        for root_id, g in task_graphs.items():
            for subg in self._enumerate_connected_subgraphs(g):
                wl_hash = nx.weisfeiler_lehman_graph_hash(
                    subg, node_attr="label", iterations=3
                )
                wl_counter[wl_hash] += 1
                wl_instances[wl_hash].append((root_id, subg, g))

        min_count = max(1, int(total_tasks * self.min_support))
        skills = []
        for wl_hash, count in wl_counter.most_common():
            if count < min_count:
                continue
            instances = wl_instances[wl_hash]
            sample_subg = instances[0][1]
            dag_def = self._graph_to_dict(sample_subg)
            name = self._name_dag(sample_subg)
            param_template = self._extract_param_template(instances)

            skills.append({
                "name": name,
                "dag_definition": json.dumps(dag_def),
                "param_template": json.dumps(param_template),
                "frequency": round(count / total_tasks, 4),
                "status": "canary",
            })

        return skills

    # ── private helpers ──────────────────────────────────────────────

    def _group_by_root(self, traces: list[dict]) -> dict[str, list[dict]]:
        roots = [
            t for t in traces
            if t.get("trace_type") == "task_root" and t.get("parent_trace_id") is None
        ]
        trees: dict[str, list[dict]] = {}
        for root in roots:
            children = [
                t for t in traces
                if t.get("parent_trace_id") == root["trace_id"]
            ]
            trees[root["trace_id"]] = children
        return trees

    def _build_dag(self, root_id: str, children: list[dict]) -> nx.DiGraph | None:
        """Build a nx.DiGraph from a single task's atomic traces.

        Nodes represent tools called; edges represent execution order
        (sorted by rowid / creation order). The task_root trace itself
        is excluded — only the child tool calls form the graph.
        """
        if not children:
            return None

        ordered = sorted(children, key=lambda t: t.get("rowid", 0))
        g = nx.DiGraph()

        tool_counts: Counter[str] = Counter()
        node_ids: list[str] = []

        for c in ordered:
            tool_name = c["tool_name"]
            count = tool_counts[tool_name]
            tool_counts[tool_name] += 1
            node_id = tool_name if count == 0 else f"{tool_name}_{count}"

            params = c.get("params", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}

            g.add_node(
                node_id,
                label=tool_name,
                tool_name=tool_name,
                params=params,
                param_keys=frozenset(params.keys()) if params else frozenset(),
            )
            node_ids.append(node_id)

        # Sequential execution edges
        for i in range(len(node_ids) - 1):
            g.add_edge(node_ids[i], node_ids[i + 1])

        return g

    def _enumerate_connected_subgraphs(self, g: nx.DiGraph) -> list[nx.DiGraph]:
        """Return all weakly-connected induced subgraphs of size 2..max_nodes."""
        subgraphs: list[nx.DiGraph] = []
        nodes = list(g.nodes())
        n = len(nodes)

        for size in range(2, min(self.max_nodes, n) + 1):
            for node_subset in combinations(nodes, size):
                subg = g.subgraph(node_subset).copy()
                if nx.is_weakly_connected(subg):
                    subgraphs.append(subg)

        return subgraphs

    def _graph_to_dict(self, g: nx.DiGraph) -> dict:
        nodes = [
            {"tool_name": attrs.get("tool_name", n)}
            for n, attrs in g.nodes(data=True)
        ]
        edges = [{"from": u, "to": v} for u, v in g.edges()]
        return {"nodes": nodes, "edges": edges}

    def _name_dag(self, g: nx.DiGraph) -> str:
        try:
            order = list(nx.topological_sort(g))
        except nx.NetworkXUnfeasible:
            order = list(g.nodes())
        names = [g.nodes[n].get("tool_name", n) for n in order]
        return " → ".join(names[:5])

    def _extract_param_template(
        self, instances: list[tuple[str, nx.DiGraph, nx.DiGraph]]
    ) -> dict:
        """Collect params across instances and compute per-param summaries.

        For each tool node in the pattern, aggregates param values from
        every matched instance and emits type-aware summary statistics
        (median/range for numeric, frequency for string, true-ratio for bool).
        """
        collected: dict[str, dict] = {}  # tool_name -> {"samples": int, "params": {name: [values]}}

        for _, subg, _ in instances[:50]:
            for node_id in subg.nodes():
                tool_name = subg.nodes[node_id].get("tool_name", node_id)
                if tool_name not in collected:
                    collected[tool_name] = {"samples": 0, "params": defaultdict(list)}
                collected[tool_name]["samples"] += 1

                params = subg.nodes[node_id].get("params", {})
                for k, v in params.items():
                    collected[tool_name]["params"][k].append(v)

        result: dict = {}
        for tool_name, data in collected.items():
            entry: dict = {"sample_count": data["samples"], "params": {}}
            for param_name, values in data["params"].items():
                if not values:
                    continue
                if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    sv = sorted(values)
                    entry["params"][param_name] = {
                        "type": "numeric",
                        "median": sv[len(sv) // 2],
                        "min": sv[0],
                        "max": sv[-1],
                    }
                elif all(isinstance(v, bool) for v in values):
                    true_ratio = sum(1 for v in values if v) / len(values)
                    entry["params"][param_name] = {
                        "type": "bool",
                        "default": true_ratio > 0.5,
                        "true_ratio": round(true_ratio, 3),
                    }
                else:
                    freq = Counter(str(v) for v in values)
                    entry["params"][param_name] = {
                        "type": "string",
                        "most_common": freq.most_common(3),
                        "unique_count": len(freq),
                    }
            result[tool_name] = entry

        return result
