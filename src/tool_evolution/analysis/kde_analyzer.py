import json
from collections import Counter
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class KDEAnalyzer:
    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self._distributions: dict[tuple, dict] = {}

    def _param_type(self, values: list) -> str:
        sample = values[0]
        if isinstance(sample, bool):
            return "bool"
        if isinstance(sample, (int, float)):
            return "int" if isinstance(sample, int) else "float"
        if isinstance(sample, str):
            return "str"
        if isinstance(sample, list):
            return "list"
        return "unknown"

    def analyze(self, tool_name: str, tool_version: str, params_list: list[dict]) -> dict[str, dict]:
        if len(params_list) < self.min_samples:
            return {}

        keys = list(params_list[0].keys())
        result = {}
        for key in keys:
            values = [p[key] for p in params_list if key in p]
            if not values:
                continue
            ptype = self._param_type(values)

            if ptype == "list":
                continue
            elif ptype == "bool":
                dist = self._analyze_bool(values)
            elif ptype == "str":
                dist = self._analyze_str(values)
            else:
                dist = self._analyze_numeric(values)

            dist["param_type"] = ptype
            dist["sample_count"] = len(values)
            result[key] = dist

        self._distributions[(tool_name, tool_version)] = result
        return result

    def _analyze_numeric(self, values: list) -> dict:
        arr = np.array(values, dtype=float)
        try:
            kde = gaussian_kde(arr)
            x_grid = np.linspace(arr.min(), arr.max(), 200)
            density = kde(x_grid)
            peak_idx = np.argmax(density)
            default = float(x_grid[peak_idx])
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
            return {
                "default_value": round(default, 4),
                "lower_bound": round(ci_low, 4),
                "upper_bound": round(ci_high, 4),
            }
        except Exception:
            mean = float(arr.mean())
            std = float(arr.std())
            return {
                "default_value": round(mean, 4),
                "lower_bound": round(mean - 2 * std, 4),
                "upper_bound": round(mean + 2 * std, 4),
            }

    def _analyze_str(self, values: list) -> dict:
        unique = list(set(values))
        if len(unique) <= 3:
            counter = Counter(values)
            most_common = counter.most_common(1)[0][0]
            return {"default_value": most_common, "unique_count": len(counter), "method": "frequency"}

        try:
            vec = TfidfVectorizer(max_features=100, analyzer="char_wb", ngram_range=(2, 4))
            tfidf_matrix = vec.fit_transform(values)
            n_clusters = min(5, len(unique))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(tfidf_matrix)
            cluster_counts = Counter(labels)
            dominant_cluster = cluster_counts.most_common(1)[0][0]
            cluster_mask = labels == dominant_cluster
            cluster_values = [values[i] for i, m in enumerate(cluster_mask) if m]
            counter = Counter(cluster_values)
            most_common = counter.most_common(1)[0][0]
            return {
                "default_value": most_common,
                "unique_count": len(unique),
                "n_clusters": n_clusters,
                "dominant_cluster_ratio": round(cluster_counts[dominant_cluster] / len(values), 3),
                "method": "tfidf_kmeans",
            }
        except Exception:
            counter = Counter(values)
            most_common = counter.most_common(1)[0][0]
            return {"default_value": most_common, "unique_count": len(counter), "method": "frequency_fallback"}

    def _analyze_bool(self, values: list) -> dict:
        true_count = sum(1 for v in values if v)
        return {"default_value": true_count > len(values) / 2}

    def get_defaults(self, tool_name: str, tool_version: str) -> dict[str, object]:
        dists = self._distributions.get((tool_name, tool_version), {})
        return {k: v["default_value"] for k, v in dists.items()}
