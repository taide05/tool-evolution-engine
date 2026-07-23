import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from ..collection.schemas import ErrorType


class FailureClassifier:
    def __init__(self):
        self.pipeline: Pipeline | None = None
        self._clf: RFC | None = None
        self.label_encoder = LabelEncoder()

    def _extract_features(self, traces: list[dict]) -> tuple[list[dict], np.ndarray]:
        X_data = []
        for t in traces:
            params = json.loads(t.get("params", "{}"))
            feat = {
                "param_count": len(params),
                "has_auth_header": 1 if any(k.lower() in ("auth", "token", "api_key") for k in params) else 0,
                "hour_of_day": int(t.get("created_at", "T00:00:00").split("T")[1].split(":")[0]) if "T" in t.get("created_at", "") else 0,
            }
            X_data.append(feat)

        tool_names = [t.get("tool_name", "") for t in traces]
        error_msgs = [t.get("error_message", "") for t in traces]
        labels = self.label_encoder.fit_transform([t["error_type"] for t in traces])

        combined = []
        for feat, tn, em in zip(X_data, tool_names, error_msgs):
            combined.append({
                "tool_name": tn,
                "error_message": em,
                **{f"feat_{k}": v for k, v in feat.items()}
            })

        return combined, labels

    def train(self, traces: list[dict]) -> None:
        combined, labels = self._extract_features(traces)
        texts = [f"{c['tool_name']} {c['error_message']}" for c in combined]
        num_features = np.array([[c[f"feat_{k}"] for k in ("param_count", "has_auth_header", "hour_of_day")] for c in combined])

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500)),
        ])

        tfidf_matrix = self.pipeline.named_steps["tfidf"].fit_transform(texts)
        X = np.hstack([tfidf_matrix.toarray(), num_features])
        self._clf = RFC(n_estimators=100, random_state=42)
        self._clf.fit(X, labels)

    def predict(self, trace: dict) -> ErrorType:
        if self._clf is None:
            raise RuntimeError("Classifier not trained. Call train() first.")
        params = json.loads(trace.get("params", "{}"))
        text = f"{trace.get('tool_name', '')} {trace.get('error_message', '')}"
        tfidf_vec = self.pipeline.named_steps["tfidf"].transform([text]).toarray()
        num = np.array([[
            len(params),
            1 if any(k.lower() in ("auth", "token", "api_key") for k in params) else 0,
            0
        ]])
        X = np.hstack([tfidf_vec, num])
        label_idx = self._clf.predict(X)[0]
        label = self.label_encoder.inverse_transform([label_idx])[0]
        return ErrorType(label)

    def feature_importance(self) -> dict[str, float]:
        if self._clf is None:
            raise RuntimeError("Classifier not trained.")
        tfidf_names = self.pipeline.named_steps["tfidf"].get_feature_names_out()
        all_names = list(tfidf_names) + ["param_count", "has_auth_header", "hour_of_day"]
        return dict(zip(all_names, self._clf.feature_importances_))

    def save(self, path: Path) -> None:
        import joblib as jl
        jl.dump({"pipeline": self.pipeline, "clf": self._clf, "le": self.label_encoder}, path)

    def load(self, path: Path) -> None:
        import joblib as jl
        data = jl.load(path)
        self.pipeline = data["pipeline"]
        self._clf = data["clf"]
        self.label_encoder = data["le"]
