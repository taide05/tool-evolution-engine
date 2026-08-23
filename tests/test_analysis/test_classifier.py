import json
import pytest
import tempfile
from pathlib import Path
from tool_evolution.analysis.classifier import FailureClassifier
from tool_evolution.collection.schemas import ErrorType


@pytest.fixture
def sample_traces():
    return [
        {"tool_name": "search", "params": json.dumps({"q": "test", "n": 5}),
         "error_type": "param_error", "error_message": "missing required param: query",
         "created_at": "2026-07-01T10:00:00"},
        {"tool_name": "search", "params": json.dumps({"query": "x"*100, "n": 5}),
         "error_type": "param_error", "error_message": "query too long",
         "created_at": "2026-07-01T11:00:00"},
        {"tool_name": "github_api", "params": json.dumps({"endpoint": "/repos"}),
         "error_type": "permission_denied", "error_message": "401 Unauthorized",
         "created_at": "2026-07-01T12:00:00"},
        {"tool_name": "github_api", "params": json.dumps({"endpoint": "/user"}),
         "error_type": "permission_denied", "error_message": "403 Forbidden",
         "created_at": "2026-07-01T13:00:00"},
        {"tool_name": "github_api", "params": json.dumps({"endpoint": "/search"}),
         "error_type": "quota_exhausted", "error_message": "API rate limit exceeded",
         "created_at": "2026-07-01T14:00:00"},
        {"tool_name": "arxiv_api", "params": json.dumps({"query": "transformer"}),
         "error_type": "timeout", "error_message": "Request timed out after 30s",
         "created_at": "2026-07-01T15:00:00"},
        {"tool_name": "official_docs", "params": json.dumps({"url": "/api/v2"}),
         "error_type": "service_unavailable", "error_message": "503 Service Unavailable",
         "created_at": "2026-07-01T16:00:00"},
        {"tool_name": "official_docs", "params": json.dumps({}),
         "error_type": "service_unavailable", "error_message": "502 Bad Gateway",
         "created_at": "2026-07-01T17:00:00"},
    ] * 3  # 24 samples


class TestFailureClassifier:
    def test_train_and_predict(self, sample_traces):
        clf = FailureClassifier()
        clf.train(sample_traces)
        pred = clf.predict({
            "tool_name": "github_api", "params": json.dumps({"endpoint": "/repos"}),
            "error_message": "401 Unauthorized"
        })
        assert pred == ErrorType.PERMISSION_DENIED

    def test_predict_timeout(self, sample_traces):
        clf = FailureClassifier()
        clf.train(sample_traces)
        pred = clf.predict({
            "tool_name": "arxiv_api", "params": json.dumps({"query": "ml"}),
            "error_message": "timeout after 30s"
        })
        assert pred == ErrorType.TIMEOUT

    def test_feature_importance(self, sample_traces):
        clf = FailureClassifier()
        clf.train(sample_traces)
        importance = clf.feature_importance()
        assert len(importance) > 0
        assert "tool_name" in importance

    def test_feature_importance_no_dead_features(self, sample_traces):
        clf = FailureClassifier()
        clf.train(sample_traces)
        names = clf.feature_importance().keys()
        assert "hour_of_day" not in names
        assert "param_count" in names
        assert "has_cjk" in names
        assert "tool_name" in names

    def test_save_and_load(self, sample_traces):
        clf = FailureClassifier()
        clf.train(sample_traces)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "classifier.joblib"
            clf.save(path)
            clf2 = FailureClassifier()
            clf2.load(path)
            pred = clf2.predict({
                "tool_name": "arxiv_api", "params": json.dumps({}),
                "error_message": "timeout"
            })
            assert pred == ErrorType.TIMEOUT
