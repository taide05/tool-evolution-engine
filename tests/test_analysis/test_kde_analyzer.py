from tool_evolution.analysis.kde_analyzer import KDEAnalyzer


class TestKDEAnalyzer:
    def test_analyze_numeric_param(self):
        ana = KDEAnalyzer(min_samples=5)
        params_list = [
            {"max_results": i} for i in [10, 12, 15, 14, 11, 13, 14, 12, 10, 15]
        ]
        result = ana.analyze("search", "1.0.0", params_list)
        assert "max_results" in result
        dist = result["max_results"]
        assert dist["param_type"] == "int"
        assert dist["sample_count"] >= 5
        assert 10 <= dist["default_value"] <= 15
        assert dist["lower_bound"] <= dist["default_value"] <= dist["upper_bound"]

    def test_insufficient_data(self):
        ana = KDEAnalyzer(min_samples=30)
        params_list = [{"x": i} for i in range(5)]
        result = ana.analyze("t", "1.0.0", params_list)
        assert result == {}  # insufficient data, no output

    def test_string_param_clusters(self):
        ana = KDEAnalyzer(min_samples=3)
        params_list = [
            {"lang": lang} for lang in
            ["zh", "zh", "zh", "zh", "en", "en", "en", "ja", "ja", "ko"]
        ]
        result = ana.analyze("t", "1.0.0", params_list)
        if "lang" in result:
            dist = result["lang"]
            assert dist["param_type"] == "str"
            assert dist["default_value"] == "zh"

    def test_bool_param_mode(self):
        ana = KDEAnalyzer(min_samples=3)
        params_list = [{"verbose": True} for _ in range(7)] + [{"verbose": False} for _ in range(3)]
        result = ana.analyze("t", "1.0.0", params_list)
        assert "verbose" in result
        assert result["verbose"]["default_value"] is True

    def test_list_param_skipped(self):
        ana = KDEAnalyzer(min_samples=3)
        params_list = [{"tags": ["a", "b"]} for _ in range(10)]
        result = ana.analyze("t", "1.0.0", params_list)
        assert "tags" not in result

    def test_get_defaults(self):
        ana = KDEAnalyzer(min_samples=5)
        params_list = [
            {"max_results": i, "lang": "zh" if i % 2 == 0 else "en"}
            for i in range(10, 30)
        ]
        ana.analyze("search", "1.0.0", params_list)
        defaults = ana.get_defaults("search", "1.0.0")
        assert "max_results" in defaults
