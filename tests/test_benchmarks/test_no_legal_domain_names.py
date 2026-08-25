"""去领域化锁死测试：生产/评测/测试代码与 benchmark 数据中法律领域旧名零残留。

红灯记录（v1 执行中途真实发生）：改名进行到一半时 test_relation_store 的
law_name 字段断言失败（1 failed, 185 passed）——本测试将该约束固化为永久门禁。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_NAMES = [
    "search_law", "get_law_detail", "analyze_compliance", "generate_report",
    "run_compliance_check", "law_name",
    "劳动合同", "劳动法", "加班费", "查法规", "审合规", "出报告",
    "compliance",  # analyze_compliance 改名后该词不应以任何形式再出现
]


def _scan(paths) -> list[str]:
    offenders = []
    for p in paths:
        if p.name == "test_no_legal_domain_names.py":
            continue  # 排除自身（LEGACY_NAMES 定义与本测试文档必然包含旧名）
        text = p.read_text(encoding="utf-8")
        for name in LEGACY_NAMES:
            if name in text:
                offenders.append(f"{p.relative_to(ROOT)}:{name}")
    return offenders


class TestNoLegalDomainNames:
    def test_src_clean(self):
        offenders = _scan((ROOT / "src").rglob("*.py"))
        assert not offenders, f"src 法律领域残留: {offenders}"

    def test_scripts_clean(self):
        offenders = _scan((ROOT / "scripts").rglob("*.py"))
        assert not offenders, f"scripts 法律领域残留: {offenders}"

    def test_tests_clean(self):
        offenders = _scan((ROOT / "tests").rglob("*.py"))
        assert not offenders, f"tests 法律领域残留: {offenders}"

    def test_benchmark_tasks_clean(self):
        tasks = json.loads(
            (ROOT / "scripts" / "benchmark_tasks.json").read_text(encoding="utf-8"))
        serialized = json.dumps(tasks, ensure_ascii=False)
        offenders = [n for n in LEGACY_NAMES if n in serialized]
        assert not offenders, f"benchmark_tasks 法律领域残留: {offenders}"
