"""End-to-end demo: seed data -> classifier train -> KDE analysis -> DAG mining -> skill governance -> summary."""
import asyncio
import json
import sys
sys.path.insert(0, "src")

from tool_evolution.utils.database import get_connection, init_db
from tool_evolution.collection.store import TraceStore
from tool_evolution.analysis.classifier import FailureClassifier
from tool_evolution.analysis.distiller import CounterfactualDistiller
from tool_evolution.analysis.kde_analyzer import KDEAnalyzer
from tool_evolution.analysis.dag_miner import DAGMiner
from tool_evolution.knowledge.rule_engine import RuleEngine
from tool_evolution.knowledge.param_template import ParamTemplateManager
from tool_evolution.knowledge.skill_pack import SkillPackManager
from tool_evolution.governance.governor import SkillGovernor


async def main():
    conn = await get_connection()
    await init_db(conn)

    store = TraceStore(conn)

    # 1. Data overview
    failures = await store.count_failures(None)
    total = len(await store.get_all_traces(limit=10000))
    print(f"=== Step 1: Data overview ===")
    print(f"Total traces: {total}, Failures: {failures}, Failure rate: {failures/max(total,1):.1%}")

    # 2. Classifier training
    print(f"\n=== Step 2: Failure classifier training ===")
    cursor = await conn.execute("SELECT * FROM trajectories WHERE success=0")
    failed_rows = [dict(r) for r in await cursor.fetchall()]
    if failed_rows:
        clf = FailureClassifier()
        clf.train(failed_rows)
        importance = clf.feature_importance()
        top3 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 features: {top3}")

    # 3. Distill rules
    print(f"\n=== Step 3: Counterfactual distillation ===")
    distiller = CounterfactualDistiller()
    rules = distiller.distill_batch(failed_rows[:20])
    engine = RuleEngine(conn)
    for rule in rules:
        await engine.add_rule(rule)
    print(f"Generated {len(rules)} fix rules")

    # 4. KDE parameter analysis
    print(f"\n=== Step 4: KDE parameter analysis ===")
    mgr = ParamTemplateManager(conn)
    for tool in ["search_law", "get_law_detail"]:
        tmpl = await mgr.generate(tool, "1.0.0")
        if tmpl:
            print(f"  {tool}: discovered {len(tmpl)} param distributions")

    # 5. DAG mining
    print(f"\n=== Step 5: DAG frequent subgraph mining ===")
    all_traces = await store.get_all_traces(limit=10000)
    miner = DAGMiner(min_support=0.05, max_nodes=10)
    skills = miner.mine(all_traces)
    skill_mgr = SkillPackManager(conn)
    for skill in skills:
        await skill_mgr.add_discovery(skill)
    print(f"Discovered {len(skills)} skill patterns")
    for s in skills:
        print(f"  - {s['name']} (frequency: {s['frequency']:.1%})")

    # 6. Scoring & governance
    print(f"\n=== Step 6: Skill governance ===")
    discoveries = await skill_mgr.list_discoveries()
    for disc in discoveries[:3]:
        dep_id = await skill_mgr.promote_to_deployed(disc["id"])
        gov = SkillGovernor(conn)
        score = await gov.score_skill(dep_id)
        print(f"  Skill {disc['name']}: credit score = {score:.1f}")

    # 7. Summary
    print(f"\n=== Step 7: Summary ===")
    deployed = await skill_mgr.list_deployed()
    active_count = sum(1 for s in deployed if s["status"] == "active")
    canary_count = sum(1 for s in deployed if "canary" in s["status"])
    print(f"Active skills: {active_count}, Canary skills: {canary_count}")

    cursor = await conn.execute("SELECT COUNT(*) FROM rules WHERE status='active'")
    rule_count = (await cursor.fetchone())[0]
    print(f"Active rules: {rule_count}")
    print("\n=== Demo complete ===")

    await conn.close()


asyncio.run(main())
