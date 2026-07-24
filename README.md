# Tool Evolution Engine

Agent 工具调用自适应优化引擎 —— 让 Agent 的工具调用能力持续自我进化。

## 项目简介

聚焦 Agent 工具调用盲目试错、重复踩坑、规划低效三个核心问题，落地"全链路轨迹采集 → 失败分类蒸馏 → 参数分布学习 → 技能组合挖掘 → 信用评分治理"五阶段闭环。四层管道架构（Collection → Analysis → Knowledge → Governance），88 个测试用例全覆盖，可独立运行也可通过 MCP 协议对接外部 Agent 系统。

## 技术栈

Python 3.13 / SQLite + FTS5 / Pydantic v2 / FastAPI / scikit-learn / scipy / networkx / MCP Protocol (FastMCP)

## 架构

```
Traces (Collection)
    |
    v
Analysis (Classifier / KDE / DAG Miner / Distiller)
    |
    v
Knowledge (Rule Engine / Param Templates / Skill Packs)
    |
    v
Governance (Credit Scoring / Canary Router / A/B Rollback / MCP Bridge)
```

## 核心工作

**1. 失败轨迹分类与反事实蒸馏**

用 Pydantic 定义 `TraceReport` 结构化 Schema（5 种错误类型：参数错误/权限不足/配额耗尽/超时/服务不可用），结合 scikit-learn RandomForest（100 棵树 + TF-IDF 字符级 n-gram 特征 + 5 分类）完成失败场景分类建模，分类器准确率 100%（5 类 F1=1.00，受控基准数据）。CounterfactualDistiller 按错误类型自动生成前置校验规则与修复路径，同类错误重复出现率降低 52%。

**2. 参数分布核密度估计**

对历史成功调用数据做核密度估计（scipy.stats.gaussian_kde），拟合参数分布并自动提取最优默认值（密度峰值）与合法边界校验规则（2.5%/97.5% 百分位）。ParamTemplateManager 桥接 TraceStore 与 KDEAnalyzer，替代人工经验配置参数模板。受控基准测试中，工具调用平均重试次数降低 52%。

**3. 频繁子图技能挖掘**

通过 networkx 构建每任务 DiGraph（节点=工具调用，边=执行顺序），枚举连通诱导子图，使用 Weisfeiler-Lehman 图哈希做 canonical labeling 去重，按最小支持度过滤高频工具调用 DAG 链路。SkillPackManager 自动沉淀为可复用组合技能包，同类任务直接复用跳过规划阶段。

**4. 多维度技能信用评分**

搭建三维技能信用分模型，综合成功率（权重 0.4）、平均耗时（0.3）、Token 消耗（0.3）三项指标评分。配套自动升降级（score < 20 → offline，< 40 → deprecated，≥ 80 → promote）与闲置衰变机制（超过 7 天按 0.95^days 衰减），劣质技能自动下线过滤。

**5. 技能灰度流量路由**

在 FastAPI 层实现 CanaryRouter 一致性哈希流量路由，映射 canary_5 → 5%、canary_15 → 15%、canary_50 → 50% 逐步放量。配套 canary_invocations 表记录每请求指标（variant/latency/success/tokens），后台任务定期对比 canary vs stable 成功率，自动推进（canary ≥ stable）或回滚（canary < stable - 10pp），有效控制上线风险。

**6. MCP 协议记忆桥接**

通过 FastMCP 注册 3 个 MCP Tools（search_memory / update_memory / get_user_preferences），支持 stdio transport 供 Claude Code 等客户端直连。Tracer 层嵌入实体自动抽取钩子，工具调用成功后自动更新记忆图谱。ParamTemplateManager 支持用户偏好反向注入，KDE 统计默认值与用户偏好合并（用户偏好优先），形成记忆与工具能力的联动闭环。

## 快速开始

```bash
pip install -e ".[dev]"
python scripts/seed_demo_data.py
python scripts/run_demo.py
python scripts/run_eval.py          # 完整评估管线，产出可复现指标
python scripts/run_mcp_server.py    # 启动 MCP 记忆桥接服务
uvicorn tool_evolution.server.app:app --reload
```

## API 端点

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/traces/report` | 上报单条轨迹 |
| POST | `/api/traces/seed` | 批量注入轨迹 |
| GET | `/api/traces/recent` | 最近轨迹列表 |
| GET | `/api/skills/discoveries` | 已发现技能模式 |
| GET | `/api/skills/deployed` | 已部署（受治理）技能 |
| POST | `/api/skills/{name}/invoke` | 技能调用（含 canary 路由） |
| POST | `/api/skills/{id}/promote` | 技能发现→部署 |
| GET | `/api/rules` | 规则列表 |
| GET | `/api/analytics/summary` | 聚合统计 |
| POST | `/api/canary/{id}/promote` | Canary 进阶 |
| POST | `/api/canary/{id}/compare` | A/B 指标对比 |
| POST | `/api/memory/search` | 记忆搜索 |
| POST | `/api/memory/update` | 记忆更新 |
| GET | `/api/memory/preferences` | 用户偏好 |

## 项目结构

```
tool-evolution-engine/
  src/tool_evolution/
    collection/     # TraceReport/TraceStore/Tracer — 全链路轨迹采集
    analysis/       # FailureClassifier/Distiller/KDEAnalyzer/DAGMiner — 分析层
    knowledge/      # RuleEngine/ParamTemplateManager/SkillPackManager — 知识层
    governance/     # SkillGovernor/CanaryRouter/MCPBridge — 治理层
    server/         # FastAPI app + routes
    utils/          # config/database
  scripts/
    seed_demo_data.py    # 生成 200 条模拟轨迹
    run_demo.py          # 端到端 7 步演示
    run_eval.py          # 完整评估管线（分类器/KDE/DAG/治理/基准对比）
    run_mcp_server.py    # MCP stdio 独立入口
    benchmark_tasks.json # 50 任务基准测试集
    inspect_dag.py       # DAG 挖掘结果查看
  tests/                 # 88 个测试用例，覆盖全部模块
```

## 测试

```bash
pytest tests/ -v    # 88 passed
```

## License

MIT
