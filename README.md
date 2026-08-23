# Tool Evolution Engine — Agent 工具调用自适应优化引擎

> TEE 本身不是 Agent——它不包含 LLM 调用、prompt engineering 或 Agent 编排。它是运行在 Agent 和工具之间的**数据分析管道**：采集 Agent 的工具调用轨迹，用统计/机器学习/图算法分析优化，把结果反馈给 Agent 系统。

让 Agent 的工具调用能力从"瞎试"变成"越用越聪明"——采集每次工具调用的轨迹数据，分析失败模式、学习参数分布、挖掘技能组合，最终通过信用评分和灰度路由安全地把优化部署上线。

## 为什么值得关注

Agent 项目通常关注 prompt 怎么写、模型选哪个，很少关注工具调用本身的质量。现实是：

- 同一个工具被调了 1000 次，传错参数的模式是收敛的——但你不知道
- Agent A 和 Agent B 都做"查法规→审合规→出报告"这个三件套——但你发现不了这个规律
- 上线了一个新版本的搜索工具，你怎么判断它是更好了还是更坏了？

Tool Evolution Engine 在 Agent 的工具层和 LLM 之间加了一个"运维+优化"层，用数据驱动的方式回答这些问题。它不依赖特定 Agent 框架——只要你的工具调用能产出结构化轨迹，就能接进来。

## 核心特性

- **全链路轨迹采集**：Pydantic 结构化 Schema（5 种错误类型）+ SQLite FTS5 全文索引，不只是记日志，是可以搜索、聚合、分析的结构化数据
- **失败模式自动分类**：Random Forest 分类器（TF-IDF 字符级 n-gram 特征 + 工具名 + 参数数量 + CJK 语言特征），把失败归入 5 类，不再人肉排查
- **反事实规则蒸馏**：从失败轨迹反推"应该怎么做"，按错误类型自动生成前置校验规则（参数范围校验/授权检查/重试策略/超时阈值/熔断器）
- **参数分布学习**：对历史成功调用的参数做核密度估计（scipy gaussian_kde），自动给出默认值和合法边界（2.5%/97.5% 百分位），替代人工经验配置。支持用户偏好反向注入——MCP 记忆里存的偏好值优先于统计默认值。注：KDE 的价值在分布形状与 CI 边界（实测边界外 0.0%），点估计（mode）在 MAE 口径上不优于中位数——不做"默认值更准"声明
- **频繁子图技能挖掘**：networkx 构建每任务 DiGraph → Weisfeiler-Lehman 图哈希做同构去重 → 按最小支持度过滤高频工具调用 DAG 链路。比如自动发现"查法规→审合规→出报告"是高频任务的公共模式
- **3D 信用评分 + 自动升降级**：成功率(0.4) + 延迟(0.3) + Token 消耗(0.3)，< 20 分自动下线，> 80 分自动晋升，闲置 > 7 天按 0.95^days 衰减
- **一致哈希灰度路由 + A/B 回滚**：canary_5 → canary_15 → canary_50 → active 四段放量，每段要求 min 30 样本才决策。canary < stable - 10pp 自动回滚
- **MCP 协议记忆桥接**：3 个 MCP Tools（search_memory / update_memory / get_user_preferences），stdio transport 供 Claude Code 等客户端直连。Trace 采集成功后自动抽取实体更新记忆

## 快速开始

```bash
# 1. 安装
pip install -e ".[dev]"

# 2. 生成演示数据（约 200 条轨迹，7 种工具，25% 失败率，3 种预埋 DAG 模式）
python scripts/seed_demo_data.py

# 3. 跑完整演示管道（7 步：分类器→规则→KDE→DAG→治理→总结）
python scripts/run_demo.py

# 4. 跑评估管线（分类器 F1 + KDE 覆盖率 + DAG 召回 + 治理模拟 + 优化前后对比）
python scripts/run_eval.py

# 5.（可选）启动 API 服务（fail-closed 鉴权：必须先设置 TOOLEVO_API_KEY）
# PowerShell: $env:TOOLEVO_API_KEY="your-key"
uvicorn tool_evolution.server.app:app --reload

# 6.（可选）启动 MCP 记忆桥接
python scripts/run_mcp_server.py
```

## 架构概览

```
                    Agent 工具调用
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Collection（采集层）                             │
│  TraceReport (Pydantic) → Tracer → TraceStore (SQLite+FTS5) │
│  异步批量写入 + MCP 实体自动抽取钩子                         │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 2: Analysis（分析层）                               │
│  FailureClassifier    RandomForest 5 分类，TF-IDF 字符级特征 │
│  CounterfactualDistiller  失败 → 前置校验规则（5 类→5 种规则） │
│  KDEAnalyzer          scipy gaussian_kde 参数分布学习       │
│  DAGMiner             WL 图哈希 + 频繁子图挖掘              │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 3: Knowledge（知识层）                              │
│  RuleEngine           规则持久化 + hit/miss 反馈计数        │
│  ParamTemplateManager 桥接 TraceStore→KDE，注入用户偏好     │
│  SkillPackManager     DAG 发现→技能包→手动晋升到部署        │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Layer 4: Governance（治理层）                             │
│  SkillGovernor        3D 信用评分 + 自动升降级 + 闲置衰减    │
│  CanaryRouter         一致性哈希分流 + A/B 对比 + 自动回滚   │
│  MCPBridge            3 MCP Tools + 实体自动抽取            │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
              FastAPI 端点 + MCP stdio 接口
```

**数据闭环**：Agent 调工具 → Tracer 采集轨迹 → Classifier 分类失败 + KDE 学参数 + DAG Miner 挖模式 → RuleEngine/ParamTemplate/SkillPack 存储知识 → Governor 评分治理 → CanaryRouter 灰度部署 → 新版本上线后继续采集轨迹 → 循环。

## 关键技术决策

**用 Weisfeiler-Lehman 图哈希而非直接图匹配做 DAG 去重**。Agent 每次执行"查法规→审合规→出报告"时，tool_call 的 node_id 不同，但图结构相同。WL 哈希做的是图同构的 canonical labeling——相同结构的 DAG 映射到相同哈希，不依赖 node_id 或执行时间。这是计算机科学里解决"这两个图本质上是不是同一个"的标准方法。

**用一致性哈希而非随机分流做 Canary 路由**。`MD5(request_hash) % 100` 保证同一用户的请求始终路由到同一个变体。随机分流会导致用户在两次请求间看到不同的工具行为，A/B 对比数据被污染。一致性哈希是基础设施层解决"确定性路由"的经典方案。

**TF-IDF 字符级 n-gram(2,4) + has_cjk 特征检测**。中文错误消息在字符级 n-gram 中与英文无共享子串，单独增加 CJK 字符检测特征位（0/1），给模型显式的语言区分信号。在不依赖语言模型的条件下，字符级 n-gram + CJK 检测是最务实的跨语言文本特征方案。

**离线批处理架构，不做实时拦截**。TEE 设计为离线分析管道——轨迹采集后批量处理，不是 Agent 每次工具调用前的同步拦截器。原因：KDE 参数学习、DAG 子图枚举、信用评分更新都是统计计算，需要足够样本量才有意义（`min_samples=30`）。对实时性要求高的场景（如每次工具调用前必须校验参数），应该在 Agent 代码中直接嵌入 `CounterfactualDistiller` 产出的规则——这些规则是纯确定性逻辑，可以同步执行。全管道耗时 ~52-74s（1000 seed + 400 benchmark，2026-08-23 种子固化后两次运行实测），适合小时级/日级调度，不适合毫秒级在线决策。

**4 层架构 + 0 Agent + 参数上限的复杂度约束**。为什么是 4 层？采集→分析→知识→治理，每层对应数据加工的一个独立阶段，层间通过 Pydantic 模型传递，接口明确。拆成 5 层会把分析和知识割裂（比如 Classifier 的产出直接被 RuleEngine 消费，拆开只增加序列化开销），合并成 3 层会把存储和部署混在一起（知识库和灰度路由的职责完全不同）。为什么 0 Agent？失败分类（RF）、参数学习（KDE）、模式挖掘（WL 哈希）、规则蒸馏（确定性映射）——每一个都是确定性或统计方法能解决的子问题，引入 LLM 只会增加延迟和不确定性。`max_dag_nodes=10` 的根因是子图枚举的计算复杂度 O(2^n)：在 n=10 时最坏情况 ~1024 个子图/任务，在 n=15 时 ~32768——10 是保证枚举在百毫秒级完成的工程上限，同时覆盖了实际 Agent 工作流中 90%+ 的工具调用链路长度。

## 量化指标

> 评测规模：1000 seed tasks + 400 benchmark tasks（50 基础 × 8 参数变体），2026-08-23 run_eval.py 实测（增量零全链路 D-light→E→C→D-full→I→E'→C'→D-full'→V 门禁验证，种子固化 _MAIN_RNG=42 两次运行逐字段一致）。指标来源：template-data/metrics-snapshot.md（2026-08-23 快照）。

| 指标 | 值 | 说明 |
|------|-----|------|
| 分类器 Macro F1 | 0.982 | 5 分类，690 训练 / 297 测试，504 维特征（含 has_cjk） |
| 失败率下降 | -56.2%（率口径）/ -60.5%（次数口径） | 失败率 0.1624→0.0711；228→90 failures，400 benchmark，注入全部 5 种错误类型 |
| Token 消耗下降 | -44.6% | 225,500→124,950 tokens，KDE 默认值省略参数 + 规则前置校验 |
| 重试次数下降 | -60.5% | 228→90 retries，规则前置校验在调用前拦截参数错误 |
| 失败类型拆分 | 全部 5 种有对比 | param_error -64%、permission -44%、quota -63%、service_unavail -65%、timeout -66% |
| DAG 模式召回 | 100.0%（8/8） | 8 种预埋模式全找回，额外发现 10 个高频子模式 |
| 参数模板覆盖率 | 100%（7/7） | 7 种工具 28 个参数全生成 KDE 模板，CI 边界外比例 0.0% |
| 规则准确率 | 0%（如实披露） | 评测管道当前不产 rules（distiller 仅接 run_demo 演示路径）——接入 run_eval 属增量一后计划 |
| 灰度上线 | 3/3 技能走通全路径 | canary_5→canary_15→canary_50→active，310 calls/skill；A/B 实测回滚 True |
| 权重敏感性 | 40/30/30 最优（5P vs 3P vs 3P） | 三组权重对比验证，默认权重晋升最多 |
| 简化场景 RF vs 规则 | RF 84.8% vs 规则 33.3%（F1 0.809 vs 0.252） | 含拼写错误+中英混用+同义词，RF 显著优于规则 |
| 跨语言分类 | EN 100.0% / CN 33.3% | has_cjk 特征改善中文分类，char_wb 跨语言仍有限 |
| 测试覆盖 | 128 tests | 生产就绪修复 10 条全部测试锁死（鉴权/迁移/并发/容器） |
| 评测复现性 | 两次运行逐字段一致 | 种子固化（增量零新增保证） |

## 技术栈

Python 3.13 · SQLite + FTS5 · Pydantic v2 · FastAPI · scikit-learn（RandomForest + TF-IDF + KMeans） · scipy（gaussian_kde） · networkx（DiGraph + WL hashing） · FastMCP · aiosqlite · pytest-asyncio

## API 端点

| Method | Path | 说明 |
|--------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/traces/report` | 上报单条工具调用轨迹 |
| POST | `/api/traces/seed` | 批量注入轨迹（演示/测试用） |
| GET | `/api/traces/recent` | 最近轨迹分页查询 |
| GET | `/api/skills/discoveries` | 已发现的技能模式（DAG Miner 输出） |
| GET | `/api/skills/deployed` | 已部署技能（含信用分 + 状态） |
| POST | `/api/skills/{name}/invoke` | 路由决策（一致性哈希分流 stable/canary，不执行不写指标） |
| POST | `/api/skills/{id}/promote` | 技能发现 → 部署 |
| GET | `/api/rules` | 规则列表（按工具名过滤） |
| GET | `/api/analytics/summary` | 聚合分析摘要 |
| POST | `/api/canary/{id}/promote` | 灰度进阶（5→15→50→active） |
| POST | `/api/memory/search` | MCP 记忆搜索 |
| POST | `/api/memory/update` | MCP 记忆更新 |
| GET | `/api/memory/preferences` | 用户偏好查询 |

## 项目结构

```
src/tool_evolution/
├── collection/       # TraceReport + TraceStore + Tracer（采集层）
├── analysis/         # Classifier + Distiller + KDEAnalyzer + DAGMiner（分析层）
├── knowledge/        # RuleEngine + ParamTemplateManager + SkillPackManager（知识层）
├── governance/       # SkillGovernor + CanaryRouter + MCPBridge（治理层）
├── server/           # FastAPI app + 6 路由模块
└── utils/            # Config（pydantic-settings）+ Database（DDL）

scripts/
├── seed_demo_data.py      # 生成 200 条模拟轨迹（7 工具 + 8 DAG 预埋模式）
├── run_demo.py            # 端到端 7 步演示
├── run_eval.py            # 完整评估（6 模块 + before/after 对比）
├── run_mcp_server.py      # MCP stdio 独立入口
└── benchmark_tasks.json   # 50 任务基准测试集

examples/
└── deepchoice_integration.py  # 嵌入 DeepChoice 检索器的示例代码
```

## License

MIT
