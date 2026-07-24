# Tool Evolution Engine — Agent 工具调用自适应优化引擎

让 Agent 的工具调用能力从"瞎试"变成"越用越聪明"——采集每次工具调用的轨迹数据，分析失败模式、学习参数分布、挖掘技能组合，最终通过信用评分和灰度路由安全地把优化部署上线。

## 为什么值得关注

Agent 项目通常关注 prompt 怎么写、模型选哪个，很少关注工具调用本身的质量。现实是：

- 同一个工具被调了 1000 次，传错参数的模式是收敛的——但你不知道
- Agent A 和 Agent B 都做"查法规→审合规→出报告"这个三件套——但你发现不了这个规律
- 上线了一个新版本的搜索工具，你怎么判断它是更好了还是更坏了？

Tool Evolution Engine 在 Agent 的工具层和 LLM 之间加了一个"运维+优化"层，用数据驱动的方式回答这些问题。它不依赖特定 Agent 框架——只要你的工具调用能产出结构化轨迹，就能接进来。

## 核心特性

- **全链路轨迹采集**：Pydantic 结构化 Schema（5 种错误类型）+ SQLite FTS5 全文索引，不只是记日志，是可以搜索、聚合、分析的结构化数据
- **失败模式自动分类**：Random Forest 分类器（TF-IDF 字符级 n-gram 特征 + 工具名 + 参数数量 + 时间维度），把失败归入 5 类，不再人肉排查
- **反事实规则蒸馏**：从失败轨迹反推"应该怎么做"，按错误类型自动生成前置校验规则（参数范围校验/授权检查/重试策略/超时阈值/熔断器）
- **参数分布学习**：对历史成功调用的参数做核密度估计（scipy gaussian_kde），自动给出最优默认值（密度峰值）和合法边界（2.5%/97.5% 百分位），替代人工经验配置。支持用户偏好反向注入——MCP 记忆里存的偏好值优先于统计默认值
- **频繁子图技能挖掘**：networkx 构建每任务 DiGraph → Weisfeiler-Lehman 图哈希做同构去重 → 按最小支持度过滤高频工具调用 DAG 链路。比如自动发现"查法规→审合规→出报告"是 87% 任务的公共模式
- **3D 信用评分 + 自动升降级**：成功率(0.4) + 延迟(0.3) + Token 消耗(0.3)，< 20 分自动下线，> 80 分自动晋升，闲置 > 7 天按 0.95^days 衰减
- **一致哈希灰度路由 + A/B 回滚**：canary_5 → canary_15 → canary_50 → active 四段放量，每段要求 min 30 样本才决策。canary < stable - 10pp 自动回滚
- **MCP 协议记忆桥接**：3 个 MCP Tools（search_memory / update_memory / get_user_preferences），stdio transport 供 Claude Code 等客户端直连。Trace 采集成功后自动抽取实体更新记忆

## 快速开始

```bash
# 1. 安装
pip install -e ".[dev]"

# 2. 生成演示数据（200 条轨迹，7 种工具，25% 失败率，3 种预埋 DAG 模式）
python scripts/seed_demo_data.py

# 3. 跑完整演示管道（7 步：分类器→规则→KDE→DAG→治理→总结）
python scripts/run_demo.py

# 4. 跑评估管线（分类器 F1 + KDE 覆盖率 + DAG 召回 + 治理模拟 + 优化前后对比）
python scripts/run_eval.py

# 5.（可选）启动 API 服务
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

**TF-IDF 字符级 n-gram(2,4) 而非词级分词做失败分类**。"timeout" vs "timed out" vs "超时"——这三个在词级别是完全不同的 token，但在字符级 n-gram 共享大量子串。在不依赖语言模型的条件下，字符级 n-gram 是最务实的跨语言文本特征方案。

## 量化指标（200 条演示轨迹 + 50 任务基准）

| 指标 | 值 | 说明 |
|------|-----|------|
| 分类器 Macro F1 | 1.00 | 5 分类，受控数据下每类全部分对 |
| 失败率下降 | -60%（相对） | 20% baseline → 8% optimized，KDE 默认值 + 规则前置校验 |
| DAG 模式召回 | 66.7% | 3 个预埋模式中找出 2 个 |
| 参数模板覆盖率 | 57.1% | 7 种工具中 4 种有足够样本生成模板 |
| 规则准确率 | 100% | 5 条规则全部有效，2 条去重 |
| 吞吐量 | 57 traces/s | 批量写入 |
| 治理动作 | 2 晋升 + 1 回滚 + 0 离线 | 模拟 60 次调用后的自动治理结果 |
| 测试覆盖 | 88 tests | 全部通过 |

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
| POST | `/api/skills/{name}/invoke` | 模拟技能调用（含 canary 路由） |
| POST | `/api/skills/{id}/promote` | 技能发现 → 部署 |
| GET | `/api/rules` | 规则列表（按工具名过滤） |
| GET | `/api/analytics/summary` | 聚合分析摘要 |
| POST | `/api/canary/{id}/promote` | 灰度进阶（5→15→50→active） |
| POST | `/api/canary/{id}/compare` | A/B 变体指标对比 |
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
├── seed_demo_data.py      # 生成 200 条模拟轨迹（7 工具 + 3 DAG 模式）
├── run_demo.py            # 端到端 7 步演示
├── run_eval.py            # 完整评估（6 模块 + before/after 对比）
├── run_mcp_server.py      # MCP stdio 独立入口
└── benchmark_tasks.json   # 50 任务基准测试集

examples/
└── deepchoice_integration.py  # 嵌入 DeepChoice 检索器的示例代码
```

## License

MIT
