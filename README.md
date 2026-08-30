# Tool Evolution Engine — Agent 工具调用自适应优化引擎

> TEE 本身不是 Agent——它不包含 LLM 调用、prompt engineering 或 Agent 编排。它是运行在 Agent 和工具之间的**数据分析管道**：采集 Agent 的工具调用轨迹，用统计/机器学习/图算法分析优化，把结果反馈给 Agent 系统。

让 Agent 的工具调用能力从"瞎试"变成"越用越聪明"——采集每次工具调用的轨迹数据，分析失败模式、学习参数分布、挖掘技能组合，最终通过信用评分和灰度路由安全地把优化部署上线。

## 为什么值得关注

Agent 项目通常关注 prompt 怎么写、模型选哪个，很少关注工具调用本身的质量。现实是：

- 同一个工具被调了 1000 次，传错参数的模式是收敛的——但你不知道
- Agent A 和 Agent B 都做"检索→分析→出报告"这个三件套——但你发现不了这个规律
- 上线了一个新版本的搜索工具，你怎么判断它是更好了还是更坏了？

Tool Evolution Engine 在 Agent 的工具层和 LLM 之间加了一个"运维+优化"层，用数据驱动的方式回答这些问题。它不依赖特定 Agent 框架——只要你的工具调用能产出结构化轨迹，就能接进来。

## 核心特性

- **全链路轨迹采集**：Pydantic 结构化 Schema（5 种错误类型）+ SQLite FTS5 全文索引，不只是记日志，是可以搜索、聚合、分析的结构化数据
- **失败模式自动分类**：Random Forest 分类器（TF-IDF 字符级 n-gram 特征 + 工具名 + 参数数量 + CJK 语言特征），把失败归入 5 类，不再人肉排查
- **反事实规则蒸馏**：从失败轨迹反推"应该怎么做"，按错误类型自动生成前置校验规则（参数范围校验/授权检查/重试策略/超时阈值/熔断器）
- **参数分布学习**：对历史成功调用的参数做核密度估计（scipy gaussian_kde），自动给出默认值和合法边界（2.5%/97.5% 百分位），替代人工经验配置。支持用户偏好反向注入——MCP 记忆里存的偏好值优先于统计默认值。注：KDE 的价值在分布形状与 CI 边界（3000 规模实测 per-param 边界外多数 ≤5%——双分母口径），点估计（mode）在 MAE 口径上不优于中位数（30 工具 0/10 赢）——不做"默认值更准"声明
- **频繁子图技能挖掘**：networkx 构建每任务 DiGraph → Weisfeiler-Lehman 图哈希做同构去重 → 按最小支持度过滤高频工具调用 DAG 链路。比如自动发现"检索→分析→出报告"是高频任务的公共模式
- **3D 信用评分 + 自动升降级**：成功率(0.4) + 延迟(0.3) + Token 消耗(0.3)，< 20 分自动下线，> 80 分自动晋升，闲置 > 7 天按 0.95^days 衰减
- **一致哈希灰度路由 + A/B 回滚**：canary_5 → canary_15 → canary_50 → active 四段放量，每段要求 min 30 样本才决策。canary < stable - 10pp 自动回滚
- **MCP 协议记忆桥接**：5 个 MCP Tools（search_memory / update_memory / get_user_preferences / search_relations / get_repair_hint），stdio transport 供 Claude Code 等客户端直连。Trace 采集成功后自动抽取实体更新记忆
- **实体共现关系建模**（增量一）：从任务树的成功轨迹中跨 trace 池化挖掘实体共现对（0 LLM，确定性两两成对），evidence_trace_ids 全量溯源 + 幂等重建，relation_type 当前仅 co_occur（语义层是规划中的扩展点）
- **用户偏好学习闭环**（增量一）：直方图判定个人参数偏好（样本量 ≥20 + 占比 >60% 严格大于 + 偏离全局 KDE mode），executor: 前缀轨迹隔离，偏好经 MCP 缓存反向注入模板生成（source=user_preference）
- **LLM 修复建议生成**（增量二）：蒸馏时离线批量调用 DeepSeek 为拦截规则生成结构化修复建议 `{suggestion, fix, reason}`（JSON mode + thinking disabled），content_hash 幂等（含工具名，copy-on-hit 零冗余调用）+ fail-open 降级（LLM 不可用时建议降级为模板，规则拦截不受影响）；重放有效性区间 50%~100% 由错误信息质量决定（含有效范围的上界 100%、模糊信息下界 50%）
- **内置执行层**（增量三）：技能包的消费端——`execution/` 包（三适配器 Mock/HTTP/MCP + 确定性匹配器 + 计划装配器 + DAG 拓扑执行器 + 执行审计 + LLM 规划基线）。2016 任务 benchmark 实测：技能包路径成功率 1.0、规划成本 0ms，vs LLM 规划基线成功率 0.985、平均规划 953.9ms——**TEE 节省 token -34.8%、耗时 -31.1%**（差额=重复的 LLM 规划开销；跨口径禁比较——旧 50/2000 任务口径已随规模升级封存）。跨任务熔断三态、取消路径、数据流引用、幂等执行（task_id 即 Idempotency-Key）、executor: 轨迹口径隔离（防自我污染）

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

# 7.（增量三）执行层评测：2000 任务 skill_plan vs llm_plan 对比
#    （llm_plan live 臂需 .env 配 TOOLEVO_DEEPSEEK_API_KEY，无 key 自动 degraded 诚实标注）
python scripts/run_execution_eval.py --num-tasks 2000 --output exec_eval_results.json
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
│  PreferenceLearner    直方图偏好判定 + executor 轨迹隔离（增量一） │
│  RepairAdvisor        LLM 修复建议生成，content_hash 幂等 + fail-open（增量二） │
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
│  MCPBridge            5 MCP Tools + 实体自动抽取            │
│  RelationStore        实体共现挖掘 + 全量证据幂等（增量一）  │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
              FastAPI 端点 + MCP stdio 接口
```

**消费端（增量三，非管道层）**：`execution/` 包消费 deployed_skills——任务描述 → 确定性匹配（工具名命中打分）→ 装配（KDE 默认值+偏好注入+规则前置校验）→ DAG 拓扑执行（并行分支/运行时规则/修复建议重试）→ 轨迹回写管道（executor: 前缀隔离，执行即采集，闭环真实成立）。LLM 规划是它的对照组基线，不是 TEE 组件。

**数据闭环**：Agent 调工具 → Tracer 采集轨迹 → Classifier 分类失败 + KDE 学参数 + DAG Miner 挖模式 → RuleEngine/ParamTemplate/SkillPack 存储知识 → Governor 评分治理 → CanaryRouter 灰度部署 → 新版本上线后继续采集轨迹 → 循环。

## 关键技术决策

**用 Weisfeiler-Lehman 图哈希而非直接图匹配做 DAG 去重**。Agent 每次执行"检索→分析→出报告"时，tool_call 的 node_id 不同，但图结构相同。WL 哈希做的是图同构的 canonical labeling——相同结构的 DAG 映射到相同哈希，不依赖 node_id 或执行时间。这是计算机科学里解决"这两个图本质上是不是同一个"的标准方法。

**用一致性哈希而非随机分流做 Canary 路由**。`MD5(request_hash) % 100` 保证同一用户的请求始终路由到同一个变体。随机分流会导致用户在两次请求间看到不同的工具行为，A/B 对比数据被污染。一致性哈希是基础设施层解决"确定性路由"的经典方案。

**TF-IDF 字符级 n-gram(2,4) + has_cjk 特征检测**。中文错误消息在字符级 n-gram 中与英文无共享子串，单独增加 CJK 字符检测特征位（0/1），给模型显式的语言区分信号。在不依赖语言模型的条件下，字符级 n-gram + CJK 检测是最务实的跨语言文本特征方案。

**离线批处理架构，不做实时拦截**。TEE 设计为离线分析管道——轨迹采集后批量处理，不是 Agent 每次工具调用前的同步拦截器。原因：KDE 参数学习、DAG 子图枚举、信用评分更新都是统计计算，需要足够样本量才有意义（`min_samples=30`）。对实时性要求高的场景（如每次工具调用前必须校验参数），应该在 Agent 代码中直接嵌入 `CounterfactualDistiller` 产出的规则——这些规则是纯确定性逻辑，可以同步执行。修复建议同样离线生成（蒸馏时批量调 LLM），触发时直接查表返回。全管道 run_eval 耗时 294.6s（3000 seed + 2520 benchmark 新口径，2026-08-30 实测）；执行层 2016 任务评测 ~35min（1888 次 LLM 规划，离线批量统计非产品延迟），适合小时级/日级调度，不适合毫秒级在线决策。

**4 层架构 + 0 Agent + 参数上限的复杂度约束**。为什么是 4 层？采集→分析→知识→治理，每层对应数据加工的一个独立阶段，层间通过 Pydantic 模型传递，接口明确。拆成 5 层会把分析和知识割裂（比如 Classifier 的产出直接被 RuleEngine 消费，拆开只增加序列化开销），合并成 3 层会把存储和部署混在一起（知识库和灰度路由的职责完全不同）。为什么 0 Agent？失败分类（RF）、参数学习（KDE）、模式挖掘（WL 哈希）、规则蒸馏（确定性映射）——每一个都是确定性或统计方法能解决的子问题，引入 LLM 只会增加延迟和不确定性。`max_dag_nodes=10` 的根因是子图枚举的计算复杂度 O(2^n)：在 n=10 时最坏情况 ~1024 个子图/任务，在 n=15 时 ~32768——10 是保证枚举在百毫秒级完成的工程上限，同时覆盖了实际 Agent 工作流中 90%+ 的工具调用链路长度。

**全系统只保留 2 个 LLM 点，且各有不可替代的理由**（增量三后）。①修复建议生成——开放式文本生成任务，统计方法生不出"把 max_results 改成 5"这种建议；fail-open（LLM 挂了规则照拦，建议降级模板）。②LLM 规划基线——对照组，存在的意义就是被对比（证明"技能包跳过规划"的收益）；fail-closed（规划失败=任务 failed 诚实记录）。其余 8 个模块全是确定性方法——成本低、可复现、可解释。执行层同步执行（无后台队列）：Mock 毫秒级、LLM 规划受 30s 超时约束，异步化的触发条件（长任务常态/多 worker/取消需求）已显式声明。执行器轨迹用 `exec-` trace_id + `executor:` agent_id 双前缀，DAG 挖掘/KDE 训练/偏好学习三入口过滤——防 mode collapse（用自己产出的参数拟合自己的 mode）。

## 量化指标

> 评测规模（新口径，2026-08-30 终态 E 全维度重跑）：**3000 seed tasks + 2520 benchmark tasks（63 基础 × 40 参数变体，30 异构工具）+ 三档 degradation（750/1500/3000）+ 执行层 2016 任务对比评测**（live 实测），run_eval.py + run_execution_eval.py 实测（种子固化 _MAIN_RNG=42；3000 规模确定性字段逐字段一致，唯一差异=LLM 输出长度自然方差）。**口径切换声明**：旧口径（1000 seed/400 benchmark/50 任务、2000 seed/2000 benchmark/7 工具）已封存——新旧口径不可比，禁止混用。指标来源：template-data/metrics-snapshot.md（新口径段，2026-08-30）。

| 指标 | 值 | 说明 |
|------|-----|------|
| 分类器 Macro F1 | 1.000（种子同分布）/ **0.609（噪声 holdout）** | 5 分类，1768 训练 / 759 测试（同分布 1.000）；**跨分布 noisy-holdout 0.609**（clean 2650→noisy 108，加噪合成数据——typo/中英混用/同义词；只吃 eval-* 种子，旧 0.982 已封存） |
| 失败率下降 | **-61.5%**（率口径）/ **-65.8%**（次数口径） | 失败率 0.1693→0.0652；1508→516 failures，2520 benchmark（63 基础 × 40 变体，30 工具），注入全部 5 种错误类型 |
| Token 消耗下降 | **49.0%** | 1,547,550→789,975 tokens，KDE 默认值省略参数 + 规则前置校验 |
| 重试次数下降 | **-65.8%** | 1508→516 retries，规则前置校验在调用前拦截参数错误 |
| 失败类型拆分 | 全部 5 种有对比 | param_error **67.0%**、permission **68.5%**、quota **62.3%**、service_unavail **63.7%**、timeout **67.4%** |
| DAG 模式召回 | 100.0%（21/21） | 21 种预埋模式全找回 + 33 个高频子模式；**三档 degradation（750/1500/3000 seed）各档均 100%** |
| 参数模板覆盖率 | 100%（30/30） | 30 种异构工具 74 个参数全生成 KDE 模板；CI 边界外多数 ≤5%（个别超 5%——双分母口径）；**mode vs median 点估计 KDE 0/10 赢——不优于中位数，KDE 价值在分布形状与 CI（诚实披露）** |
| 规则准确率 | 0%（如实披露） | gsm 主口径采集点在 gsm 构建前（评测中段不产 rules）；stage 11 已接入规则蒸馏与修复建议评测（rep- 4 规则 → 4 建议，增量二） |
| 灰度上线 | **3/3 走通（310 calls）**（5 技能中 4 晋升 active） | 三阶放量全路径实测 + 权重 40/30/30 最优（5P vs 3P）；S 终审修复后生产 canary 闭环（execute_skill 一致性哈希路由变体 + 双变体实测落 canary_invocations） |
| 权重敏感性 | 40/30/30 最优（5P vs 3P vs 3P） | 三组权重对比验证，默认权重晋升最多 |
| 简化场景 RF vs 规则 | RF **93.9%** vs 规则 **24.2%**（F1 0.941 vs 0.177） | 含拼写错误+中英混用+同义词，RF 显著优于规则；抽样依赖 _MAIN_RNG 状态，如实披露 |
| 跨语言分类 | EN **100.0%**（11/11）/ CN **33.3%**（1/3） | has_cjk 特征改善中文分类（旧 66.7% 已封存）；char_wb 跨语言仍有限 |
| 测试覆盖 | **309 tests** | 增量零 10 条生产就绪修复 + 三增量 + 去领域化 + 评测规模化 + S 终审修复（canary 闭环/异常兜底/白名单/值校验等）全部测试锁死 |
| 执行层对比（增量三） | skill_plan 成功率 **1.0**（1888/1888）、规划成本 **0ms**；llm_plan **0.985**（1860/1888）、平均规划 **953.9ms**；**TEE 节省 token -34.8%、耗时 -31.1%**；修复闭环 **5/5** | 2016 任务 benchmark（live；匹配率 0.937；llm_plan 28 失败为 LLM 规划缺参被 fail-closed 拒绝——单任务隔离机制实测生效）；skill_plan 可复现、llm_plan 结构稳定（LLM 输出有方差——诚实口径）；**跨口径禁比较**（任务构成/并行度/LLM 时点方差） |
| 评测复现性 | 3000 规模：确定性字段逐字段一致；LLM 字段结构稳定 | 种子固化（增量零建立、三增量延续；LLM 字段有运行间方差——诚实口径） |
| 记忆联动（增量一） | 关系召回 30/30、幂等重建 True；偏好闭环 learned/injected/source 全 True、重试再降 78.0%（89→11） | 共现挖掘 + 直方图偏好判定，0 LLM；阈值敏感性实测 20/60 为平衡点（15/50 放过 60% 边际噪声、30/70 漏掉 25 样本强偏好）；无规模依赖，沿用增量一数据 |
| 修复建议（增量二） | 结构化成功率 1.0、参数覆盖 1.0、重放上界 100%（90/90）、下界 50%（15/30）、幂等复用 4/4 | DeepSeek v4-flash + thinking disabled（每规则输出 ~78 token）；上界=错误信息含有效范围、下界=模糊错误信息——**有效性区间由错误信息质量决定** |

## 技术栈

Python 3.13 · SQLite + FTS5 · Pydantic v2 · FastAPI · scikit-learn（RandomForest + TF-IDF + KMeans） · scipy（gaussian_kde） · networkx（DiGraph + WL hashing + 拓扑排序） · FastMCP · aiosqlite · httpx（AsyncClient 连接池） · DeepSeek API（v4-flash：修复建议生成 + LLM 规划基线） · pytest-asyncio

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
| GET | `/api/rules/{id}/hint` | 规则的修复建议查询（增量二） |
| GET | `/api/analytics/summary` | 聚合分析摘要 |
| POST | `/api/canary/{id}/promote` | 灰度进阶（5→15→50→active） |
| POST | `/api/memory/search` | MCP 记忆搜索 |
| POST | `/api/memory/update` | MCP 记忆更新 |
| GET | `/api/memory/preferences` | 用户偏好查询 |
| GET | `/api/memory/relations` | 实体共现关系查询（增量一） |
| POST | `/api/templates/generate` | 参数模板生成 + 用户偏好注入（增量一） |
| POST | `/api/execute/task` | 执行层任务执行（增量三）：三模式 auto/skill_plan/llm_plan，幂等（task_id 即 Idempotency-Key：409+轮询 hint/重复提交返回已存），adapter 三选 mock/http/mcp，客户端断开→cancelled |
| GET | `/api/execute/task/{task_id}` | 执行审计查询（任务状态 + steps + 修复证据）（增量三） |

## 项目结构

```
src/tool_evolution/
├── collection/       # TraceReport + TraceStore + Tracer（采集层）
├── analysis/         # Classifier + Distiller + RepairAdvisor + KDEAnalyzer + DAGMiner + PreferenceLearner（分析层）
├── knowledge/        # RuleEngine + ParamTemplateManager + SkillPackManager（知识层）
├── governance/       # SkillGovernor + CanaryRouter + MCPBridge + RelationStore（治理层）
├── execution/        # 消费端（增量三）：tool_specs + adapters + matcher + assembler + executor + audit + planner
├── server/           # FastAPI app + 8 路由模块
└── utils/            # Config（pydantic-settings，.env 支持）+ Database（DDL + 迁移 v6）

scripts/
├── seed_demo_data.py      # 生成 200 条模拟轨迹（7 工具 + 8 DAG 预埋模式）
├── run_demo.py            # 端到端 7 步演示
├── run_eval.py            # 完整评估（11 阶段：分类器/KDE/DAG/治理/before-after/退化/简化/关系/偏好/修复建议）
├── run_execution_eval.py  # 执行层评测（增量三）：skill_plan vs llm_plan + 修复闭环 + 口径隔离（--num-tasks 默认 2000，--output 落 JSON）
├── run_mcp_server.py      # MCP stdio 独立入口
└── benchmark_tasks.json   # 63 基础任务（--num-variants 扩展为 2520 确定性变体，30 工具）

examples/
└── deepchoice_integration.py  # 嵌入 DeepChoice 检索器的示例代码
```

## License

MIT
