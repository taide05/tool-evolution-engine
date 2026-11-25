# ToolGuard — tool-call reliability middleware for LLM agents

ToolGuard 是一个夹在 Agent 与工具之间的中间件：**采集每次工具调用 → 分类失败 → 从真实失败轨迹中学习修复策略 → 运行时修复并记录指标**。它不训练模型、不改工具实现，只做工具层的确定性修复，因此可解释、低成本、易部署。

> **外部审阅请从 [REVIEW.md](REVIEW.md) 开始**：它给出不依赖作者自述的核验路径、每个 claim → artifact → 复核命令、已知局限与审阅 checklist。

当前版本有三层证据：

1. **离线真实数据评测**：在 **30,120 次记录调用（7,636 条解析 trace 展开；764 个唯一 query、2,303 个唯一 (query, tool) 组合）/ 1,033 次真实失败**（StableToolBench 公开基线轨迹，Apache-2.0）上验证修复策略，并重点做了**跨任务泛化**；修复提议现在按“提议修复 / 被 schema 拒绝 / 通过”的漏斗统计（完整结构校验：类型、必填/多余参数、enum、format、数值范围）；
2. **live 端到端 A/B**：真实 DeepSeek agent + 真实免 key 公开 API，从 27 个短任务到 **31 个多步依赖任务（V3 漂移加固任务池）**；V2 把交互预算做成**三档扫描 + 重复**，给出成功-预算曲线、Wilson CI 与配对检验；第四轮再把同一协议**复制到 3 个模型**（qwen3.8-flash / DeepSeek-V4.1-Flash / qwen4-max）；比较类任务的答案校验改为**关系断言**（见下文）；
3. **真实 MCP 集成**：同一套中间件不改一行，接到**两个真实 stdio MCP server** 上，覆盖多 server 路由、server 进程真实崩溃重连、catalog drift 未知工具重解析。

## 结果（离线确定性评测）

> **2026-11-24 证据链修复轮（未重跑 live）**：修复提议现在按**两层证据**统计，`schema_valid_rate` 不再是构造性的 100%：
> **第一层（schema-valid）**：通过完整结构校验（类型、必填/多余参数、enum、format、数值范围；声明类型 coercion 先做一次、单独计数）。
> **第二层（value-verified）**：schema 不通过的提议，只有当它与**同一 task/function 的逐字真实成功调用**完全一致时才计入（经验证据，单独报告，绝不并入 schema 合法率）。
> `accepted = schema-valid + value-verified`；`schema_valid_rate = schema-valid / 提议数`。

| 协议 | 可评估失败 | 提议修复 | schema 通过（含 coercion） | schema 拒绝失败 | value-verified | **accepted 覆盖** | value agreement（可验证中） | 缓存验证 |
|---|---|---|---|---|---|---|---|---|
| **LOMO（固定三模型训练 universe 内留一）** | 969 | 583 | **472（81.0%；150 条 coercion）** | 111 | 59 | **531/969 = 54.8%** | 127/276 = 46.0% | 48 恢复 / 0 反例 / 483 miss（48/531=9.0%） |
| cross-model（3→2 模型） | 277 | 173 | **123（71.1%；30 条 coercion）** | 50 | 42 | **165/277 = 59.6%** | 78/147 = 53.1% | 15 恢复 / 150 miss |
| **cross-group（跨任务集，未见任务）** | 76 | 57 | **30（52.6%；30 条 coercion）** | 27 | **27** | **57/76 = 75.0%** | **27/38 = 71.1%** | 全部 cache miss |

**如何读这张表（关键）**：
- 旧表里的 54.7% / 58.1% / 75.0% 只检查“参数名+必填”，是构造性通过率；现在的覆盖数字与旧口径**同量级甚至略高**，但每一分都拆成“schema 合法”或“逐字真实成功调用”两层证据，且 schema 合法率明显低于 100%（81.0% / 71.1% / 52.6%）。**不是靠放宽校验保数字**。
- 覆盖率恢复主要来自两个**真实修复**：① 提议修复现在会先把值转换为 schema 声明的类型（`"1970"`→`1970` 等；LOMO 150 条、cross-model 30 条、cross-group 30 条通过 coercion 后合法）；② 上游 `api_list` 自身有保真度缺陷（把 watchmode 的 `start_date/end_date` 声明为 NUMBER，而**记录中的真实成功调用用的是 ISO 日期字符串**；部分函数参数列表为空），这些提议过不了声明 schema，但若与同 task/function 的真实成功调用逐字一致，则作为 value-verified 计入（cross-group 的 27 条文本抽取修复即此类，**27/27 与真实成功调用一致**）。
- **cross-group 75% 的构成**：30 条来自 schema 默认值补全（coercion 后 schema-valid，未经验证）+ 27 条来自学习式文本抽取（value-verified，27/27 值一致）。旧口径把 57 条全部当“schema 合法”，是不对的；现在的分解更弱但更诚实。
- **LOMO 协议诚实性（P1-6）**：主数字**不是**真正的五模型 leave-one-model-out。训练 universe 固定为 chatgpt / toolllama / gpt-3.5-turbo-1106：留出这三者之一时只用另外两个训练；留出 gpt-4-0613 / gpt-4-turbo-preview 时用全部三个训练——**两个 gpt-4 家族从不进入训练**。`metrics.json` 的 `protocol.lomo` 已如实写明。真 LOMO 敏感性（每个 fold 用其余四个家族训练）：**542/969 = 55.9%**（`results.lomo_sensitivity_exclude_fold`）。

**规则类别（两层证据口径，cross-model / cross-group）**：

| 规则 | cross-model | cross-group | 说明 |
|---|---|---|---|
| 任务文本抽取（学习得到） | 12 accepted（全部 value-verified，12/12 值一致） | 27 accepted（全部 value-verified，27/27 值一致） | 日期/年份/关键词抽取；过不了声明 schema 时用“与真实成功调用逐字一致”验证 |
| 经验记忆重放 | 87 accepted（57 schema-valid + 30 value-verified；66/84 值一致） | — | 同任务、其他模型跑成功过 |
| 学习式参数众数 | 4 accepted（schema-valid） | — | 同函数的成功参数分布 |
| schema 默认值补全 | 52 accepted（schema-valid，未验证提议） | 30 accepted（schema-valid，未验证提议） | 错误未给参数名时的启发式 |
| 必填参数默认填充 | 10 accepted（schema-valid） | — | 从 schema example/default 补必填 |

**其他离线口径**：
- `dataset.real_failures=1,033` 与主表 `evaluable_failures=969` 相差 **64 = 30 条 raw arguments + 34 条 tool not in schemas**（这些失败在 `dataset.real_failures` 内、但不进入修复分母）；`metrics.json` 的 `results.skipped_no_schema*` 逐项解释（P1-5）。
- 数据解析：**7,636 条 trace、30,120 次记录调用（764 个唯一 query、2,303 个唯一 (query, tool)、1,522 个工具）**；解析器对 **20 条** observation 无法解析、按成功处理（0.07%，`dataset.unparseable_observations_treated_as_success`）。
- 缓存口径：`cache_verified_recovered=48 / cache_miss=483`，分母是 531 条 accepted 修复（9.0%）；正文里的 51.6% 是**参考解输入的缓存覆盖率**，两个分母不要混用（P1-4）。
- 产物由 `scripts/build_metrics_artifact.py` 从被解析 records + `eval_toolguard_rules.py` 输出**确定性重建**，不再手工维护（P1-3）；复现链见下文“产物与复现溯源”。

## Live 端到端 demo（真实公开 API + 真实 DeepSeek agent）

四轮递进验证：前三轮使用真实 DeepSeek `deepseek-v4-flash` agent，第四轮换成 3 个模型（qwen3.8-flash / DeepSeek-V4.1-Flash / qwen4-max）；全程真实免 key 公开 API、程序化答案校验（**无 LLM judge**）。比较类答案的判定已从“候选名出现”升级为**关系断言**（winner 必须与数值排序一致、否定词处理、million 等单位归一；见下）。合成元素只有**确定性的故障/崩溃/配置漂移注入**，全部在 artifact 中逐项披露；DeepSeek 或公开 API 不可用时 demo 停止报错，**不会回退离线/mock**。

### 第一轮：27 个短任务 / 6 个 API（可靠性与成本口径）

6 个真实公开 API（World Bank、PyPI、Open-Meteo、Hacker News、GitHub、汇率），27 个可自动校验任务，middleware on/off 配对 A/B。故障注入 = 每个逻辑调用的首次尝试可能被注入 timeout/rate-limit，模拟离线分析里发现的瞬态失败类。

| 条件 | 任务成功 | 工具调用失败 | 工具调用数 | 轮数 | tokens |
|---|---|---|---|---|---|
| normal（6 轮预算）baseline | 27/27 | 14 | 44 | 68 | 57,042 |
| normal guard | **27/27** | **1**（-93%） | 31（-30%） | 55（-19%） | **44,217（-22.5%）** |
| stress（4 轮预算 + 更高故障率）baseline | 25/27 | 20 | 48 | 71 | 59,409 |
| stress guard | **26/27**（McNemar p=1.0） | **4**（-80%） | 33（-31%） | 57（-20%） | **46,295（-22.1%）** |

**诚实结论**：27 个任务对 baseline 太容易（normal 双方 27/27），所以这一轮**任务成功率没有显著提升**；价值体现在工具失败 **-80%~-93%**、tokens **-22%**、轮数 **-19%~-20%**。这是“可靠性与成本”结果，不是任务成功率结果。artifact：[`benchmarks/toolguard/live_demo.json`](benchmarks/toolguard/live_demo.json)。

### 第二轮：多步依赖任务 × 预算扫描 × 重复（任务成功率口径）

#### 为什么从 V1 重做（自我批评）

V1（20 任务 / 单预算 / 单次运行）的问题：①任务量和重复不足，只有点估计没有方差；②单预算恰好卡在“无故障需求 +1 轮”，容易被质疑“把预算调到让 baseline 死”；③单模型、指标面窄。V2 把**预算公开成实验变量**并加重复：

- 任务池 20 → **32**（V2）→ **31**（V3，任务集修订见本节末）；类别覆盖 weather / worldbank / pypi / hn / fx；
- **预算扫描**：每任务测 `clean+1` / `clean+4` / `10` 三档（clean = 无故障 calibration 轮数），得到**成功-预算曲线**；
- **重复**：紧预算 `clean+1` 用 3 个故障 seed（93 格/臂），其余各 1 个 seed（31 格/臂）；报告 Wilson 95% CI 与配对 McNemar（按 task×repeat 配对）；
- 故障注入同 V1（每个逻辑调用前 2 次尝试失败，确定性），单模型 `deepseek-v4-flash`（决策：不加模型轴，预算用于重复）；
- V3 共 **310 格**（3 预算档 × 2 臂 × 31 任务 + 重复），墙钟 1,367s，2 个 worker。

#### 结果（V3 当前主口径）

| 预算条件 | 平均预算 | baseline 成功 | guard 成功 | 配对 McNemar | turns/格 (b→g) | tokens (b→g) | agent 可见工具错误 (b→g) |
|---|---|---|---|---|---|---|---|
| **clean+1** | 5.06 | **51/93 (54.8%, CI 44.7–64.6%)** | **93/93 (100%, CI 96.0–100%)** | **p=4.55e-13** | 4.52→3.07 | 660,703→419,964（**-36.4%**） | **263→5** |
| clean+4 | 8.06 | 28/31 (90.3%) | 31/31 (100%) | p=0.25 (n.s.) | 5.45→3.07 | 282,878→140,551（-50.3%） | 100→0 |
| 10 | 10.0 | 30/31 (96.8%) | 30/31 (96.8%) | p=1.0 (n.s.) | 5.45→2.94 | 281,229→133,991（-52.4%） | 91→0 |

**成功-预算曲线**：baseline 54.8% → 90.3% → 96.8%（预算增加才恢复），guard **100% → 100% → 96.8%**，且 turns/tokens 基本不随预算变化。这直接回答“是不是把预算调死”的质疑：预算越低 baseline 越差、guard 不变；预算充足时两者成功率持平，guard 的成本优势更大（tokens -36%~-52%）。

其他发现：
- clean+1 的 42 个 baseline 失败**全部是 `no_answer`（预算耗尽）**；guard 在 clean+1/clean+4 全部成功，10 轮有 1 个失败（模型把人口四舍五入成 “5.61 million”，未命中精确校验——模型回答格式问题，不是中间件失败）；
- 即使 10 轮预算，baseline 仍有 HN 任务失败（`story_id` schema/后端类型不匹配让 baseline 反复重试直到耗尽，而 guard 在同一轮内完成类型修复）；
- 全 V3 运行共注入 **782 个瞬态故障 epoch**（timeout / rate_limit / server_error）：baseline 臂 **382 个**（该臂无中间件，这些 epoch 直接暴露给 agent，是其 263 个 agent 可见工具错误的来源）、guard 臂 **400 个**（中间件吸收后 agent 可见工具错误仅 5 个）。旧文案“382 全部在中间件内吸收”不成立；本项数字全部可由 artifact records 的 `faults_injected`/`fault_kinds` 重算；
- guard 的重试退避总开销 **18.15s / 93 格 ≈ 195ms/格**（可忽略），换来 120 次轮内恢复；
- V2 还修掉了 V1 从轨迹里发现的缺陷：修复后的调用若再遇瞬态故障，现在会在同一个 call 内继续重试（`error_hint_coercion+transient_retry`，commit `13af709`）。

artifact：[`benchmarks/toolguard/live_demo_hard_v3.json`](benchmarks/toolguard/live_demo_hard_v3.json)（含逐格记录、calibration、逐条修复事件、Wilson CI、预算曲线、失败模式与局限）。V2 的 32 任务 artifact（`live_demo_hard_v2.json`，commit `43c4e89`）保留为演进记录。

> **V3 任务集/验证器修订**：①HN `karma` 是**直播可变 ground truth**（同一作者 karma 在一小时内 112→161），V3 用同样两工具链把 3 个 karma 任务换成**不可变**的“账号创建年份”任务，任务池 32 → **31**；②验证器的数字解析修复了 Unicode 负号（模型常写 “−3.4” 而不是 “-3.4”，旧验证器会解析成正数并误判正确答案为失败）；③多模型评测在每个模型块前重建 ground truth，把漂移窗口压到单模型块。V2 的 32 任务数字由 commit `43c4e89` 的代码产生，保留为历史。

#### HN schema drift：一致 schema 对照实验（2026-11-24 新增，回应 P1-1）

背景：`get_hn_item` 的 agent-visible schema 故意声明 `story_id: string`，后端要求 integer；baseline 踩中后只能反复重试直到预算耗尽，guard 用 backend error-hint coercion 在同一轮内修复。为量化这部分“工具契约缺陷”对 headline 的贡献，本轮做了**一致 schema 对照**：只把 `get_hn_item.story_id` 改成 integer（其余任务、seed、预算、重复完全相同），跑 clean+1 × 31 任务 × 3 repeats × 2 臂 = **186 格**。

| 运行 | baseline clean+1 | 其中 HN 子集 | 非 HN 子集 | guard clean+1 |
|---|---|---|---|---|
| V3 主口径（schema 故意不一致） | 51/93（54.8%） | **2/12** | 49/81（60.5%） | 93/93（HN 12/12，非 HN 81/81） |
| **一致 schema 对照（新增）** | **61/93（65.6%）** | **12/12** | 49/81（60.5%，与旧 run 逐格相同） | 93/93（HN 12/12，非 HN 81/81） |

- **归因**：HN 契约缺陷贡献了 baseline-vs-guard 差距的 **10/42 = 23.8%**；剩下 32 格差距来自真实的瞬态故障修复（重试/coercion/name resolution 等）。非 HN 子集两次运行**逐格相同**，说明对照干净、差异确实来自 schema 修复；
- guard 在两种 schema 下都是 100%，说明 guard 的 coercion 修复与“一致 schema”效果相同（对它不是作弊，而是把工具契约错误修好了）；
- 成本：**$0.18 / 658s / 186 格**（DeepSeek 价格）。artifact：[`benchmarks/toolguard/live_demo_hard_v3_hnconsistent.json`](benchmarks/toolguard/live_demo_hard_v3_hnconsistent.json)；复现：`--budgets clean+1 --repeats clean+1:3 --hn-schema consistent --workers 2`；
- 注意：这只覆盖单模型；多模型复制仍用原 schema（其 HN 失败占比 20.6%–25.0%，见第四轮），扩展到多模型留作下一步。

### 第三轮：真实 MCP 双 server + transport 崩溃 + catalog drift（产品化口径）

V2 的 MCP 验证从“同一中间件在 MCP 上也能重试”升级为**部署/传输层容错**：

- **双 MCP stdio server**：`--group worldbank` 与 `--group market` 两个独立进程，客户端 adapter 合并工具表并按工具路由（真实多 server 部署形态）；
- **真实 transport 故障**：worldbank server 进程在第 2 次调用时 `os._exit(17)` 硬崩溃，stdio session 真实断链；adapter 在下一个调用时重连新进程，中间件把断链当作可重试的 `request_error` 在同一 agent 轮内重试；
- **catalog drift / 未知工具**：客户端广告旧名字 `fetch_exchange_rate`，server 实际注册的是 `get_exchange_rate`；调用得到 MCP unknown-tool 错误后，中间件用离线验证过的**函数名相似度规则（≥0.55）**在真实工具表上解析并重路由；
- server 侧瞬时故障 epoch 照旧；16 个多步任务、预算扫描 `clean+1` / `clean+4`、`clean+1` 两 repeat（96 格），每格**全新两个 server 进程**；模型 `deepseek-v4-flash`。

**V2（历史协议，budget 用 guard 校准）**：

| 预算条件 | baseline 成功 | guard 成功 | 配对 McNemar | turns/格 (b→g) | tokens (b→g) | agent 可见工具错误 (b→g) |
|---|---|---|---|---|---|---|
| **clean+1** | **11/32 (34.4%, CI 20.4–51.7%)** | **32/32 (100%, CI 89.3–100%)** | **p=9.54e-07** | 4.56→3.25 | 205,600→134,998 | **115→4** |
| clean+4 | 14/16 (87.5%) | 16/16 (100%) | p=0.5 (n.s.) | 5.81→3.13 | 138,002→63,679 | 63→1 |

**V3（2026-11-24 修正，P1-2：baseline/guard 分开校准、预算锚定 baseline fault-free 轮数）**：

- 校准两臂各测一次：guard **16/16** 无故障成功；baseline **14/16**——两个 FX 任务在无中间件时连 fault-free 都过不了（catalog drift 别名只有中间件能解析），这两个任务的预算按 baseline 的 7/3 轮锚定；其余任务两臂 turns 基本相同；
- clean+1 预算略大于 V2（按 baseline 轮数锚定，fx 任务预算 3→8），因此 baseline 从 11/32 变为 **13/32（40.6%）**——这是**更公平**的预算，不是守卫变弱；guard 仍 **32/32（100%）**，McNemar **p=3.81e-06**；
- clean+4：baseline 14/16（87.5%）vs guard 16/16（100%，p=0.5）；tokens 224,875→132,376（clean+1，**-41%**）、138,565→63,467（clean+4，-54%）；agent 可见工具错误 clean+1 **131→4**；
- 真实崩溃/重连 55 次（两臂合计；每格 0–2 次），全部被中间件在同一 agent 轮内吸收；guard 修复规则：`transient_retry` 77、`function_name_resolution` 9（+6 叠加）——与 V2 相同，说明 adapter 不改中间件逻辑；
- 成本 **$0.098 / 667s / 96 格**（含两臂校准 32 次 fault-free 运行）；
- 结论：修正预算不对称后，guard 相对 baseline 的优势仍然成立（clean+1 **+59.4pp**，V2 口径 +65.6pp；差距主要是 baseline 在 fx 任务上获得了更多预算），且 guard 在两种锚点下都是 100%。

- 传输无关性验证：`ToolGuard` 本体一行未改，新增的是 `MCPBridge`（含重连）/`MCPMultiBackend`（多 server 路由 + 未注册别名）adapter；
- 测试：MCP SDK 内存 transport 集成测试覆盖 schema 转换、`isError`→taxonomy、参数修复、transient 重试、双 server 路由、catalog drift 解析；toolguard 测试 **115 passed**（全仓 403 passed，另有 1 failed + 20 errors 为与本仓库无关的 Windows 临时目录 ACL 环境错误）。

artifact：V2 [`benchmarks/toolguard/mcp_demo_v2.json`](benchmarks/toolguard/mcp_demo_v2.json)（历史协议），V3 [`benchmarks/toolguard/mcp_demo_v3.json`](benchmarks/toolguard/mcp_demo_v3.json)（当前主口径；`experiment=ToolGuard_mcp_multiserver_live_ab_v3_baseline_anchor`、`protocol.budget_anchor=baseline`；本轮用 `--from-records` 从 checkpoint 重建生成元数据，records/aggregate 未变）。

> V1 的窄口径 artifact 保留作为演进记录：`live_demo_hard.json`（20 任务单预算，commit `02ac71a`）、`mcp_demo.json`（10 任务单 server，commit `b1d7609`）；对外口径以 V3 为准。

### 第四轮：多模型复制（3 个模型 × 同一协议，回应“单模型”质疑）

单模型结果最大的风险是“只在 DeepSeek 上成立”。第四轮在**完全相同的任务池、故障 seed、预算语义**下跑 3 个模型（`scripts/run_multimodel_live.py`）：每个模型先做 31 次无故障 calibration 定位能力档，再跑 186 个评测格（31 任务 × 2 臂 × clean+1 两 repeat + clean+4）。V3 版本还做了两处 ground-truth 加固：把直播可变的 HN karma 任务换成**不可变**的“账号创建年份”任务，并在**每个模型块开始前重建 ground truth**（漂移窗口从整轮 ~1h 压到单模型块 ~20min）。

| 档位（按价格） | 模型 | provider | 官方价格 $/M（入/出） | 无故障 placement |
|---|---|---|---|---|
| cheap | qwen3.8-flash | 阿里百炼 | 0.15 / 0.47 | 27/31（87.1%），4.00 轮 |
| mid | DeepSeek-V4.1-Flash-1017 | DeepSeek | 0.15 / 0.60 | 31/31（100%），3.32 轮 |
| strong | qwen4-max | 阿里百炼 | 1.413 / 4.24 | 29/31（93.5%），3.29 轮 |

| 模型 | clean+1 baseline | clean+1 guard | McNemar | tokens (b→g) | clean+4 baseline | clean+4 guard |
|---|---|---|---|---|---|---|
| qwen3.8-flash | **28/62 (45.2%)** | **57/62 (91.9%, CI 82.5–96.5%)** | **p=1.3e-07** | 482,951→313,253（-35%） | 25/31 (80.6%) | 31/31 (100%) |
| DeepSeek-V4.1-Flash | **32/62 (51.6%)** | **62/62 (100%)** | **p=1.9e-09** | 427,751→286,278（-33%） | 30/31 (96.8%) | 31/31 (100%) |
| qwen4-max | **34/62 (54.8%)** | **62/62 (100%)** | **p=7.5e-09** | 412,151→272,604（-34%） | 28/31 (90.3%) | 31/31 (100%) |

- `guard >= baseline` 在**所有模型 × 所有条件**成立（artifact `consistency` 字段）；
- **“能力越弱，中间件收益越大”不成立**：qwen3.8-flash 紧预算提升 **+46.7pp**（45.2%→91.9%），mid/strong 为 **+48.4pp / +45.2pp**，最大的是 mid 而不是最弱模型；去掉 HN 任务后三个模型的提升变为 **+40.7 / +42.6 / +38.9pp**（cheap / mid / strong），mid 仍然最大——排序不变，但幅度下降，说明 HN 契约缺陷放大了 guard 收益；
- **HN schema drift 占比披露（P1-1）**：两臂共享同一个故意不一致的 `get_hn_item` schema（schema 声明 string，后端要求 int；baseline 踩中后失败，guard 用 error-hint coercion 修复，这不是单向作弊）。但它在 headline 中占比不小：单模型 clean+1 baseline 的 42 个失败里 **10 个（23.8%）** 是 HN 相关任务；多模型 baseline 失败中 HN 占比 **deepseek 23.3% / qwen-flash 20.6% / qwen4-max 25.0%**。因此“placement 27/31、29/31”是“模型 + 是否踩到被故意做坏的 schema”的混合结果，**不是纯模型能力**；去掉 HN 任务后的主数字见上一条。把 HN 单独成臂 + 一致 schema 对照需要重跑 live（预算与成本见本轮修复报告，待批准），本轮没有重跑；
- **贵模型没有买到“无故障鲁棒性”**：qwen4-max 单价比 qwen3.8-flash 高约 9 倍，但 fault-free placement（29/31、3.29 轮）没有超过 DeepSeek-Flash（31/31、3.32 轮）。三个模型都会踩同一类失败（HN `story_id` 契约不匹配），说明这类问题不是模型能力问题而是**工具契约问题**——正好是中间件该管的；
- **V3 后 guard 残余失败已不再是数据漂移**：qwen3.8-flash 的 5 个 clean+1 guard 失败全部可归因于**弱模型自身行为**（如声称 `get_weather` 不可用、GDP 数值编造 $106,588、答非所问回答 France），另两个模型 guard 失败为 0。V2 里那些 HN karma 漂移造成的假阴性已消失；
- 成本：评测 $2.08（qwen4-max 占 80%）+ calibration $0.30 ≈ **$2.38**（数据集比 V2 少一个任务，故略低）。

artifact：[`benchmarks/toolguard/live_demo_multimodel_v3.json`](benchmarks/toolguard/live_demo_multimodel_v3.json)（558 格逐条记录、三模型 placement、Wilson CI、成本核算、逐条修复事件、`tasks_per_model` 与每条记录的 `expected` 快照）。V2 版 artifact（`benchmarks/toolguard/live_demo_multimodel.json`，含 karma 任务）保留为演进记录，复现见 commit `fb13a1f`。该 artifact 产生于验证器 Unicode 负号修复之前；离线核对显示 12 条含 `−`（U+2212）的答案当时均已判成功，因此成功率不受该修复影响。

**这一轮的局限**：3 个模型是复制而非模型间假设检验；cheap/strong 同属 Qwen 家族；每个模型只做 1 次 calibration；故障仍是确定性注入；两个较强模型的紧预算 guard 为 100%，存在 ceiling effect（进一步提升空间只能靠更难的题）。

## 架构：一个中间件，三种传输层

中间件只依赖 `ToolBackend` 协议（`schemas()` 返回工具清单，`call(name, arguments)` 返回统一 `CallResult`）：

| 后端 | 用途 | 额外能力 |
|---|---|---|
| `RecordedBackend` | 离线真实轨迹 + 发布方响应缓存复放 | — |
| `LiveBackend` / `LiveHardBackend` | 真实公开 HTTP API（6 / 12 工具） | — |
| `MCPBackend` + `MCPBridge` | 单个真实 MCP stdio server（12 工具） | transport 断链检测 + 自动重连 |
| `MCPMultiBackend` + `MCPBridge` | 多个 MCP server 合并工具表 | 多 server 路由、隐藏工具、catalog drift 别名 |

因此“接入新工具层”是 adapter 级别的工作，不是重写中间件；MCP demo 里 `ToolGuard` 本体零改动。MCP V2 覆盖的三类真实故障（server 进程硬崩溃/断链重连、server 侧 isError epoch、未知工具名漂移）最终都归一到中间件的同一套 `CallResult` 错误 taxonomy 上。

## 工作原理（五级修复，便宜且通用的优先）

1. **确定性 schema 规则**：函数名纠错、删除/重命名 schema 外参数、schema 默认值补必填、类型转换；
2. **学习式任务文本抽取**（跨任务泛化核心）：在训练集上按参数名学习“哪个抽取器能从任务文本中取到该参数的真实值”（ISO 日期 / 年月 / 年份 / 引号关键词 / 数字 / 国家码），部署时对未见任务直接抽取缺失参数；
3. **经验记忆**：同一 task/function 曾在有效轨迹中成功 → 重放那次成功调用；
4. **学习式参数众数**：同函数成功调用的参数取值分布；
5. **schema 默认值补全**：错误未给参数名时，用 schema 声明的 example/default 补全（**未验证提议**）。

修复提议先登记、再做**完整结构校验**（`src/tool_evolution/toolguard/schema_check.py`）：函数存在、参数在 schema 内、必填齐全，且按 schema 实际声明的约束检查**类型 / enum / format / 数值范围 / 字符串长度与 pattern / 数组 items**（schema 没声明的约束不强行编造）；未通过的提议计入“被 schema 拒绝”而非静默丢弃，live guard 在把修复后的调用发给 backend 之前做同一套校验并记录 `schema_errors`（拒绝的修复返回原错误，不静默放过）。另外做两类外部验证：
- **value agreement**：修复后的参数值是否与同一任务/函数某次真实成功调用完全一致；
- **cache verification**：修复后的调用是否在真实响应缓存中命中且返回成功。

## 数据来源

- 轨迹：StableToolBench baseline trajectory release（Apache-2.0，公开）；**7,636 条 trace、30,277 条原始记录调用（剔除 157 条 DFS sentinel 后 30,120 条评估）**
- 工具 schema：同一 release 的 solvable_queries `api_list`（2,461 个函数）
- 真实响应缓存：ToolBench Apr-2024 缓存（仅用于修复结果的补充验证；覆盖参考解输入，最终尝试命中率 51.6%）
- 数据不随仓库提交；`scripts/download_toolguard_data.py` 从 HuggingFace 镜像下载

## 产物与复现溯源

每个 artifact 都能追溯到产生它的代码/任务池版本。**任务池随版本演进**（V1 20 任务 → V2 32 任务 → V3 31 任务、漂移加固 GT），因此历史 artifact 的数字与当前脚本默认输出不保证逐位一致；历史数字用对应 commit 的代码复现，当前主口径用 V3。

| Artifact | 实验 | 产生 commit | 复现 |
|---|---|---|---|
| `metrics.json` | 离线真实轨迹评测 v3（LOMO / cross_model / cross_group，严格 schema 漏斗） | 本轮证据链修复（生成脚本新增） | 下载/解析轨迹后依次运行 `eval_toolguard_rules.py`（默认 + `--lomo-train-policy exclude-fold`）与 `python scripts/build_metrics_artifact.py`（见快速开始） |
| `live_demo.json` | live v1：6 API / 27 短任务 / 两条件 | `b0564ae` | `python scripts/run_live_demo.py` |
| `live_demo_hard.json` | 困难任务 V1：20 任务 / 单预算 | `02ac71a` | checkout `02ac71a` 后运行 V1 脚本 |
| `live_demo_hard_v2.json` | 单模型预算扫描 V2：32 任务 / 3 档预算 / 重复 | `43c4e89` | checkout `43c4e89` 后运行 `run_live_demo_hard_v2.py` |
| `live_demo_hard_v3.json` | 单模型预算扫描 V3：31 任务 / 漂移加固 GT（当前单模型主口径；本轮用 `--from-records` 重建，重算 failure_mode、修 reproduce 路径，**数字未变**） | `8c76d4c` 产物，本轮未重跑 | `--from-records` 重建；全新运行见快速开始 `--mode calibrate` + `--workers 2` |
| `mcp_demo.json` | MCP V1：单 server / 10 任务 | `b1d7609` | checkout `b1d7609` |
| `mcp_demo_v2.json` | MCP V2：双 server / 崩溃重连 / catalog drift / 16 任务（历史协议） | `a1f8b23` | checkout `a1f8b23` |
| `mcp_demo_v3.json` | MCP V3：baseline/guard **分开校准**、以 baseline fault-free 轮数做预算锚点 | 本轮（重跑） | `--mode calibrate`（两臂分开测）+ `--budget-anchor baseline --workers 2` |
| `live_demo_multimodel.json` | 多模型 V2：3 模型 / 576 格（含直播 karma 任务） | `fb13a1f` | checkout `fb13a1f` |
| `live_demo_multimodel_v3.json` | 多模型 V3：3 模型 / 558 格 / 漂移加固 GT（当前多模型主口径；本轮用 `--from-records` 重建，重算 failure_mode、补 `tasks_per_model`、修 reproduce 路径，**数字未变**） | `dcd5f20` 产物，本轮未重跑 | `--from-records` 重建；全新运行见 `--mode calibrate` + `--mode eval` |
| `live_demo_hard_v3_hnconsistent.json` | HN schema drift **一致 schema 对照**：31 任务 / clean+1 / 3 repeats / 186 格（只改 `get_hn_item.story_id` 类型） | 本轮（重跑） | `--budgets clean+1 --repeats clean+1:3 --hn-schema consistent --workers 2` |

一致性由 `python scripts/validate_toolguard_artifacts.py`（以及 `tests/test_toolguard/test_artifact_contracts.py`）检查：artifact 的 aggregate 中 **success/rate/Wilson/turns/tokens/tool_errors/p 等所有派生字段**都必须能从逐条 records 重算（`src/tool_evolution/toolguard/artifact_metrics.py`），`metrics.json` 的比率与 schema 漏斗由同一模块复算；必须有 limitations/reproduce、复现脚本必须存在、多模型 `tasks_per_model` 快照必须与每条记录的 `expected` 一致。篡改上述任一字段都会被 `tests/test_toolguard/test_artifact_tamper.py` 的负例抓住。V3 多模型 artifact 产生于关系验证器之前，用新验证器离线重判 **504 条比较类记录，success 变化 0**（`python scripts/rescore_toolguard_verifier.py`，只读、不改 artifact）。

## 快速开始

```bash
python scripts/download_toolguard_data.py --data-dir .scratch/toolguard
python scripts/parse_toolbench_trajectories.py --input .scratch/toolguard/baselines/data_baselines --output-dir .scratch/toolguard
# 离线评测 + 确定性重建 metrics.json（严格 schema 漏斗；无需 API key）
python scripts/eval_toolguard_rules.py --output .scratch/toolguard/rules_eval.json
python scripts/eval_toolguard_rules.py --lomo-train-policy exclude-fold \
  --output .scratch/toolguard/rules_eval_lomo_true.json
python scripts/build_metrics_artifact.py --parser-stats .scratch/toolguard/stats.json
python scripts/toolguard_cli.py summary

# 修复一个失败调用
python scripts/toolguard_cli.py repair \
  --tool-name trends_for_youtube_music_api_detailed \
  --arguments '{"country": "ZZ"}' \
  --error "The country code 'ZZ' is not recognized or not supported."

# live 端到端 demo（需要 .env 中的 TOOLEVO_DEEPSEEK_API_KEY；真实公开 API，无其他 key）
python scripts/run_live_demo.py
python scripts/run_live_demo.py --max-turns 4 --p-timeout 0.35 --p-rate-limit 0.15

# 单模型预算扫描（V3 任务池 31 任务 + 漂移加固 GT）：先校准无故障轮数，再跑 3 档预算 + 重复（约 30 分钟，2 worker）
python scripts/run_live_demo_hard_v2.py --mode calibrate --calibration .scratch/toolguard/hard_calibration_v3.json
python scripts/run_live_demo_hard_v2.py --workers 2 --calibration .scratch/toolguard/hard_calibration_v3.json \
  --output benchmarks/toolguard/live_demo_hard_v3.json

# MCP 集成：双 stdio server + server 崩溃重连 + catalog drift（需要真实进程管道，不能在工作区沙箱内运行）
python scripts/run_mcp_demo_v2.py --mode calibrate
python scripts/run_mcp_demo_v2.py --workers 2 --output benchmarks/toolguard/mcp_demo_v2.json
# MCP baseline/guard 分开校准（2026-11-24 P1-2 修正）：两臂各测一次无故障轮数，预算锚定 baseline
python scripts/run_mcp_demo_v2.py --mode calibrate --calibration .scratch/toolguard/mcp_calibration_v3.json
python scripts/run_mcp_demo_v2.py --budget-anchor baseline --workers 2 \
  --calibration .scratch/toolguard/mcp_calibration_v3.json \
  --output benchmarks/toolguard/mcp_demo_v3.json

# HN schema drift 一致 schema 对照实验（2026-11-24 P1-1）：只把 get_hn_item.story_id 改成 integer
python scripts/run_live_demo_hard_v2.py --budgets clean+1 --repeats clean+1:3 --hn-schema consistent \
  --workers 2 --calibration .scratch/toolguard/hard_calibration_v3.json \
  --output benchmarks/toolguard/live_demo_hard_v3_hnconsistent.json
# V3 artifact 从 checkpoint 记录重建（免费、不调用 API）：重算 aggregate/failure_mode、修 reproduce 路径
python scripts/run_live_demo_hard_v2.py --from-records --output benchmarks/toolguard/live_demo_hard_v3.json

# 多模型复制：需要 TOOLEVO_QWEN_API_KEY（qwen3.8-flash + qwen4-max）与 TOOLEVO_DEEPSEEK_API_KEY
python scripts/check_providers.py --smoke        # 先验证 key / 工具调用 / model id（不打印 key）
python scripts/run_multimodel_live.py --mode calibrate
python scripts/run_multimodel_live.py --mode eval --workers 2 --output benchmarks/toolguard/live_demo_multimodel_v3.json
python scripts/run_multimodel_live.py --from-records --output benchmarks/toolguard/live_demo_multimodel_v3.json  # 从 checkpoint 重建 artifact

# artifact 一致性（aggregate ↔ records、reproduce/limitations 完整）
python scripts/validate_toolguard_artifacts.py
# 从 artifact 重算所有主数字（供审阅/复核，无需 API key）
python scripts/toolguard_evidence_summary.py
# 用关系验证器离线重判已提交 artifact 的比较类答案（只读，不改 artifact）
python scripts/rescore_toolguard_verifier.py

python -m pytest tests/test_toolguard -q -p no:cacheprovider
```

> task-pool 版本：V1 20 任务、V2 32 任务、V3 31 任务（漂移加固）。历史 artifact 用对应 commit 的代码复现（见上表），**当前主口径：单模型 `live_demo_hard_v3.json`，多模型 `live_demo_multimodel_v3.json`，MCP `mcp_demo_v3.json`，HN 对照 `live_demo_hard_v3_hnconsistent.json`**。
> 多模型轴的 key 放在仓库根 `.env`（见 [.env.example](.env.example)），`.env` 已 gitignore，永不提交。

artifact：[`benchmarks/toolguard/metrics.json`](benchmarks/toolguard/metrics.json)（含完整口径、分规则验证与局限）

## 局限与下一步

- **离线覆盖率采用两层证据，且声明 schema 本身有保真度上限**：`schema_valid_rate` 是 81.0%（LOMO，含 150 条 coercion）/ 71.1%（cross-model）/ 52.6%（cross-group），不再是 100%；accepted 覆盖 = schema-valid + value-verified。上游 `api_list` 把部分日期参数声明成 number（真实成功调用却用 ISO 日期字符串）、部分函数参数列表为空，因此 cross-group 的 27 条文本抽取修复只能通过“与真实成功调用逐字一致”这一层计入。**不能**再把 54.7% / 75% 说成“schema 合法修复覆盖率”；
- **LOMO 名不副实（已如实标注）**：主协议是“固定三模型训练 universe、逐模型留出”，gpt-4-0613 / gpt-4-turbo-preview 从不进入训练；真 LOMO 敏感性为 **542/969=55.9%**（`results.lomo_sensitivity_exclude_fold`）；
- 离线 replay：缓存只覆盖参考解输入（最终尝试命中率 51.6%），修复验证命中 **48/531=9.0%**，未命中不等于失败；
- value agreement 只在“某模型解过同一任务”时可验证，是严格下界；
- 服务端/瞬态错误（全数据集 96 条）离线不可确定性修复；
- 数据解析把 **20 条**无法解析的 observation 按成功处理（0.07%），可能低估真实失败；30,120 是**记录调用/重试次数**，来自 7,636 条 trace（764 个唯一 query、2,303 个唯一 (query, tool)），不是 30,120 个独立任务；
- live 单模型结果来自 `deepseek-v4-flash`；多模型轮覆盖 **qwen3.8-flash / DeepSeek-V4.1-Flash / qwen4-max** 三个模型（cheap/strong 同属 Qwen 家族），是**复制**而非模型间假设检验；
- 故障/崩溃/漂移都是**确定性注入**，不是生产观测；`clean+N` 预算锚定在“每任务一次无故障 calibration”，calibration 本身是点估计；
- `clean+1` 是受控压力条件，不能解读为“真实生产故障率下 baseline 会失败 45–53%”；正确读法是**成功-预算曲线**：预算越低 baseline 越差、guard 恒定；预算充足时成功率持平、guard 成本更低；
- 任务池随版本演进（V1 20 / V2 32 / V3 31 任务，且 V3 换了 ground truth）：**不同 artifact 的绝对数字不要横向逐位比较**；同一 artifact 内的 baseline-vs-guard 配对才是有效证据，跨版本请以 V3 主口径为准；
- 直播可变 ground truth：V3 已把会漂移的 HN karma 换成不可变账号创建年份，并在每个模型块前重建 ground truth；但天气/FX 仍是当前值，跨模型块/跨 run 的瞬时偏差仍可能带来极小量假阴性；artifact 的 tasks 元数据如从 checkpoint 重建，数值可能与运行时刻略有差异（`--from-records` 现在会从记录的 `expected` 快照重建 `tasks_per_model`，不再丢快照）；
- **验证器（P0-3）**：比较类任务已改为关系断言（winner 必须被断言、与数值排序一致，处理否定词/million 等单位/多候选；V1 的版本比较也改成真正的版本号比较）。用新验证器离线重判已提交 artifact 的 **504 条**比较类记录，success **0 变化**（397→397），说明已提交数据未被发现污染；但旧验证器无法排除反向答案，旧成功率只能视为上界。V1 两个 artifact 没有保存逐条 answer，无法离线重判；
- **失败模式分类（P0 附带）**：旧 `classify_failure()` 会被 expected 里的 story_id/经纬度等 metadata 标量误触发，把“完全没回答”标成 `format_miss`。代码已修（只考虑答案字段）；两个 V3 artifact 已用 `--from-records` 从 checkpoint 重建并**重算了 failure_mode**（多模型 V3：format_miss 20→3、no_expected_value 2→19，其余标签不变），reproduce 输出路径同时修正；**未重跑 live**；
- **HN schema drift 占比 + 一致 schema 对照实验（P1-1，已执行）**：旧 V3（HN schema 故意不一致）clean+1 baseline 51/93（54.8%），其中 HN 子集 2/12、非 HN 49/81；guard 93/93。新做的**一致 schema 对照**（把 `get_hn_item.story_id` 改成 integer，其余任务/seed/预算完全相同，186 格）baseline **61/93（65.6%）**：HN 子集 **12/12**、非 HN **49/81（与旧 run 逐格相同）**；guard 93/93。归因：HN 契约缺陷贡献 baseline-vs-guard 差距的 **10/42 = 23.8%**，其余 32 格来自真实的瞬态故障修复；非 HN 子集两次运行完全一致，说明对照干净。成本 $0.18、墙钟 658s。artifact：[`benchmarks/toolguard/live_demo_hard_v3_hnconsistent.json`](benchmarks/toolguard/live_demo_hard_v3_hnconsistent.json)；
- **MCP baseline 预算不对称（P1-2，已重跑修正）**：旧 V2 用 guard 校准 budget；新 `mcp_demo_v3.json` 两臂分别校准、预算锚定 baseline fault-free 轮数。结果：clean+1 baseline **13/32（40.6%）** vs guard **32/32（100%）**（p=3.81e-06）、clean+4 14/16 vs 16/16；baseline 在 fx 任务上获得更多预算（这是更公平而非更弱的对照），guard 优势仍为 +59.4pp。旧 V2 artifact 保留为历史；
- qwen4-max 在本任务集上没有表现出比 DeepSeek-Flash 更强的 fault-free 能力（V3：29/31 vs 31/31），"贵模型=更可靠"不成立；
- MCP V2 中 server 崩溃（第 2 次调用 `os._exit`）与 catalog drift（`fetch_exchange_rate` 别名）也是注入场景，其余（stdio 传输、进程死亡与重连、多 server 路由、HTTP、LLM）都是真实的；
- MCP stdio 需要真实子进程管道（受限沙箱内需放宽权限），测试用 MCP SDK 内存 transport 覆盖 adapter 逻辑；
- 免费公开 API 非生产级：GitHub 限流已从任务集移除，API 漂移可能影响复跑；
- 下一步（待执行）：① MCP baseline/guard 分开校准的新 artifact 结果已写入 `mcp_demo_v3.json`（见“产物与复现溯源”与局限）；② 继续修正 schema 索引保真度（日期参数类型、空参数列表）后再看离线覆盖是否还能提升；③ 用真实线上故障 trace 替代部分注入；④ 两个较强模型的紧预算 guard 已到 100%，需要更难的题/更紧的预算；⑤ 把 HN 一致 schema 对照实验扩展到多模型（本轮只做了单模型）。
