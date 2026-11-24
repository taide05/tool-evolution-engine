# ToolGuard — tool-call reliability middleware for LLM agents

ToolGuard 是一个夹在 Agent 与工具之间的中间件：**采集每次工具调用 → 分类失败 → 从真实失败轨迹中学习修复策略 → 运行时修复并记录指标**。它不训练模型、不改工具实现，只做工具层的确定性修复，因此可解释、低成本、易部署。

当前版本有三层证据：

1. **离线真实数据评测**：在 **30,120 次真实工具调用 / 1,033 次真实失败**（StableToolBench 公开基线轨迹，Apache-2.0）上验证修复策略，并重点做了**跨任务泛化**：把“重放同一任务的旧调用”升级为“从任务文本学习参数抽取规则”；
2. **live 端到端 A/B**：真实 DeepSeek agent + 真实免 key 公开 API，从 27 个短任务到 **32 个多步依赖任务**；V2 把交互预算做成**三档扫描 + 重复**，给出成功-预算曲线、Wilson CI 与配对检验；第四轮再把同一协议**复制到 3 个模型**（qwen3.8-flash / DeepSeek-V4.1-Flash / qwen4-max）；
3. **真实 MCP 集成**：同一套中间件不改一行，接到**两个真实 stdio MCP server** 上，覆盖多 server 路由、server 进程真实崩溃重连、catalog drift 未知工具重解析。

## 结果（离线确定性评测）

| 协议 | 可评估失败 | 修复覆盖 | 其中**可验证**（参数值与真实成功调用一致） | schema 合法率 | 缓存验证 |
|---|---|---|---|---|---|
| **LOMO（5 折留一模型）** | 969 | **54.7%**（530） | 89 例（275 例可验证中 32.4%） | **100%** | 48 恢复 / **0 反例** |
| cross-model（3→2 模型） | 277 | 58.1% | 48 例（143 例可验证中 33.6%） | 100% | 15 恢复 / 0 反例 |
| **cross-group（跨任务集，未见任务）** | 76 | **75.0%**（此前 0%） | **27 例（38 例可验证中 71.1%）** | 100% | 全部 cache miss |

**跨任务泛化的关键结果（cross-group）**：
- 只靠“实例记忆”时覆盖率 **0%**（同任务没被见过 → 没有旧成功调用可重放）；
- 加入**从任务文本学习的参数抽取器**后，覆盖率达到 **75%**，其中 **27 条修复的参数值与真实成功调用完全一致（27/27，100% value agreement）**；
- 另有 30 条来自“schema 默认值补全”的修复是**未经验证的启发式提议**（value agreement 0），在 artifact 中单独标注、不计入质量证据。

**规则类别（cross-model / cross-group）**：

| 规则 | cross-model | cross-group | 说明 |
|---|---|---|---|
| 任务文本抽取（学习得到） | 12 修复 / 12 一致 | **27 修复 / 27 一致** | 日期、年份、引号关键词、数字、国家码 |
| 经验记忆重放 | 57 修复 / 36 一致 | — | 同任务、其他模型跑成功过 |
| 学习式参数众数 | 10 修复 | — | 同函数的成功参数分布 |
| schema 默认值补全 | 72（未验证） | 30（未验证） | 错误未给参数名时的启发式 |

两协议合计：**学习式文本抽取 39/39 全部与真实成功调用一致**。

## Live 端到端 demo（真实公开 API + 真实 DeepSeek agent）

三轮递进验证，全部使用真实 DeepSeek `deepseek-v4-flash` agent、真实免 key 公开 API、程序化答案校验（**无 LLM judge**）。合成元素只有**确定性的故障/崩溃/配置漂移注入**，全部在 artifact 中逐项披露；DeepSeek 或公开 API 不可用时 demo 停止报错，**不会回退离线/mock**。

### 第一轮：27 个短任务 / 6 个 API（可靠性与成本口径）

6 个真实公开 API（World Bank、PyPI、Open-Meteo、Hacker News、GitHub、汇率），27 个可自动校验任务，middleware on/off 配对 A/B。故障注入 = 每个逻辑调用的首次尝试可能被注入 timeout/rate-limit，模拟离线分析里发现的瞬态失败类。

| 条件 | 任务成功 | 工具调用失败 | 工具调用数 | 轮数 | tokens |
|---|---|---|---|---|---|
| normal（6 轮预算）baseline | 27/27 | 14 | 44 | 68 | 57,042 |
| normal guard | **27/27** | **1**（-93%） | 31（-30%） | 55（-19%） | **44,217（-22.5%）** |
| stress（4 轮预算 + 更高故障率）baseline | 25/27 | 20 | 48 | 71 | 59,409 |
| stress guard | **26/27**（McNemar p=1.0） | **4**（-80%） | 33（-31%） | 57（-20%） | **46,295（-22.1%）** |

**诚实结论**：27 个任务对 baseline 太容易（normal 双方 27/27），所以这一轮**任务成功率没有显著提升**；价值体现在工具失败 **-80%~-93%**、tokens **-22%**、轮数 **-19%~-20%**。这是“可靠性与成本”结果，不是任务成功率结果。artifact：[`benchmarks/toolguard/live_demo.json`](benchmarks/toolguard/live_demo.json)。

### 第二轮：32 个多步依赖任务 × 预算扫描 × 重复（任务成功率口径）

#### 为什么从 V1 重做（自我批评）

V1（20 任务 / 单预算 / 单次运行）的问题：①任务量和重复不足，只有点估计没有方差；②单预算恰好卡在“无故障需求 +1 轮”，容易被质疑“把预算调到让 baseline 死”；③单模型、指标面窄。V2 把**预算公开成实验变量**并加重复：

- 任务池 20 → **32**：新增 5–8 步、派生算术（比值/差值/求和）、跨源日期比较、三国人口聚合等 12 个任务；类别覆盖 weather 8 / worldbank 12 / pypi 5 / hn 4 / fx 3；
- **预算扫描**：每任务测 `clean+1` / `clean+4` / `10` 三档（clean = 无故障 calibration 轮数），得到**成功-预算曲线**；
- **重复**：紧预算 `clean+1` 用 3 个故障 seed（96 格/臂），其余各 1 个 seed（32 格/臂）；报告 Wilson 95% CI 与配对 McNemar（按 task×repeat 配对）；
- 故障注入同 V1（每个逻辑调用前 2 次尝试失败，确定性），单模型 `deepseek-v4-flash`（决策：不加模型轴，预算用于重复）；
- 共 320 格（3 预算档 × 2 臂 × 32 任务 + 重复），墙钟 1,619s，2 个 worker。

#### 结果

| 预算条件 | 平均预算 | baseline 成功 | guard 成功 | 配对 McNemar | turns/格 (b→g) | tokens (b→g) | agent 可见工具错误 (b→g) |
|---|---|---|---|---|---|---|---|
| **clean+1** | 5.0 | **53/96 (55.2%, CI 45.3–64.8%)** | **96/96 (100%, CI 96.2–100%)** | **p=2.27e-13** | 4.41→3.05 | 657,860→430,769 | **266→6** |
| clean+4 | 8.0 | 29/32 (90.6%) | 32/32 (100%) | p=0.25 (n.s.) | 5.59→3.00 | 299,273→141,751 | 99→0 |
| 10 | 10.0 | 29/32 (90.6%) | 32/32 (100%) | p=0.25 (n.s.) | 5.75→3.06 | 310,950→143,643 | 104→0 |

**成功-预算曲线**：baseline 55.2% → 90.6% → 90.6%（预算增加才恢复），guard **三档都是 100%** 且 turns/tokens 基本不随预算变化。这直接回答“是不是把预算调死”的质疑：预算越低 baseline 越差、guard 不变；预算充足时两者成功率持平，guard 的成本优势更大（tokens -52.6%）。

其他发现：
- 即使 10 轮预算，baseline 仍有 **3 个 HN 任务失败**（`story_id` schema/后端类型不匹配让 baseline 反复重试直到耗尽，而 guard 在同一轮内完成类型修复）；
- clean+1 的 43 个 baseline 失败全部是 `no_answer`（预算耗尽）；guard 侧只有 6 个 agent 可见的残余工具错误；
- guard 的重试退避总开销 **18s / 96 格 ≈ 187ms/格**（可忽略），换来 118 次轮内恢复；
- V2 还修掉了 V1 从轨迹里发现的缺陷：修复后的调用若再遇瞬态故障，现在会在同一个 call 内继续重试（`error_hint_coercion+transient_retry`，commit `13af709`）。

artifact：[`benchmarks/toolguard/live_demo_hard_v2.json`](benchmarks/toolguard/live_demo_hard_v2.json)（含逐格记录、calibration、逐条修复事件、Wilson CI、预算曲线、失败模式与局限）。

> **V3 任务集修订**：V2 之后发现 HN `karma` 类任务是**直播可变 ground truth**（同一作者 karma 在一小时内 112→161），会把数据漂移误记成修复失败。V3 用**同样的两工具链**把 3 个 karma 任务换成**不可变**的“账号创建年份”任务，任务池变为 **31 个**（weather 8 / worldbank 12 / pypi 5 / hn 3 / fx 3）；多模型评测还在**每个模型块开始前重建 ground truth**，把漂移窗口从整轮约 1 小时压缩到单个模型块（约 20 分钟）。V2 的 32 任务数字由 commit `43c4e89` 的代码产生，保留为历史。

### 第三轮：真实 MCP 双 server + transport 崩溃 + catalog drift（产品化口径）

V2 的 MCP 验证从“同一中间件在 MCP 上也能重试”升级为**部署/传输层容错**：

- **双 MCP stdio server**：`--group worldbank` 与 `--group market` 两个独立进程，客户端 adapter 合并工具表并按工具路由（真实多 server 部署形态）；
- **真实 transport 故障**：worldbank server 进程在第 2 次调用时 `os._exit(17)` 硬崩溃，stdio session 真实断链；adapter 在下一个调用时重连新进程，中间件把断链当作可重试的 `request_error` 在同一 agent 轮内重试；
- **catalog drift / 未知工具**：客户端广告旧名字 `fetch_exchange_rate`，server 实际注册的是 `get_exchange_rate`；调用得到 MCP unknown-tool 错误后，中间件用离线验证过的**函数名相似度规则（≥0.55）**在真实工具表上解析并重路由；
- server 侧瞬时故障 epoch 照旧；16 个多步任务、预算扫描 `clean+1` / `clean+4`、`clean+1` 两 repeat（96 格），每格**全新两个 server 进程**；模型 `deepseek-v4-flash`。

| 预算条件 | baseline 成功 | guard 成功 | 配对 McNemar | turns/格 (b→g) | tokens (b→g) | agent 可见工具错误 (b→g) |
|---|---|---|---|---|---|---|
| **clean+1** | **11/32 (34.4%, CI 20.4–51.7%)** | **32/32 (100%, CI 89.3–100%)** | **p=9.54e-07** | 4.56→3.25 | 205,600→134,998 | **115→4** |
| clean+4 | 14/16 (87.5%) | 16/16 (100%) | p=0.5 (n.s.) | 5.81→3.13 | 138,002→63,679 | 63→1 |

- 真实崩溃/重连在整个实验里发生 **17–20 次**（`mcp_restarts`），全部被中间件在同一 agent 轮内吸收；
- guard 修复规则计数：`transient_retry` 77 次、**`function_name_resolution` 9 次**（另有 6 次与瞬态重试叠加）——catalog drift 确实由离线学到的函数名解析规则救回；
- 传输无关性验证：`ToolGuard` 本体一行未改，新增的是 `MCPBridge`（含重连）/`MCPMultiBackend`（多 server 路由 + 未注册别名）adapter；
- 测试：MCP SDK 内存 transport 集成测试覆盖 schema 转换、`isError`→taxonomy、参数修复、transient 重试、双 server 路由、catalog drift 解析；全仓 toolguard 测试 **50 passed**。

artifact：[`benchmarks/toolguard/mcp_demo_v2.json`](benchmarks/toolguard/mcp_demo_v2.json)

> V1 的窄口径 artifact 保留作为演进记录：`live_demo_hard.json`（20 任务单预算，commit `02ac71a`）、`mcp_demo.json`（10 任务单 server，commit `b1d7609`）；对外口径以 V2 为准。

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
- **能力越弱，中间件收益越大**：qwen3.8-flash 紧预算提升 **+46.7pp**（45.2%→91.9%），另两个模型 **+48.4pp / +45.2pp**；但 guard 绝对水平上，两个较强模型做到 **100%（62/62）**，弱模型 91.9%；
- **贵模型没有买到“无故障鲁棒性”**：qwen4-max 单价比 qwen3.8-flash 高约 9 倍，但 fault-free placement（29/31、3.29 轮）没有超过 DeepSeek-Flash（31/31、3.32 轮）。三个模型都会踩同一类失败（HN `story_id` 契约不匹配），说明这类问题不是模型能力问题而是**工具契约问题**——正好是中间件该管的；
- **V3 后 guard 残余失败已不再是数据漂移**：qwen3.8-flash 的 5 个 clean+1 guard 失败全部可归因于**弱模型自身行为**（如声称 `get_weather` 不可用、GDP 数值编造 $106,588、答非所问回答 France），另两个模型 guard 失败为 0。V2 里那些 HN karma 漂移造成的假阴性已消失；
- 成本：评测 $2.08（qwen4-max 占 80%）+ calibration $0.30 ≈ **$2.38**（数据集比 V2 少一个任务，故略低）。

artifact：[`benchmarks/toolguard/live_demo_multimodel_v3.json`](benchmarks/toolguard/live_demo_multimodel_v3.json)（558 格逐条记录、三模型 placement、Wilson CI、成本核算、逐条修复事件、`tasks_per_model` 与每条记录的 `expected` 快照）。V2 版 artifact（`benchmarks/toolguard/live_demo_multimodel.json`，含 karma 任务）保留为演进记录，复现见 commit `fb13a1f`。

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

所有修复必须通过 **schema 合法性检查**（函数存在、参数都在 schema 内、必填齐全）才计入；另外做两类外部验证：
- **value agreement**：修复后的参数值是否与同一任务/函数某次真实成功调用完全一致；
- **cache verification**：修复后的调用是否在真实响应缓存中命中且返回成功。

## 数据来源

- 轨迹：StableToolBench baseline trajectory release（Apache-2.0，公开）
- 工具 schema：同一 release 的 solvable_queries `api_list`（2,461 个函数）
- 真实响应缓存：ToolBench Apr-2024 缓存（仅用于修复结果的补充验证；覆盖参考解输入，最终尝试命中率 51.6%）
- 数据不随仓库提交；`scripts/download_toolguard_data.py` 从 HuggingFace 镜像下载

## 快速开始

```bash
python scripts/download_toolguard_data.py --data-dir .scratch/toolguard
python scripts/parse_toolbench_trajectories.py --input .scratch/toolguard/baselines/data_baselines --output-dir .scratch/toolguard
python scripts/eval_toolguard_rules.py
python scripts/toolguard_cli.py summary

# 修复一个失败调用
python scripts/toolguard_cli.py repair \
  --tool-name trends_for_youtube_music_api_detailed \
  --arguments '{"country": "ZZ"}' \
  --error "The country code 'ZZ' is not recognized or not supported."

# live 端到端 demo（需要 .env 中的 TOOLEVO_DEEPSEEK_API_KEY；真实公开 API，无其他 key）
python scripts/run_live_demo.py
python scripts/run_live_demo.py --max-turns 4 --p-timeout 0.35 --p-rate-limit 0.15

# V2 困难多步任务集：先校准无故障轮数，再跑 3 档预算扫描 + 重复（约 27 分钟，2 worker）
python scripts/run_live_demo_hard_v2.py --mode calibrate
python scripts/run_live_demo_hard_v2.py --workers 2 --output benchmarks/toolguard/live_demo_hard_v2.json

# V2 MCP 集成：双 stdio server + server 崩溃重连 + catalog drift（需要真实进程管道，不能在工作区沙箱内运行）
python scripts/run_mcp_demo_v2.py --mode calibrate
python scripts/run_mcp_demo_v2.py --workers 2 --output benchmarks/toolguard/mcp_demo_v2.json

# 多模型复制：需要 TOOLEVO_QWEN_API_KEY（qwen3.8-flash + qwen4-max）与 TOOLEVO_DEEPSEEK_API_KEY
python scripts/check_providers.py --smoke        # 先验证 key / 工具调用 / model id（不打印 key）
python scripts/run_multimodel_live.py --mode calibrate
python scripts/run_multimodel_live.py --mode eval --workers 2 --output benchmarks/toolguard/live_demo_multimodel.json
python scripts/run_multimodel_live.py --from-records --output benchmarks/toolguard/live_demo_multimodel.json  # 从 checkpoint 重建 artifact

python -m pytest tests/test_toolguard -q -p no:cacheprovider
```

> V1 脚本（`run_live_demo_hard.py` / `run_mcp_demo.py`）仍可复现 V1 artifact，但对外口径以 V2/V3 为准。
> 多模型模型轴里的 key 放在仓库根 `.env`（见 [.env.example](.env.example)），`.env` 已 gitignore，永不提交。

artifact：[`benchmarks/toolguard/metrics.json`](benchmarks/toolguard/metrics.json)（含完整口径、分规则验证与局限）

## 局限与下一步

- 离线 replay：缓存只覆盖参考解输入（最终尝试命中率 51.6%），修复验证存在未命中，**未命中不等于失败**；
- value agreement 只在“某模型解过同一任务”时可验证，是严格下界；
- schema 默认值补全的 30 条 cross-group 修复未经验证，属启发式提议；
- 服务端/瞬态错误（全数据集 96 条）离线不可确定性修复；
- live 单模型结果来自 `deepseek-v4-flash`；多模型轮覆盖 **qwen3.8-flash / DeepSeek-V4.1-Flash / qwen4-max** 三个模型（cheap/strong 同属 Qwen 家族），是**复制**而非模型间假设检验；
- 故障/崩溃/漂移都是**确定性注入**，不是生产观测；`clean+N` 预算锚定在“每任务一次无故障 calibration”，calibration 本身是点估计；
- `clean+1` 是受控压力条件，不能解读为“真实生产故障率下 baseline 会失败 45–53%”；正确读法是**成功-预算曲线**：预算越低 baseline 越差、guard 恒定；预算充足时成功率持平、guard 成本更低；
- 直播可变 ground truth：V3 已把会漂移的 HN karma 换成不可变账号创建年份，并在每个模型块前重建 ground truth；但天气/FX 仍是当前值，跨模型块的瞬时偏差仍可能带来极小量假阴性（V3 guard 残余失败已全部可归因于模型自身行为）；artifact 的 tasks 元数据如从 checkpoint 重建，数值可能与运行时刻略有差异；
- qwen4-max 在本任务集上没有表现出比 DeepSeek-Flash 更强的 fault-free 能力（30/32 vs 31/32），"贵模型=更可靠"不成立；
- MCP V2 中 server 崩溃（第 2 次调用 `os._exit`）与 catalog drift（`fetch_exchange_rate` 别名）也是注入场景，其余（stdio 传输、进程死亡与重连、多 server 路由、HTTP、LLM）都是真实的；
- MCP stdio 需要真实子进程管道（受限沙箱内需放宽权限），测试用 MCP SDK 内存 transport 覆盖 adapter 逻辑；
- 免费公开 API 非生产级：GitHub 限流已从任务集移除，API 漂移可能影响复跑；
- 下一步：用真实线上故障 trace 替代部分注入以进一步逼近生产；把“默认值补全”类修复变成可验证规则；两个较强模型的紧预算 guard 已到 100%，需要更难的题/更紧的预算才有提升空间。
