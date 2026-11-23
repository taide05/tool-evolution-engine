# ToolGuard — tool-call reliability middleware for LLM agents

ToolGuard 是一个夹在 Agent 与工具之间的中间件：**采集每次工具调用 → 分类失败 → 从真实失败轨迹中学习修复策略 → 运行时修复并记录指标**。它不训练模型、不改工具实现，只做工具层的确定性修复，因此可解释、低成本、易部署。

当前版本有三层证据：

1. **离线真实数据评测**：在 **30,120 次真实工具调用 / 1,033 次真实失败**（StableToolBench 公开基线轨迹，Apache-2.0）上验证修复策略，并重点做了**跨任务泛化**：把“重放同一任务的旧调用”升级为“从任务文本学习参数抽取规则”；
2. **live 端到端 A/B**：真实 DeepSeek agent + 真实免 key 公开 API，从 27 个短任务到 20 个**多步依赖任务**，在配对实验里量化可靠性与交互成本；
3. **真实 MCP 集成**：同一套中间件不改一行，接到真实 stdio MCP server 上，验证“换工具层只换 adapter”。

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

三轮递进验证，全部使用真实 DeepSeek `deepseek-v4-flash` agent、真实免 key 公开 API、程序化答案校验（**无 LLM judge**）。唯一的合成元素是**确定性瞬时故障注入**；DeepSeek 或公开 API 不可用时 demo 停止报错，**不会回退离线/mock**。

### 第一轮：27 个短任务 / 6 个 API（可靠性与成本口径）

6 个真实公开 API（World Bank、PyPI、Open-Meteo、Hacker News、GitHub、汇率），27 个可自动校验任务，middleware on/off 配对 A/B。故障注入 = 每个逻辑调用的首次尝试可能被注入 timeout/rate-limit，模拟离线分析里发现的瞬态失败类。

| 条件 | 任务成功 | 工具调用失败 | 工具调用数 | 轮数 | tokens |
|---|---|---|---|---|---|
| normal（6 轮预算）baseline | 27/27 | 14 | 44 | 68 | 57,042 |
| normal guard | **27/27** | **1**（-93%） | 31（-30%） | 55（-19%） | **44,217（-22.5%）** |
| stress（4 轮预算 + 更高故障率）baseline | 25/27 | 20 | 48 | 71 | 59,409 |
| stress guard | **26/27**（McNemar p=1.0） | **4**（-80%） | 33（-31%） | 57（-20%） | **46,295（-22.1%）** |

**诚实结论**：27 个任务对 baseline 太容易（normal 双方 27/27），所以这一轮**任务成功率没有显著提升**；价值体现在工具失败 **-80%~-93%**、tokens **-22%**、轮数 **-19%~-20%**。这是“可靠性与成本”结果，不是任务成功率结果。artifact：[`benchmarks/toolguard/live_demo.json`](benchmarks/toolguard/live_demo.json)。

### 第二轮：20 个多步依赖任务 / 12 个工具（任务成功率口径）

针对第一轮“太容易”的问题重新设计任务集：每个任务需要 **2–6 次有依赖关系的跨工具调用**，例如

- 国家 → 首都 → `geocode_city` 坐标 → 天气（坐标不在题面里，必须走工具链）；
- 首都 → 反查国家 → 人口 / GDP per capita / 收入等级 / 地区（World Bank）；
- 包名 → 最新版本 → 该版本的发布日期（PyPI），再比较两个包谁更新；
- Hacker News item → 作者 → 作者 karma；
- EUR→USD、USD→JPY 两个真实汇率相乘得到隐含 EUR→JPY 交叉汇率。

**协议**（`scripts/run_live_demo_hard.py`）：
1. 先跑 20 次**无故障 calibration**，测出每个任务的“无故障最少交互轮数”；
2. `budgeted` 条件：每任务轮数预算 = calibration 轮数 **+ 1**；
3. `generous` 条件：所有任务固定 10 轮；
4. 故障注入：按 (seed, 函数名, 参数) 确定式判定，故障 epoch = 该逻辑调用**前 2 次尝试**失败（timeout / HTTP 429 / HTTP 5xx）。一次 epoch 让 baseline agent 多花 **2 轮**，而 ToolGuard 在**同一个 agent 轮**内重试吸收；
5. 每任务 baseline/guard 各跑一次，做配对 McNemar exact 检验。

| 条件 | 任务成功 | 轮数 | 工具调用 | 工具错误 | tokens | 预算耗尽 |
|---|---|---|---|---|---|---|
| **budgeted** baseline | **8/20（40.0%）** | 85 | 95 | 51 | 128,518 | 12 |
| **budgeted** guard | **19/20（95.0%）** | 68（-20%） | 62（-35%） | **6（-88%）** | **97,274（-24%）** | **0** |
| **generous** baseline | 19/20（95.0%） | 107 | 104 | 51 | 170,601 | 1 |
| **generous** guard | **20/20（100%）** | 68（**-36%**） | 62（-40%） | **6（-88%）** | **97,158（-43%）** | 0 |

配对结果：budgeted **guard_only 12 / baseline_only 1，McNemar p=0.0034**；generous 双方 19/20 vs 20/20（p=1.0）。

**怎么读这个结果（面试口径）**：
- `budgeted` 的预算是按“无故障需求 + 1 轮”校准的；任何一次故障 epoch 都要求 baseline 额外花 2 轮，因此 baseline 在该条件下成功 ≈ “全程未遇到瞬时故障”的概率。这是**指标定义下的必然结果**，度量的是“把瞬态错误暴露给计划循环”的隐藏交互成本，不是模型能力差异。
- `generous` 条件交叉验证：预算充足时成功率无显著差异（19/20 vs 20/20），但 guard 轮数 **-36%**、tokens **-43%**、工具错误 **-88%**。两条件一起说明：差异来自**交互预算**，中间件不改变答案质量。
- guard 唯一失败是**答案格式问题**（回答把人口四舍五入成 "5.61 million"，校验器要求精确整数），与中间件无关；artifact 保留原判定（对 guard 更保守），未做事后修正。
- 从轨迹里发现的真实改进点：类型修复后的调用（如 HN `story_id` 字符串→整数）如果**本身**被注入瞬态故障，当前中间件不会在同一个 call 内继续重试修复后的调用（只重试原始调用），hn 任务各浪费 2 轮；本轮 artifact 保持该行为未改，列入下一步。

artifact：[`benchmarks/toolguard/live_demo_hard.json`](benchmarks/toolguard/live_demo_hard.json)（含 calibration、逐任务结果、逐条修复事件、协议与局限）。

### 第三轮：真实 MCP server 集成（产品化口径）

把同一套中间件接到真实 **MCP（Model Context Protocol）** 工具层：

- `scripts/mcp_toolguard_server.py`：真实的 stdio MCP server（`mcp` SDK 1.26.0 / FastMCP），暴露与第二轮相同的 **12 个 live 工具**；工具错误以 MCP `isError` 返回，并带 `TOOL_ERROR[kind]` 结构化标记；
- `src/tool_evolution/toolguard/mcp_backend.py`：`MCPBridge` 在后台事件循环里跑真实 stdio session（子进程 + stdin/stdout 管道），`MCPBackend` 把 `tools/list` 的 `inputSchema` 转成 function schema，把 `tools/call` 的 `isError` 映射回中间件离线学到的错误 taxonomy（timeout / rate_limit / invalid_argument / …）；
- **`ToolGuard` 中间件一行未改**：同一套 retry / coercion / fallback 规则直接作用于 MCP 工具层。

**协议**：10 个多步任务（第二轮任务集的子集），同样的 calibration（无故障 MCP 运行）+1 轮预算；故障注入放在 **MCP server 侧**（确定性 `isError` epoch），每个 arm 启动**全新 server 进程**，两侧面对同一故障面。

| 条件 | 任务成功 | 轮数 | MCP 调用 | agent 可见工具错误 | tokens | 预算耗尽 |
|---|---|---|---|---|---|---|
| baseline | **4/10（40%）** | 36 | 41 | 18 | 44,949 | 6 |
| guard | **10/10（100%）** | 30（-17%） | 27（-34%） | **0** | **36,842（-18%）** | **0** |

配对：guard_only 6 / baseline_only 0，**McNemar p=0.031**；guard 内部完成 **11 次 `transient_retry`**（server 实际返回 22 次错误、agent 只看到 0 次）。这说明中间件对 MCP 工具层是“传输无关”的：换协议只需换 adapter（约 250 行），规则与证据链不变。

MCP 集成测试：`tests/test_toolguard/test_mcp_backend.py` 使用 MCP SDK 的**内存 transport** 做真实协议集成测试（无网络、无子进程），覆盖 schema 转换、`isError`→taxonomy、error-hint 参数修复、transient 重试、非瞬态不重试；`test_fault_epoch.py` 固定故障 epoch 语义。当前 **39 tests passed**。

artifact：[`benchmarks/toolguard/mcp_demo.json`](benchmarks/toolguard/mcp_demo.json)

## 架构：一个中间件，三种传输层

中间件只依赖 `ToolBackend` 协议（`schemas()` 返回工具清单，`call(name, arguments)` 返回统一 `CallResult`）：

| 后端 | 用途 |
|---|---|
| `RecordedBackend` | 离线真实轨迹 + 发布方响应缓存复放 |
| `LiveBackend` / `LiveHardBackend` | 真实公开 HTTP API（6 / 12 工具） |
| `MCPBackend` + `MCPBridge` | 真实 MCP stdio server（12 工具） |

因此“接入新工具层”是 adapter 级别的工作，不是重写中间件；MCP demo 里 `ToolGuard` 本体零改动。

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

# 困难多步任务集：先校准无故障轮数，再跑 budgeted + generous 两个条件（约 20 分钟）
python scripts/run_live_demo_hard.py --mode calibrate
python scripts/run_live_demo_hard.py --output benchmarks/toolguard/live_demo_hard.json

# MCP 集成 demo：真实 stdio MCP server 子进程（需要真实管道，不能在工作区沙箱内运行）
python scripts/run_mcp_demo.py --output benchmarks/toolguard/mcp_demo.json

python -m pytest tests/test_toolguard -q -p no:cacheprovider
```

artifact：[`benchmarks/toolguard/metrics.json`](benchmarks/toolguard/metrics.json)（含完整口径、分规则验证与局限）

## 局限与下一步

- 离线 replay：缓存只覆盖参考解输入（最终尝试命中率 51.6%），修复验证存在未命中，**未命中不等于失败**；
- value agreement 只在“某模型解过同一任务”时可验证，是严格下界；
- schema 默认值补全的 30 条 cross-group 修复未经验证，属启发式提议；
- 服务端/瞬态错误（全数据集 96 条）离线不可确定性修复；
- live demo 全部是**单模型、单次运行**（每任务每臂 1 次），统计功效有限；McNemar 只基于配对 discordant 数；
- 故障注入是**确定性构造**，不是真实观测；epoch 长度（2 次尝试）与预算定义共同决定 budgeted 差异；`budgeted` 是受控压力条件，不能解读为“真实生产故障率下 baseline 会失败 60%”；
- guard 在第二轮的唯一失败是答案格式问题（人口被四舍五入），不是中间件失效；artifact 未做事后修正；
- 修复调用的二次瞬态重试缺失：类型修复后的调用若被注入瞬态故障，当前实现不会在同一 call 内继续重试修复后的调用（只重试原始调用），hn 任务因此各浪费 2 轮；
- MCP demo 是单模型、10 任务、单次运行；MCP stdio 需要真实子进程管道（受限沙箱内需放宽权限），测试用 MCP 内存 transport 规避该环境限制；
- 免费公开 API 非生产级：GitHub 限流已从困难任务集移除，API 漂移可能影响复跑；
- 下一步：把修复后的调用也纳入瞬时重试；把 live 中“默认值补全”类修复变成可验证规则；扩到多模型/多次运行以给任务成功率差异置信区间。
