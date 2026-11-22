# ToolGuard — tool-call reliability middleware for LLM agents

ToolGuard 是一个夹在 Agent 与工具之间的中间件：**采集每次工具调用 → 分类失败 → 从真实失败轨迹中学习修复策略 → 运行时修复并记录指标**。它不训练模型、不改工具实现，只做工具层的确定性修复，因此可解释、低成本、易部署。

当前版本聚焦「离线真实数据评测」：在 **30,120 次真实工具调用 / 1,033 次真实失败**（StableToolBench 公开基线轨迹，Apache-2.0）上验证修复策略，并重点做了**跨任务泛化**：把“重放同一任务的旧调用”升级为“从任务文本学习参数抽取规则”。

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

python -m pytest tests/test_toolguard -q -p no:cacheprovider
```

artifact：[`benchmarks/toolguard/metrics.json`](benchmarks/toolguard/metrics.json)（含完整口径、分规则验证与局限）

## 局限与下一步

- 当前只有离线 replay，没有 live API 端到端 A/B（下一步：GitHub/Wikipedia/Open-Meteo 等免 key 公开 API + DeepSeek agent，middleware on/off 对照）；
- 缓存只覆盖参考解输入（最终尝试命中率 51.6%），修复验证存在未命中，**未命中不等于失败**；
- value agreement 只在“某模型解过同一任务”时可验证，是严格下界；
- schema 默认值补全的 30 条 cross-group 修复未经验证，属启发式提议；
- 服务端/瞬态错误（全数据集 96 条）离线不可确定性修复，归为“可重试”类，留给 live 中间件处理。
