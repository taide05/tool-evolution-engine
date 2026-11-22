# ToolGuard — tool-call reliability middleware for LLM agents

ToolGuard 是一个夹在 Agent 与工具之间的中间件：**采集每次工具调用 → 分类失败 → 从真实失败轨迹中学习修复规则 → 运行时修复并记录指标**。它不训练模型、不改工具实现，只做工具层的确定性修复，因此可解释、低成本、易部署。

当前版本聚焦「离线真实数据评测」：在 **30,120 次真实工具调用 / 1,033 次真实失败**（StableToolBench 公开基线轨迹，Apache-2.0）上验证修复规则。

## 结果（离线确定性评测）

| 协议 | 可评估失败 | 可修复 | 覆盖率 | schema 合法率 | 缓存验证 |
|---|---|---|---|---|---|
| **LOMO（5 折留一模型）** | 969 | 321 | **33.1%** | **100%** | 48 恢复 / 0 反例 / 273 未命中 |
| cross-model（3→2 模型） | 277 | 106 | 38.3% | 100% | 15 恢复 / 0 反例 |
| cross-group（跨任务集） | 76 | 0 | **0%** | — | — |

- 真实失败构成：缺必填参数 **724**、非法取值 98、服务端/瞬态错误 96、其他 85、输入解析 30；
- 修复手段：记忆重放 69、学习式参数补全 45、schema 默认值 10（cross-model 口径）；
- **诚实边界**：实例级记忆在“同任务、其他模型跑成功过”时有效；跨任务集（全新表述/未见过的函数）覆盖率为 0 —— 这是当前方法的 hard limit，也是下一步要解决的问题。

artifact：[`benchmarks/toolguard/metrics.json`](benchmarks/toolguard/metrics.json)

## 工作原理（三级修复）

1. **确定性 schema 规则**：函数名纠错（幻觉函数 → 可用工具中最相似者）、删除/重命名 schema 外参数、按 schema 默认值/示例值补必填参数、类型强制转换；
2. **经验记忆**：同一 task/function 曾在有效轨迹中成功调用过 → 重放那次成功调用的参数组合；
3. **学习式参数补全/归一化**：从成功调用的参数取值分布中取众数，补齐或修正参数值。

所有修复必须通过 **schema 合法性检查**（函数存在、参数都在 schema 内、必填齐全）才计入“可修复”，避免“拍脑袋修复”。

## 数据来源

- 轨迹：StableToolBench baseline trajectory release（Apache-2.0，公开）
- 工具 schema：同一 release 的 solvable_queries `api_list`（2,461 个函数）
- 真实响应缓存：ToolBench Apr-2024 缓存（仅用于修复结果的补充验证）
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

## 局限与下一步

- 当前只有离线 replay，没有 live API 端到端 A/B（下一步：GitHub/Wikipedia/Open-Meteo 等免 key 公开 API + DeepSeek agent）；
- 跨任务泛化为 0，需要更通用的修复策略；
- 缓存只覆盖参考解输入（最终尝试命中率 51.6%），修复验证存在 273 例“未命中”；未命中不等于失败，但无法验证；
- 单模型族粒度统计，未做多 seed（replay 本身确定性）。
