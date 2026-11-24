# ToolGuard 审阅指引（REVIEW）

> **给外部审阅者（AI 或人）**：这份文档告诉你这次会话实现/修改了什么、证据在哪里、如何**不依赖作者自述**地核对这些数字、以及哪些结论**不能**采信。若只读三份文件：本文件 → [TOOLGUARD.md](TOOLGUARD.md)（设计与结果）→ `benchmarks/toolguard/*.json`（逐格证据）。

- 分支：`feat/toolguard`（基线 `opt/tee-tools` @ `7fd644c`），审阅范围 = `git diff 7fd644c..HEAD`
- HEAD 以 `git rev-parse HEAD` 为准（本文写作时为 `8c76d4c`，加上审阅包自身 commit）；本地 16 个功能 commit，**未 push**
- 审阅包内容：`REVIEW.md`（本文）+ `TOOLGUARD.md`（正式结果文档）+ 9 个 JSON artifact + 代码/测试
- 作者：AI coding agent（单次会话），所以本文档按"可被反驳"的方式写：每个结论都给了可执行核验方式

---

## 0. 30 秒核验路径（不需要 API key、不联网）

```bash
python -m pytest tests/test_toolguard -q -p no:cacheprovider        # 期望：55 passed
python scripts/validate_toolguard_artifacts.py                       # 期望：9 artifacts, all consistent
python scripts/toolguard_evidence_summary.py                         # 从 artifact 重算所有主数字
```

三条命令的输出就是本文/TOOLGUARD 中所有数字的**机器可复核版本**。如果 `toolguard_evidence_summary.py` 打印的数字与 TOOLGUARD.md 不一致，以 artifact 重算结果为准，并把差异作为 finding 记录。

想看"这次会话改了什么"：

```bash
git log --oneline 7fd644c..HEAD
git diff --stat 725a9e1..HEAD   # 收口轮（V3 多模型/单模型/验证器/契约/文档）的变更集
git show --stat dcd5f20 8ad347e adbb9e5 8c76d4c   # 最近四个关键 commit
```

---

## 1. 审阅范围与不在范围

**本次会话实现的内容**（全部在 `feat/toolguard` 分支的 16 个 commit 里）：

1. **ToolGuard 中间件**：离线（30,120 次真实调用）→ live 端到端（真实公开 API + 真实 LLM agent）→ 真实 MCP（双 stdio server、进程崩溃重连、catalog drift）→ 多模型复制（3 模型）。核心代码在 `src/tool_evolution/toolguard/`。
2. **实验与证据**：4 轮递进实验、9 个 artifact、逐格记录 + 逐条修复事件 + 局限声明、Wilson CI / 配对 McNemar。
3. **收口**：V3 ground-truth 漂移加固、验证器 Unicode 负号修复、公开 API 超时重试、artifact 契约检查、文档口径统一（`README.md` 顶部指引、TOOLGUARD 溯源表）。

**明确不在本次范围**（不要当作本次会话的交付来评）：

- `README.md` 正文与 TEE 旧阶段代码：历史底座，README 中指标是旧口径，顶部已加"当前分支指引"说明；
- `docs/`、`STATUS.md`：作者侧本地工作笔记，`.gitignore` 已忽略、不随仓库提交（若审阅者从 git 获取仓库，看不到它们是预期行为）；
- 任何"生产环境效果"的声明：故障/崩溃/漂移全部是**确定性注入**，不是生产观测。

---

## 2. 仓库地图（审阅时建议的阅读顺序）

| 位置 | 作用 | 审阅要点 |
|---|---|---|
| `TOOLGUARD.md` | 正式结果文档（离线 + 4 轮 live + 架构 + 局限 + 产物溯源表） | 数字是否都能在 artifact 中复算；局限是否诚实 |
| `src/tool_evolution/toolguard/guard.py` | 中间件核心：五级修复阶梯 + `call_with_repair` | 修复是否发生在工具调用边界、是否校验 schema/值、是否真的重试 |
| `.../rules.py`、`extractors.py`、`naming.py` | schema 规则 / 任务文本参数抽取 / 未知工具名相似度解析 | 规则是否数据驱动、阈值来源（0.55 是离线评测选出） |
| `.../agent.py` | live agent（OpenAI-compatible Chat Completions 客户端 + 轮内修复日志） | 真实 LLM 调用、无 mock、usage 统计 |
| `.../live.py`、`live_hard.py` | 真实公开 API backend（6/12 工具）+ 故障注入 epoch + ground truth 构造 | 故障注入是确定性的；ground truth 从同一批真实 API 抓取 |
| `.../verify.py` | 程序化验证（无 LLM judge）：数字/文本/日期、Wilson、McNemar、Unicode 负号归一化 | 答案判定是否可复算；边界处理 |
| `.../mcp_backend.py`、`scripts/mcp_toolguard_server.py` | 真实 MCP stdio adapter + 双 server + 崩溃/drift 注入 | 进程管道真实；middleware 零改动（adapter 隔离） |
| `.../artifact_contracts.py`、`scripts/validate_toolguard_artifacts.py` | artifact 契约（aggregate ↔ records、reproduce/limitations、calibration 对齐） | 9 个 artifact 全过 |
| `tests/test_toolguard/` | 55 个测试 | 覆盖规则/命名/guard/backend/MCP/故障 epoch/统计/契约 |
| `benchmarks/toolguard/*.json` | 9 个 artifact（逐格 records + aggregate + key_findings + limitations + reproduce） | 见 §3 证据地图与 TOOLGUARD 的"产物与复现溯源"表 |

---

## 3. 证据地图：每个主要 claim → 用哪条命令核

所有数字由 `python scripts/toolguard_evidence_summary.py` 从 artifact 重算；下表给出更细的定位。

| Claim（TOOLGUARD 中的主数字） | 证据 artifact | 如何独立核验 |
|---|---|---|
| 离线：30,120 次真实调用 / 1,033 次真实失败；LOMO 修复覆盖 **54.7%**（969 可评估失败）、schema 合法率 **100%**、缓存验证 **48 恢复 / 0 反例** | `metrics.json` | `evidence_summary.py` 的 `offline` 段；或读 `results.lomo_aggregate` 直接用合计值除 |
| 跨任务泛化：cross-group 覆盖 **0% → 75.0%**（76 失败中修 57），学习式文本抽取 **27/27** 值一致；两协议合计文本抽取 **39/39** | `metrics.json` | `results.cross_group`；注意 30 条 schema 默认值补全是**未验证提议**，不当证据 |
| live V1：27 短任务，normal 双方 27/27；stress baseline 25/27 vs guard 26/27（p=1.0）；工具失败 **-80%~-93%**、tokens **-22%** | `live_demo.json` | `results.normal/stress`（per_task 27 条） |
| 单模型 V3：clean+1 baseline **51/93 (54.8%)** vs guard **93/93 (100%)**，McNemar **p=4.55e-13**，tokens **-36.4%**，工具错误 263→5；clean+4 90.3%→100%；10 轮 96.8%→96.8% | `live_demo_hard_v3.json`（310 格） | `evidence_summary.py` 的 `live_demo_hard_v3`；aggregate 与 records 由契约检查强制一致；McNemar 可由 paired_cells 用二项检验复算 |
| 多模型 V3：3 模型真·placement **27/31、31/31、29/31**；clean+1 baseline **45.2%/51.6%/54.8%** → guard **91.9%/100%/100%**（p=1.3e-7/1.9e-9/7.5e-9）；`guard >= baseline` 全模型全条件成立 | `live_demo_multimodel_v3.json`（558 格） | `evidence_summary.py` 的 `live_demo_multimodel_v3`；三模型同任务池/故障 seed/预算语义 |
| MCP V2：双 server 96 格，clean+1 baseline **11/32 (34.4%)** vs guard **32/32 (100%)**（p=9.54e-7）；**17–20 次真实 server 崩溃/重连**全部吸收；`function_name_resolution` 9 次 | `mcp_demo_v2.json` | `guard_repairs_by_rule` 字段；records 中 `faults_injected`/`guard_events` |
| 成本：多模型 V3 评测 **$2.08**（qwen4-max 占 80%）+ calibration $0.30 | `live_demo_multimodel_v3.json` | `total_estimated_cost_usd` + 各臂 usage × 公开价格（价格常量在 `scripts/run_multimodel_live.py` 的 `model_specs`） |

**历史 artifact（不要在跨版本间逐位比较）**：`live_demo_hard.json`（V1 20 任务）、`live_demo_hard_v2.json`（V2 32 任务）、`live_demo_multimodel.json`（多模型 V2，含直播可变 karma 任务）。任务池随版本演进：V1 20 → V2 32 → V3 31（HN karma 换成不可变账号创建年份），TOOLGUARD 的"产物与复现溯源"表逐行写明 artifact → commit。

---

## 4. 审阅方式：哪些能纯离线做，哪些要 key（建议不花钱做）

**建议审阅者做的（免费、无需 key、无需网络）**：

1. 跑 §0 三条命令；
2. 抽查任意一个 claim：用 `evidence_summary.py --json` 输出与 TOOLGUARD.md 对照；
3. 代码审阅（每项都可 grep 验证）：
   - `guard.py::call_with_repair`：修复是否在**同一个工具调用**内重试、是否校验 schema、是否记录 `guard_events`；
   - `verify.py`：答案判定是否纯程序化（无 LLM judge）、Unicode 负号/负温度是否处理（本会话修过）；
   - `live_hard.py`：故障注入是否**确定性**（blake2b seed → epoch，前 N 次尝试失败）、ground truth 是否从同一 API 抓取；
   - `mcp_backend.py`：进程崩溃重连、`_TRANSIENT` 是否包含 transport 错误；
   - `artifact_contracts.py`：aggregate 是否真由 records 复核（而不是直接信 aggregate）；
   - `run_multimodel_live.py::model_specs`：三个模型 id / 价格 / 上下文行为；`--mode eval` 的 per-model ground-truth 重建。
4. 用 `git diff 7fd644c..HEAD` 检查"是否有一处代码把失败偷偷吞掉/放宽验证"。

**需要 key/网络的（不建议审阅时跑，除非想复现）**：live demo 的重跑（DeepSeek + 阿里百炼 key；单模型 310 格约 20 分钟、约 1–2 元；多模型 558 格约 1 小时、约 $2.4）。`scripts/check_providers.py --smoke` 可做一次极小的 key/工具调用冒烟。**不要向作者索要 `.env`**：密钥永不进仓库，`.env.example` 只有占位符。

---

## 5. 已知局限与"不能声称"清单（审阅重点压测处）

1. **故障/崩溃/漂移是确定性注入**，不是生产观测。`clean+1` 是"无故障需求 +1"的受控压力条件，不能读作"生产失败率"；正确口径是成功-预算曲线。
2. **天气/FX 等 ground truth 是直播值**：V3 换了不可变 HN 任务 + 每模型重建 GT；仍有极小漂移（单模型 V3 只剩 1 个模型格式失败）。V3 artifact 的 `expected` 快照可逐条核对。
3. **每任务只做 1 次无故障 calibration**：预算锚点本身是点估计。
4. **多模型是复制而非模型间假设检验**：只 3 个模型；cheap/strong 同属 Qwen 家族；"越弱收益越大"是描述性结论，不是统计检验。
5. **两个强模型紧预算 guard 已 100%**（ceiling effect）：更高预算下差异缩小，提升空间要靠更难任务/更紧预算。
6. **离线 LOMO 的"修复覆盖"不等于端到端成功**：value agreement 只在"其他模型解过同任务"时能验证（严格下界）；30 条 schema 默认值补全未经验证。
7. **MCP V2 artifact 的任务池是 V2 时代**（16 任务子集），与 V3 单模型/多模型数字不同源；复现见 commit `a1f8b23`。
8. **验证器修复的时间差**：多模型 V3 artifact 产生于 Unicode 负号修复之前，离线核对 12 条含 `−` 的回答均已是成功，成功率不受影响；单模型 V3 用修复后重跑（这是"为什么 guard clean+1 是 100%"的部分原因，不是挑数据）。
9. **免费公开 API 非生产级**：GitHub 限流已从任务集移除；API 漂移/抖动可能影响复跑（已加有界重试）。
10. **没有 push、没有 CI**：所有验证在单机完成；审阅者若复跑数字会有随机波动（LLM 非确定性、live 数据变化），允许 ±1–2 格差异，但不能出现方向性反转。

---

## 6. 本次会话 commit 清单（`7fd644c..HEAD`，从旧到新）

| commit | 内容 |
|---|---|
| `15baa59` | 离线修复中间件 + 30k 真实调用评测（metrics.json） |
| `6f8b300` | 跨任务泛化：学习式任务文本参数抽取（cross-group 0%→75%） |
| `b0564ae` | live V1：6 个公开 API / 27 短任务（live_demo.json） |
| `02ac71a` | 困难多步任务集 V1（20 任务/单预算） |
| `b1d7609` | 真实 MCP stdio 集成 V1（10 任务，middleware 零改动） |
| `40d55d3` | TOOLGUARD.md V1 结果 |
| `13af709` | 修复"修复后的调用再遇瞬态故障不重试"缺陷 |
| `43c4e89` | V2：32 任务 × 3 档预算 × 重复（320 格） |
| `a1f8b23` | V2 MCP：双 server + 真实崩溃重连 + catalog drift（96 格） |
| `725a9e1` | TOOLGUARD.md V2 数字与预算曲线口径 |
| `e1c391e` | 多 provider 配置 + OpenAI-compatible client + provider 冒烟门禁 |
| `fb13a1f` | 多模型复制 V2（3 模型 × 576 格，含直播 karma 任务） |
| `8ad347e` | V3 ground-truth 稳定化：HN karma→不可变创建年份 + 每模型重建 GT |
| `dcd5f20` | V3 多模型 artifact（558 格） |
| `adbb9e5` | 收口修复：验证器 Unicode 负号、API 超时重试、artifact 契约 + 测试 |
| `8c76d4c` | 收口：V3 单模型 artifact（310 格）+ README/TOOLGUARD 溯源与一致性修订 |

**审阅范围里最值得逐行看的 diff**：`git show 8ad347e`（GT 稳定化）、`git show adbb9e5`（验证器/契约）、`git show 8c76d4c -- TOOLGUARD.md`（口径修订）、`git show dcd5f20 -- scripts/run_multimodel_live.py`（多模型协议与 per-model GT refresh）。

---

## 7. 审阅 checklist（可勾选）

- [ ] `pytest` 55 passed；`validate_toolguard_artifacts.py` 9 artifacts 全过
- [ ] `toolguard_evidence_summary.py` 输出与 TOOLGUARD.md 主数字一致
- [ ] `guard.py` 的修复确实发生在工具调用边界，且 schema 校验不是"永远 True"
- [ ] 每个 live 实验的 failure 都能在 records 里找到对应记录（不是 aggregate 单独编造）
- [ ] 故障注入确定性：同 seed 同任务可复现同一 failure 序列（见 `test_fault_epoch.py`）
- [ ] 验证器无 LLM judge（grep `judge`/`gpt` 在验证路径）
- [ ] 多模型实验确实 3 个不同模型 id、同一任务池/seed/预算
- [ ] V3 任务池没有直播可变 ground truth（HN 只有创建年份；`live_hard.py` 中查 `created`）
- [ ] TOOLGUARD 的"产物与复现溯源"表与 `git log -- benchmarks/...` 一致
- [ ] 局限清单是否覆盖了 §5 的 10 条
- [ ] 报告一个具体 finding（如有），并注明证据文件 + 行号

---

## 8. 安全与边界

- `.env`（DeepSeek/百炼 key）**永不提交**、`.env.example` 只有占位符；审阅无需 key；
- `docs/`、`STATUS.md` 是作者本地笔记（gitignored），不随仓库分发；若审阅者看到"本地未提交"的提法，指的就是它们；
- 本会话所有 API 消耗由作者承担（多模型 + 收口重跑合计约 $4.9 + 6 元人民币量级），审阅者不必复跑；
- 若审阅者发现"文档数字与 artifact 不一致"，优先修复文档并记录 finding，而不是改 artifact。
