# ToolGuard 审阅指引（REVIEW）

> **给外部审阅者（AI 或人）**：这份文档告诉你交付了什么、证据在哪里、如何**不依赖作者自述**地核对这些数字、以及哪些结论**不能**采信。若只读三份文件：本文件 → [TOOLGUARD.md](TOOLGUARD.md)（设计与结果）→ `benchmarks/toolguard/*.json`（逐格证据）。
>
> **2026-11-24 证据链修复轮 + 2026-11-25 HN 契约修正（单/多模型 V4，已提交并 push 到 `origin/feat/toolguard`，最新 `af97244`）**：针对外部审阅的 3 个 P0 + P1/P2，本轮只修证据链、不加功能：证据汇总从 records 重算全部派生字段并加篡改负例、完整 JSON Schema 校验 + schema 漏斗、比较类 verifier 改为关系断言 + 离线重判、离线 metrics 改为确定性生成 + LOMO 协议如实标注、P2 文档/复现修复；随后按评审决定修正 HN schema 契约并重跑出**单/多模型 V4 主口径**，另做 schema 索引保真度修复与真实线上故障 trace 证据（见 §3/§5）。

- 分支：`feat/toolguard`（基线 `opt/tee-tools` @ `7fd644c`）；本轮修复已 commit 并 push：`origin/feat/toolguard` @ `af97244`（`3e35e47` 之后）
- 工作区状态：干净；本文件描述的是“本轮修复 commit”这一审阅状态（之前的 `3e35e47` 是被审阅的交付基线）
- 审阅包内容：`REVIEW.md`（本文）+ `TOOLGUARD.md`（正式结果文档）+ 13 个 JSON artifact + 代码/测试
- 作者：AI coding agent（单次会话），所以本文档按“可被反驳”的方式写：每个结论都给了可执行核验方式

---

## 0. 30 秒核验路径（不需要 API key、不联网）

```bash
python -m pytest tests/test_toolguard -q -p no:cacheprovider   # 期望：128 passed（全仓 437 passed）
python scripts/validate_toolguard_artifacts.py                # 期望：13 artifacts, all consistent
python scripts/toolguard_evidence_summary.py                  # 所有派生数字从 records 重算
python scripts/rescore_toolguard_verifier.py                  # 期望：756 条比较记录，611→611，0 变化
```

这四条命令的输出就是本文/TOOLGUARD 中所有数字的**机器可复核版本**。

- `validate_toolguard_artifacts.py` 不再只查 cells/success：success/rate/Wilson/turns/tokens/tool_errors/guard_recoveries/McNemar p/failure modes/多模型 placement 等**全部从逐条 records 重算**；`metrics.json` 的比率、schema 漏斗、cache 求和、skipped_no_schema 对账由同一模块复算。篡改任意这些字段的负例见 `tests/test_toolguard/test_artifact_tamper.py`。
- `toolguard_evidence_summary.py` 输出中 `live_*` / `mcp_*` 段是 records 重算值（不读 aggregate 的这些字段），`offline` 段是离线 artifact 的 counts + 重算比率（StableToolBench 原始数据不随仓库提交，无法进一步重算计数）。
- 如果输出与 TOOLGUARD.md 不一致，以重算结果为准，并把差异作为 finding 记录。

想看“本轮修复改了什么”：

```bash
git log --oneline -3         # 本轮证据链修复 commit（已 push 到 origin/feat/toolguard）
git show --stat HEAD         # 最近一次修复的内容
git show -- scripts/ src/tool_evolution/toolguard/ tests/test_toolguard/
```

---

## 1. 审阅范围与不在范围

**交付内容**（`feat/toolguard` 分支；16 个功能 commit + 本轮修复 commit `3e35e47..ea6a038`，已 push）：

1. **ToolGuard 中间件**：离线（30,120 次记录调用）→ live 端到端（真实公开 API + 真实 LLM agent）→ 真实 MCP（双 stdio server、进程崩溃重连、catalog drift）→ 多模型复制（3 模型）。核心代码在 `src/tool_evolution/toolguard/`。
2. **实验与证据**：4 轮 live 实验、13 个 contract artifact、逐格记录 + 逐条修复事件 + 局限声明、Wilson CI / 配对 McNemar；离线修复评测带完整 schema 漏斗与定量协议说明。
3. **证据链修复（本轮）**：records 重算 + 篡改负例、完整 schema 校验、关系型比较 verifier、确定性 metrics 生成器、LOMO 协议如实化、P1/P2 口径修正；单/多模型 HN 契约修正并重跑 V4 主口径；schema 索引保真度修复（4 类型 + 20 参数）；真实线上故障 trace 证据 + real-trace 故障源；V5 hard-task ceiling pilot（负结果）。
4. **额外证据数据（不是第 14 个 contract artifact）**：`benchmarks/toolguard/real_traces/` 下 4 个 JSON（真实 trace、repair 结果、real-trace A/B smoke、V5 hard pilot），由 `collect_real_fault_traces.py` / `run_real_fault_repair.py` / live runner 生成，有测试固定内容。

**明确不在本次范围**：

- 任何 live 实验的**重跑**：V4 重跑不计入此项（见 §3）；本轮的 schema 索引修正只影响离线 `metrics.json`，由 `scripts/build_metrics_artifact.py` 从 records/评测输出确定性重建；
- 旧 TEE 25 条：不属本轮范围，未改动；
- `README.md` 正文的旧 TEE 指标：历史口径，顶部已有说明；
- 任何“生产环境效果”的声明：故障/崩溃/漂移全部是**确定性注入**，不是生产观测。

---

## 2. 仓库地图（审阅时建议的阅读顺序）

| 位置 | 作用 | 审阅要点 |
|---|---|---|
| `TOOLGUARD.md` | 正式结果文档（离线 + 4 轮 live + 架构 + 局限 + 产物溯源表） | 数字是否都能在 artifact/records 中复算；局限是否诚实 |
| `src/tool_evolution/toolguard/schema_check.py` | 完整结构校验器（类型/required/额外参数/enum/format/范围/数组） | 是否真的校验而不是只查参数名 |
| `src/tool_evolution/toolguard/engine.py` | 修复阶梯 + **两层证据**：先 schema 校验（含 `schema_type_coercion` 类型转换修复），schema 不通过的候选记录在 `candidates`，若逐字命中同 task/function 真实成功调用则按 `recorded_success_match` 接受（value-verified） | 被拒提议是否被记录而不是静默丢弃；schema_valid_rate 是否确实 <100%；value-verified 是否单独报告 |
| `src/tool_evolution/toolguard/guard.py` | live 中间件：重试/coercion/name resolution；**发 backend 前做 schema 校验** | 被拒修复是否记录 `schema_errors`；backend error-hint 的 type override 是否被披露 |
| `src/tool_evolution/toolguard/artifact_metrics.py` | **所有 artifact 派生字段的 records 重算**（success/rate/Wilson/turns/tokens/p/placement/cost） | 是否直接读 aggregate 的派生字段（应读 records） |
| `src/tool_evolution/toolguard/artifact_contracts.py` | artifact 契约：逐字段重算比对 + metrics 导出量对账 | 13 个 artifact 全过；篡改字段是否会被抓 |
| `src/tool_evolution/toolguard/verify.py` | 程序化验证：关系断言、否定词、million 等单位、版本比较 | 反向答案是否判失败（见 `tests/test_toolguard/test_relation_verifier.py`） |
| `scripts/toolguard_evidence_summary.py` | 审阅者核验路径：records 重算输出 | 是否打印 records 值而非 artifact aggregate 值 |
| `scripts/build_metrics_artifact.py` | 离线 `metrics.json` 的确定性生成器 | 文档命令能否重建 artifact |
| `scripts/rescore_toolguard_verifier.py` | 用新 verifier 只读重判已提交答案 | success 变化清单（756 条，0 变化） |
| `scripts/collect_real_fault_traces.py` / `run_real_fault_repair.py` | 真实线上错误 trace 采集 + repair-layer 在线验证 | 16 条真实错误、4/4 可推导契约错误恢复、12 条语义错误诚实失败 |
| `src/tool_evolution/toolguard/{schemas,live}.py` | schema 索引保真度修正 + `RealTraceFaultBackend` 故障源 | 修正是否有证据/留痕；real-trace 注入是否确定性、两臂一致 |
| `tests/test_toolguard/` | 128 个测试 | 规则/命名/guard/backend/MCP/故障 epoch/统计/契约/篡改负例/schema 负例/关系验证器/HN schema 模式/schema 索引保真度/real-trace |
| `benchmarks/toolguard/*.json` | 13 个 contract artifact（单/多模型 V4 主口径、HN 归因对照、V3 drift 演示、MCP v3） | 见 §3 证据地图与 TOOLGUARD 的“产物与复现溯源”表 |
| `benchmarks/toolguard/real_traces/*.json` | 真实 trace / repair / A/B smoke / V5 hard pilot（证据数据，非 contract artifact） | 对应脚本重跑可复现；测试固定关键结论 |

---

## 3. 证据地图：每个主要 claim → 用哪条命令核

| Claim（TOOLGUARD 中的主数字） | 证据 artifact | 如何独立核验 |
|---|---|---|
| 离线数据：30,120 次评估调用（7,636 trace；764 唯一 query、2,303 唯一 (query, tool)）/ 1,033 次失败；64 条 skipped_no_schema = 30 raw + 34 无 schema | `metrics.json` | `results.dataset`、`results.skipped_no_schema*`；`evidence_summary.py` 的 `offline` 段 |
| 离线修复（两层证据 + schema 索引保真度修复，`metrics.json` v4）：LOMO **accepted 576/969 = 59.4%**（提议 583；schema 通过 539=92.5%，其中 131 条经类型 coercion；schema 拒绝失败 44；value-verified 37；真 LOMO 敏感性 597/969=61.6%）；cross-model **173/277 = 62.5%**；cross-group **57/76 = 75.0%**（schema 修正后 57 条全部 schema-valid） | `metrics.json` | `results.*`：`proposed = schema_passed + schema_rejected_failures`；`accepted = schema_passed + value_verified_repairs`；`schema_valid_rate = schema_passed/proposed`；`coverage = accepted/evaluable_failures`。三数均可手算 |
| **schema 索引保真度修复（本轮新增）**：4 处类型修正（watchmode `start_date/end_date` number→string；两个 `page` string→integer）+ 20 个空参数列表回填；规则保守（≥3 成功样本/≥90% 一致/不改 required/不做 integer↔number 拓宽），每处修正留痕 | `metrics.json` | `dataset.schema_index.fidelity_repair`（type_correction_examples / param_addition_examples）、`protocol.schema_fidelity`；重跑 `eval_toolguard_rules.py` + `build_metrics_artifact.py` 可复算 |
| LOMO 协议诚实性：主口径是“固定三模型训练 universe、逐模型留出”，gpt-4 家族从不训练；真 LOMO 敏感性 **597/969=61.6%** | `metrics.json` | `protocol.lomo`、`results.lomo_per_fold[*].train_models_used`、`results.lomo_sensitivity_exclude_fold` |
| schema/修复的两层证据：schema-valid（含 coercion，单独计数）+ value-verified（与同 task/function 真实成功调用逐字一致）；声明 schema 的保真度缺陷已按记录证据修正，剩余缺陷仍如实报告 | `metrics.json` + `engine.py`/`schema_check.py`/`schemas.py` | `results.*.{schema_passed, coerced_repairs, value_verified_repairs, schema_rejected_proposals}`；`protocol.schema_funnel`/`schema_fidelity`；复算脚本 `.scratch/evidence/analyze_rejection_reasons.py`、`recoverability_probe.py` |
| live V1：27 短任务，normal 双方 27/27；stress 25/27 vs 26/27；工具失败 -80%~-93%、tokens -22% | `live_demo.json` | `results.normal/stress`（per_task 27 条） |
| **单模型 V4（当前主口径，HN 契约修正后）**：clean+1 baseline **59/93 (63.4%)** vs guard **93/93 (100%)**，McNemar **p=1.16e-10**（配对 `guard_only=34 / baseline_only=0`），tokens -33.3%；clean+4 双方 31/31；10 轮 31/31 vs 30/31；注入 epoch baseline 365 / guard 382，guard 轮内恢复 191、agent 可见工具错误 216→0 | `live_demo_hard_v4.json`（310 格） | `evidence_summary.py` 的 `live_demo_hard_v4`；aggregate 与 records 由契约强制一致；预算锚点复用 V3 calibration（`protocol.calibration.source`），V3→V4 只改 HN schema |
| 单模型 V3（**历史**，HN schema drift 契约修复能力演示）：clean+1 baseline 51/93 (54.8%) vs guard 93/93，p=4.55e-13，tokens -36.4%；clean+4 90.3%→100%；10 轮 96.8%→96.8%；epoch baseline 382 / guard 400；guard 用 backend error-hint coercion 使 HN 子集 12/12 | `live_demo_hard_v3.json`（310 格） | 与 V4 对比 HN/非 HN 子集：HN 子集 baseline 2/12→12/12；本轮 `--from-records` 重建过 failure_mode/reproduce，**数字未变** |
| **HN 归因对照（促成 V4 修复）**：同 V3 calibration 下，仅把 `get_hn_item.story_id` 改成 integer → baseline clean+1 **51/93→61/93**（HN 子集 2/12→12/12；非 HN 49/81 逐格相同，证明因果） | `live_demo_hard_v3_hnconsistent.json`（186 格） | `--budgets clean+1 --repeats clean+1:3 --hn-schema consistent --calibration .scratch/toolguard/hard_calibration_v3.json`；对比 records 的 HN/非 HN 子集 |
| **MCP baseline/guard 分开校准（本轮新增，P1-2）**：两臂各测无故障轮数，预算锚定 baseline（`budget_anchor=baseline`）；旧 V2 用 guard 校准 | `mcp_demo_v3.json`（96 格） | `protocol.budget_anchor`、`calibration.tasks[*].baseline_turns/guard_turns`；`--budget-anchor baseline` |
| 多模型 **V4（当前主口径，HN 契约一致）**：placement 27/31、31/31、29/31；clean+1 baseline **39/62 (62.9%) / 40/62 (64.5%) / 43/62 (69.4%)** → guard **62/62 全模型（100%）**（p=2.4e-7/4.8e-7/3.8e-6）；clean+4 29–31/31 → 31/31；tokens -32%/-33%/-31%；agent 工具错误 133/136/134→0 | `live_demo_multimodel_v4.json`（558 格） | `evidence_summary.py`；`tasks_per_model` 与 records `expected` 由契约比对；`protocol.hn_schema=consistent`、`protocol.calibration.source` 指向复用 V3 per-model 锚点 |
| 多模型 V3（历史，HN drift 演示）：clean+1 baseline 28/62 (45.2%) / 32/62 (51.6%) / 34/62 (54.8%) → guard 57/62 (91.9%) / 62/62 / 62/62；HN 子集 baseline 1/8（三模型） | `live_demo_multimodel_v3.json`（558 格） | 与 V4 对比 HN/非 HN 子集；本轮 `--from-records` 重建 failure_mode/tasks_per_model，成功率未变 |
| MCP V2：双 server 96 格，clean+1 baseline 11/32 vs guard 32/32（p=9.54e-7）；17–20 次真实崩溃/重连 | `mcp_demo_v2.json` | `guard_repairs_by_rule`（从 guard_events 重算）；records 的 `mcp_restarts` |
| MCP V3（本轮重跑，P1-2 修正）：两臂分开校准、预算锚定 baseline；clean+1 baseline **13/32（40.6%）** vs guard **32/32（100%）**（p=3.81e-6），clean+4 14/16 vs 16/16；$0.098/667s；artifact 用 `--from-records` 重建生成元数据（records/aggregate 未变） | `mcp_demo_v3.json` | `protocol.budget_anchor=baseline`、`calibration.tasks[*].baseline_turns/guard_turns`；`guard_repairs_by_rule` 从 guard_events 重算；`--from-records` 可重建 |
| **真实线上故障 trace（本轮新增）**：对 6 类真实公开 API 构造 16 个真实错误场景并捕获真实错误（invalid_argument 6 / not_found 9 / request_error 1）；逐条回放给 guard：**4/4 可推导契约错误修复且在线验证成功**（HN story_id string→int、HN story_id integral-float→int、weather 坐标 numeric-string→float、World Bank year string→int），12 条语义/越界错误诚实失败（0 伪造修复）；live API 抖动单独记为 `verification_status=api_unavailable`；另已接进 A/B（`--fault-source real-trace`，12 格 smoke：baseline=guard=5/6，注入 7 条真实 trace）；成本 $0（smoke $0.018） | `benchmarks/toolguard/real_traces/`（traces / repair run / ab smoke） | 重跑 `collect_real_fault_traces.py && run_real_fault_repair.py`；A/B smoke 用 `run_live_demo_hard_v2.py --fault-source real-trace --task-ids ... --budgets clean+1 --repeats clean+1:1`；`test_real_fault_traces.py` + `test_real_trace_backend.py` 固定证据与后端行为 |
| **V5 更难任务 pilot（本轮新增）**：4 个 opt-in hard tasks（`--include-hard-tasks`，默认 V4 pool 不变），fault-free calibration 后跑 clean+1×3 repeats×2 arms = 24 格：baseline **10/12（83.3%）** vs guard **12/12（100%）**，guard 仍 ceiling（每格 1–5 次轮内修复、turns 3 vs 5–11）；成本 $0.039/234s。负结果：单纯加长链/算术不破 ceiling（难度仍是故障驱动，guard 在轮内吸收） | `benchmarks/toolguard/real_traces/hard_pilot_v5.json` | `python scripts/run_live_demo_hard_v2.py --include-hard-tasks --task-ids hard_gdp_triple_gap,hard_weather_spread_three,hard_pypi_release_span,hard_hn_author_age_gap --budgets clean+1 --repeats clean+1:3 --calibration .scratch/toolguard/hard_calibration_v5.json` |
| 比较类 verifier：反向答案判失败；已提交数据重判 success 变化 0/756（含单/多模型 V4） | `verify.py` + `rescore_toolguard_verifier.py` | 跑 rescore；负例见 `test_relation_verifier.py`；重判只读、不改 artifact |
| 契约可自证：任意改 tokens/rate/turns/tool_errors/Wilson/p 都失败 | `artifact_contracts.py` + `test_artifact_tamper.py` | `pytest tests/test_toolguard/test_artifact_tamper.py`（临时副本篡改，**13 个 artifact** 复制后逐字段改） |
| 成本：多模型 V4 评测 **$2.00**（558 格；复用 V3 per-model calibration，无额外校准花费）；V3 历史版 $2.08 + $0.30 | `live_demo_multimodel_v4.json` | 由 records usage × `protocol.models` 公开单价可重算（`artifact_metrics.recompute_multimodel`） |

**历史 artifact（不要在跨版本间逐位比较）**：`live_demo_hard.json`（V1 20 任务）、`live_demo_hard_v2.json`（V2 32 任务）、`live_demo_hard_v3.json`（V3 HN drift 演示）、`live_demo_multimodel.json`（多模型 V2，含直播可变 karma 任务）。任务池随版本演进：V1 20 → V2 32 → V3/V4 31（V4 与 V3 同任务池，只修 HN schema）。

---

## 4. 审阅方式：哪些能纯离线做，哪些要 key（建议不花钱做）

**建议审阅者做的（免费、无需 key、无需网络）**：

1. 跑 §0 四条命令；
2. 抽查任意 claim：`evidence_summary.py --json` 与 TOOLGUARD.md 对照；篡改一个 aggregate 字段看 validator 是否报错（或直接跑 `test_artifact_tamper.py`）；
3. 代码审阅（每项都可 grep 验证）：
   - `artifact_metrics.py`：派生数字是否只从 records 重算；
   - `schema_check.py` + `engine.py`：完整 schema 校验与“提议漏斗”是否真实（被拒提议计数而非静默丢弃）；
   - `guard.py`：修复后发 backend 前是否真的校验、被拒是否记录 `schema_errors`、HN type override 是否被披露；
   - `verify.py`：反向答案、否定词、million 单位、截断的处理；无 LLM judge；
   - `build_metrics_artifact.py`：metrics.json 能否由文档命令确定性重建；
   - `live_hard.py`：故障注入是否确定性（blake2b seed → epoch，前 N 次尝试失败）。
4. `git diff` 检查“是否有一处代码把失败偷偷吞掉/放宽验证”。

**需要 key/网络的（不建议审阅时跑）**：live demo 重跑（DeepSeek + 阿里百炼 key；单模型 310 格约 20 分钟、约 1–2 元；多模型 558 格约 1 小时、约 $2.4）。`scripts/check_providers.py --smoke` 可做一次极小的 key/工具调用冒烟（重跑 live 前必须先跑；若 402/余额不足应停止重试）。**不要向作者索要 `.env`**：密钥永不进仓库，`.env.example` 只有占位符。

---

## 5. 已知局限与“不能声称”清单（审阅重点压测处）

1. **离线覆盖率采用两层证据；声明 schema 的保真度缺陷已按记录证据修正（2026-11-25）**：`schema_valid_rate` 现在是 92.5%（LOMO，含 131 条 coercion）/ 85.0%（cross-model）/ 100%（cross-group）；accepted 覆盖 = schema-valid + value-verified。上游 `api_list` 与记录成功调用矛盾处（watchmode 日期参数声明 number、两个 page 声明 string、20 个空参数列表）已在 schema 索引层修正并逐条留痕（≥3 成功样本、≥90% 一致，不改 required、不做 integer↔number 拓宽）；cross-group 的 27 条文本抽取修复因此从 value-verified 升级为 schema-valid。**不能**把 54.8% / 75.0% 说成“schema 合法覆盖率”，也不能把修复前的 81.0% 当现状。
2. **LOMO 名不副实已如实标注**：主口径固定三模型训练 universe，gpt-4 家族只做测试；真 LOMO 敏感性 **61.6%**。
3. **故障/崩溃/漂移是确定性注入**，不是生产观测。`clean+1` 是受控压力条件，不能读作“生产失败率”；正确口径是成功-预算曲线。**补充证据**：另有真实线上错误 trace 的 repair-layer 验证（16 条真实错误：4/4 可推导契约错误修复并在线验证，12 条语义错误诚实失败；见 §3），但它不替代 A/B 的配对因果。
4. **天气/FX ground truth 是直播值**：V3 换了不可变 HN 任务 + 每模型重建 GT；仍有极小漂移。
5. **预算锚点**：单模型 V4 复用 V3 calibration、多模型 V4 复用 V3 per-model calibration（同一预算锚点，schema 是唯一变量；`protocol.calibration.source` 已记录），不是独立新校准；MCP 已改为两臂分别校准并锚定 baseline（`mcp_demo_v3.json`，clean+1 baseline 13/32 vs guard 32/32，p=3.8e-6），旧 V2 的 guard 锚定保留为历史。
6. **多模型是复制而非模型间假设检验**：3 个模型；cheap/strong 同属 Qwen 家族；**“能力越弱收益越大”在 V4 下方向成立**：clean+1 增益 qwen3.8-flash **+37.1pp** > DeepSeek-Flash **+35.5pp** > qwen4-max **+30.6pp**（去掉 HN 后 +42.6/+40.7/+35.2pp，同样单调）；但三个模型紧预算 guard 都到 100%（ceiling effect），该结论只是方向性。V3 的“mid 最大”，部分来自当时 HN 合成契约 bug 对不同模型的非对称影响，V4 已消除。多模型 V4 与单模型一样已修正 HN 契约（HN 子集 baseline clean+1 1/8→8/8，三个模型）。
7. **三个模型的紧预算 guard 都到 100%**（ceiling effect）；V4 多模型 clean+1/clean+4 guard 均无失败；**V5 硬任务 pilot 也没有打破**（baseline 10/12 vs guard 12/12）。要打破 ceiling 需要“无故障也会失败的推理难度 / 超过 guard 重试预算的故障 / 更高语义复杂度”，单纯加长链无效（负结果已记录）。
8. **离线修复覆盖 ≠ 端到端成功**：value agreement 只在“其他模型解过同任务”时可验证（严格下界）；schema 默认值补全是未验证提议。
9. **验证器**：比较类已改关系断言，离线重判 756 条（含单/多模型 V4）0 变化；旧成功率只能视为上界。V1 两个 artifact 无逐条 answer，无法重判。
10. **失败模式标签**：旧 `classify_failure()` 会被 metadata 标量误触发；本轮已用 `--from-records` 从 checkpoint 免费重建两个 V3 artifact 并重算标签（多模型 format_miss 20→3、no_expected_value 2→19；单模型无变化），reproduce 路径同修复（V4 新 artifact 生成时即用修正后的分类器）；旧 V2 artifact 标签仍为历史分类。
11. **MCP V2 任务池是 V2 时代**（16 任务子集），与 V3 单模型/多模型数字不同源。
12. **解析器**把 20 条无法解析的 observation 当成功（0.07%）；30,120 是记录调用/重试次数，不是独立任务数。
13. **免费公开 API 非生产级**：GitHub 限流已从任务集移除；API 漂移可能影响复跑。
14. **本轮修复已 commit 并 push（无 CI）**：V4 是修正后的主口径，但 live 数字仍有 LLM 采样随机性（单模型复跑 ±2 格级波动，非 HN 子集在对照实验与 V4 完整重跑间出现 2 格差异是正常的）；审阅者复跑允许 ±1–2 格差异，但不能出现方向性反转。

---

## 6. 变更清单

### HEAD 之前的功能 commit（`7fd644c..3e35e47`，从旧到新）

| commit | 内容 |
|---|---|
| `15baa59` | 离线修复中间件 + 30k 真实调用评测（metrics.json） |
| `6f8b300` | 跨任务泛化：学习式任务文本抽取 |
| `b0564ae` | live V1：6 个公开 API / 27 短任务（live_demo.json） |
| `02ac71a` | 困难多步任务集 V1（20 任务/单预算） |
| `b1d7609` | 真实 MCP stdio 集成 V1（10 任务，middleware 零改动） |
| `40d55d3` | TOOLGUARD.md V1 结果 |
| `13af709` | 修复“修复后的调用再遇瞬态故障不重试”缺陷 |
| `43c4e89` | V2：32 任务 × 3 档预算 × 重复（320 格） |
| `a1f8b23` | V2 MCP：双 server + 真实崩溃重连 + catalog drift（96 格） |
| `725a9e1` | TOOLGUARD.md V2 数字与预算曲线口径 |
| `e1c391e` | 多 provider 配置 + OpenAI-compatible client + provider 冒烟门禁 |
| `fb13a1f` | 多模型复制 V2（3 模型 × 576 格，含直播 karma 任务） |
| `8ad347e` | V3 ground-truth 稳定化：HN karma→不可变创建年份 + 每模型重建 GT |
| `dcd5f20` | V3 多模型 artifact（558 格） |
| `adbb9e5` | 收口修复：验证器 Unicode 负号、API 超时重试、artifact 契约 + 测试 |
| `8c76d4c` | 收口：V3 单模型 artifact（310 格）+ README/TOOLGUARD 溯源与一致性修订 |
| `3e35e47` | REVIEW.md 审阅指引 + evidence summary 脚本 |

### 本轮修复 commit（`3e35e47..ea6a038`，已 push 到 `origin/feat/toolguard`）

| commit | 内容 |
|---|---|
| `824e9e3` | 证据链修复：records 重算 + 契约/篡改负例；严格 schema 校验 + 两层证据；关系 verifier + 离线重判；确定性 metrics 生成；HN 对照/MCP 分开校准 artifact |
| `828096a` | 单模型 HN 契约修正（V4 310 格重跑） |
| `8c36541` | rescore 计数文档刷新 |
| `eee2c41` | 多模型 HN 契约修正（V4 558 格重跑） |
| `403b95d` | schema 索引保真度修复（4 类型 + 20 参数，离线指标重算） |
| `eeaa4e4` | 真实线上故障 trace 采集 + repair-layer 在线验证 |
| `3cb677a` | integral-float→integer coercion（真实 trace 暴露的缺口） |
| `af97244` | real-trace 故障源接入 A/B + 12 格 smoke |
| `ea6a038` | V5 hard-task ceiling pilot（opt-in，负结果） |

### 本轮变更文件汇总

| 文件 | 改动 |
|---|---|
| `src/tool_evolution/toolguard/artifact_metrics.py`（新） | 所有 artifact 派生字段的 records 重算（rate/Wilson/turns/tokens/tool_errors/p/placement/cost）+ 离线 metrics 导出量对账 |
| `src/tool_evolution/toolguard/artifact_contracts.py` | 逐字段重算比对、metrics 漏斗/对账、tasks_per_model↔record expected 一致性、负例可捕获 |
| `scripts/toolguard_evidence_summary.py` | 重写：records 重算所有 live 指标；offline 段为 counts + 重算比率 |
| `scripts/eval_toolguard_rules.py` | schema 漏斗统计（proposed/rejected/passed）、LOMO 训练策略可选（`--lomo-train-policy`）、诚实协议文本 |
| `scripts/build_metrics_artifact.py`（新） | 离线 metrics.json 确定性生成器（含 skipped_no_schema 解释、真 LOMO 敏感性、schema 索引保真度修正留痕） |
| `src/tool_evolution/toolguard/schemas.py` | **schema 索引保真度修正**：`refine_schema_index()` 按记录成功调用做保守类型修正（4 处）与空参数回填（20 个），统计写入 artifact |
| `src/tool_evolution/toolguard/{guard,engine}.py` | integral-float→integer 的无损 coercion（真实 trace 暴露的缺口）：`8863.0→8863` 同时接入 live guard 的 error-hint/schema coercion 与 engine；离线 metrics 重算逐字节未变 |
| `benchmarks/toolguard/metrics.json` | 由上述生成器重跑生成（本轮被改写的 artifact 之一） |
| `src/tool_evolution/toolguard/schema_check.py`（新） | 完整 JSON-Schema 子集校验（类型/enum/format/范围/数组/嵌套） |
| `src/tool_evolution/toolguard/{memory,rules,engine}.py` | 接入严格校验；RepairResult 增加 proposed/schema_errors/rejected_proposals/candidates/value_verified；引擎记录被拒提议；`coerce_values_to_schema()` 先做声明类型转换（`schema_type_coercion`）；schema 不通过但逐字命中同 task/function 真实成功调用时按 `recorded_success_match`（value-verified 层）接受 |
| `src/tool_evolution/toolguard/guard.py` | live 修复后发 backend 前校验；被拒记录 `schema_rejected`/`schema_errors`；backend error-hint type override 披露 |
| `src/tool_evolution/toolguard/verify.py` | 关系型比较验证（关系断言/否定词/million 单位/多个候选/截断保护）、版本号比较、negation-aware contains |
| `scripts/{run_live_demo_hard,run_live_demo}.py` | 比较类任务改用关系验证器；V1 版本比较修正 |
| `scripts/run_live_demo_hard_v2.py` | failure-mode 分类只考虑答案字段；`--from-records` 从 checkpoint 重建 artifact（重算 aggregate/failure_mode、修 reproduce）；`--hn-schema {consistent,drifting}`（默认 consistent）与 calibration source 记录；V4 复现命令 |
| `src/tool_evolution/toolguard/live_hard.py` | **HN 契约修正**：`HARD_TOOLS.get_hn_item.story_id` 默认 integer；`LiveHardBackend(hn_schema="consistent")` 为默认，`"drifting"` 仅保留为契约修复能力演示（V3） |
| `scripts/run_multimodel_live.py` | reproduce 输出指向 V3；`--from-records` 从 records 重建 `tasks_per_model`，重建时重算 failure_mode；新增 `--hn-schema`（默认 consistent，from-records 自动沿用 checkpoint 旧值） |
| `scripts/run_mcp_demo_v2.py` | 校准改为两臂分别测量（`baseline_turns`/`guard_turns`）；`--budget-anchor baseline` 修正 P1-2 的预算不对称；`--from-records` 重建生成元数据（experiment id 等，records/aggregate 不变） |
| `benchmarks/toolguard/live_demo_hard_v4.json`（新） | **单模型 V4 主口径**：HN 契约修正后 310 格重跑（clean+1 baseline 59/93 vs guard 93/93，p=1.16e-10；复用 V3 calibration 作受控对照） |
| `benchmarks/toolguard/live_demo_multimodel_v4.json`（新） | **多模型 V4 主口径**：558 格重跑（clean+1 baseline 39/62、40/62、43/62 vs guard 62/62 全模型；p=2.4e-7/4.8e-7/3.8e-6；复用 V3 per-model calibration 作受控对照） |
| `scripts/toolguard_evidence_summary.py` / `rescore_toolguard_verifier.py` | 收录 V4 单/多模型 artifact（13 artifacts；rescore 756 条） |
| `scripts/rescore_toolguard_verifier.py`（新） | 新 verifier 离线只读重判已提交答案 |
| `scripts/collect_real_fault_traces.py` / `run_real_fault_repair.py`（新） | 真实线上错误 trace 采集 + repair-layer 在线验证（16 条真实错误，4/4 可推导契约错误修复，12 条语义错误诚实失败，成本 $0；瞬态 API 抖动单独记为 api_unavailable） |
| `src/tool_evolution/toolguard/live.py` + `scripts/run_live_demo_hard_v2.py` | `RealTraceFaultBackend` + `--fault-source real-trace`：把真实捕获错误作为 A/B 故障源（确定性、两臂同故障面、逐格记录 trace id/kind）；12 格 smoke：baseline=guard=5/6（真实错误多为不可修复类，属鲁棒性检查） |
| `scripts/run_live_demo_hard.py` / `run_live_demo_hard_v2.py` | **V5 更难任务 pilot（opt-in）**：`build_tasks(..., include_hard=True)` 追加 4 个长链/多源派生任务，`--include-hard-tasks` 启用；默认 V4 pool 不变。24 格 pilot：baseline 10/12 vs guard 12/12（ceiling 未破，负结果） |
| `benchmarks/toolguard/real_traces/hard_pilot_v5.json`（新） | V5 hard pilot 逐格证据（24 格，成本 $0.039） |
| `tests/test_toolguard/test_real_fault_traces.py`（新） | 固定真实 trace 证据（16/4/12 与不可修复集合），防止证据腐化 |
| `benchmarks/toolguard/mcp_demo_v3.json`（新） | MCP baseline/guard 分开校准重跑（96 格；clean+1 baseline 13/32 vs guard 32/32，p=3.8e-6） |
| `benchmarks/toolguard/live_demo_hard_v3_hnconsistent.json`（新） | HN 一致 schema 对照（186 格；baseline 51/93→61/93，HN 子集 2/12→12/12） |
| `benchmarks/toolguard/{live_demo_hard_v3,live_demo_multimodel_v3}.json` | `--from-records` 免费重建：重算 failure_mode、恢复 `tasks_per_model`、修 reproduce；**数字未变** |
| `scripts/parse_toolbench_trajectories.py` | 统计 unparseable observation 数 |
| `scripts/download_toolguard_data.py` | 修正下一步命令的 baselines 路径 |
| `tests/test_toolguard/`（+8 个文件） | 篡改负例、schema 负例、关系验证器负例、失败模式分类负例、coercion/value-verified/HN schema 模式负例、schema 索引保真度、real-trace 证据/后端；测试数 55 → 128 |
| `TOOLGUARD.md` / `REVIEW.md` | 数字与口径按上述修正（离线主表、LOMO 标注、382/782、HN 占比与单/多模型 V4 修正、schema 保真度、real-trace、V5 pilot、四层→三层等） |

---

## 7. 审阅 checklist（可勾选）

- [ ] `pytest tests/test_toolguard` **128 passed**（全仓 437 passed）；`validate_toolguard_artifacts.py` 13 artifacts 全过；`rescore_toolguard_verifier.py` 756 条 0 变化
- [ ] 篡改 tokens/rate/turns/tool_errors/Wilson/p 后 validator 必须报错（`test_artifact_tamper.py`）
- [ ] `toolguard_evidence_summary.py` 输出与 TOOLGUARD.md 主数字一致；live 段确为 records 重算
- [ ] 真实线上故障 trace：`collect_real_fault_traces.py` + `run_real_fault_repair.py` 重跑后 16 条真实错误中 4 条可推导契约错误恢复并在线验证，其余 12 条不可语义修复（`test_real_fault_traces.py`）
- [ ] `metrics.json` 可由 `build_metrics_artifact.py` 从评测输出重建；`proposed = schema_passed + schema_rejected_failures` 且 `accepted = schema_passed + value_verified_repairs` 自洽
- [ ] HN 契约：`LiveHardBackend()` 默认广告 integer `story_id`，仅 `hn_schema="drifting"` 才注入 string 漂移（`test_live_hard_backend_hn_schema_modes`）；V4 artifact `protocol.hn_schema=consistent`、`protocol.calibration.source` 指向 V3 锚点
- [ ] `schema_valid_rate` 在修复后的口径下为 LOMO 92.5% / cross-model 85.0% / cross-group 100%（cross-group 是因为唯一被证伪的日期类型已按证据修正，不是构造性 100%）；被拒提议、coercion、value-verified 分别计数
- [ ] schema 索引保真度修正有证据/留痕（`dataset.schema_index.fidelity_repair`：4 处类型修正 + 20 个回填，≥3 样本/≥90% 一致，不改 required，不做 integer↔number 拓宽）
- [ ] V5 hard pilot 负结果与 ceiling 含义在文档中如实记录（baseline 10/12 vs guard 12/12；单纯加长链不破 ceiling）
- [ ] coercion 修复与 value-verified 层有负例/正例测试（不可转换值不得被硬转；未逐字命中真实成功调用不得计入）
- [ ] LOMO 协议字段如实说明“固定三模型训练、gpt-4 只做测试”，且敏感性数字在 artifact 内
- [ ] `schema_check.py` 对字符串冒充 integer / enum 越界 / format 非法 / range 越界都判失败（`test_schema_validation.py`）
- [ ] live guard 修复后发 backend 前校验，被拒修复记录在 `guard_events.schema_errors`
- [ ] 比较类反向答案判失败（`test_relation_verifier.py`），已提交数据重判 0 变化
- [ ] `guard.py` 的修复确实发生在工具调用边界；验证器无 LLM judge（grep）
- [ ] 每个 live 实验的 failure 都能在 records 里找到对应记录（不是 aggregate 单独编造）
- [ ] 故障注入确定性：同 seed 同任务可复现同一 failure 序列（`test_fault_epoch.py`）
- [ ] 多模型实验确实 3 个不同模型 id、同一任务池/seed/预算；`tasks_per_model` 快照可核对
- [ ] TOOLGUARD 的“产物与复现溯源”表与脚本常量/commit 一致
- [ ] 局限清单是否覆盖了 §5 的 14 条
- [ ] 报告一个具体 finding（如有），并注明证据文件 + 行号

---

## 8. 安全与边界

- `.env`（DeepSeek/百炼 key）**永不提交**、`.env.example` 只有占位符；审阅无需 key；
- `docs/`、`STATUS.md` 是作者本地笔记（gitignored），不随仓库分发；
- 本会话与上一轮所有 API 消耗由作者承担；审阅者不必复跑；
- 本轮修复已 commit 并 push 到 `origin/feat/toolguard`（最新 `ea6a038`）；不修改旧 TEE 的 25 条历史指标；
- **给审阅 AI 的建议顺序**：先跑 §0 四条命令（全部离线、无需 key）→ 读 TOOLGUARD 的「结果（离线）」「第二轮 V4」「真实线上故障 trace」「V5 hard pilot」→ 用 §3 证据地图抽查 3–5 个 claim → 读 §5 的“不能声称”清单并逐条压测 → 最后对照 §7 checklist 报告 findings；
- 证据数据（`benchmarks/toolguard/real_traces/`）不是 contract artifact，但都有生成脚本与测试固定，审阅时可离线重跑 `collect_real_fault_traces.py` + `run_real_fault_repair.py`（只打免费公开 API）；
- 若审阅者发现“文档数字与 artifact 不一致”，优先修复文档并记录 finding，而不是改 artifact。
