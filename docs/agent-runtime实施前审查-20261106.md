# Agent Runtime 实施前审查与实施决策（2026-11-06）

**任务来源**：用户指示——按《SoulX 快慢脑与任务 Runtime：实现 Agent 执行手册》
（`/home/research/krisl/agent-runtime-execution-manual.md`）实施方案；冲突点尽量遵循手册
设计；**语音前端（轮次判别、噪声抑制、pvad、grace、ASR、TTS、播放）是第一公民，
调好的行为一律不动**；按阶段顺序逐模块提交；实施期间不重启任何服务。

**基线**：commit `1151c12`，工作树干净。无 AGENTS.md。`soulx/data/`、`logs/` 已被
gitignore（SQLite 任务库落 `soulx/data/agent_runtime.sqlite3`，不入库）。

## 一、审查结论（对照手册 §2 差距表，全部经源码核实）

### 已具备
- 双入口（SoulX commit / FRC commit）收敛于 `pipeline_worker`；I1 已成立（partial
  只进 UI，任务操作只在准入后）。
- 唯一语音出口 `_run_speech_pipeline`（三态返回 + grace hold + pending-message +
  截断 + mark_cut + gen_serial），用户话轮与后台交付共用。
- 快慢脑分离（:6007 GPU6 / :6008 GPU1）；慢脑 3+1 预算、480s deadline、轮内去重。
- `<|im_start|>` logits 禁 token（5aa69a6）在 `_stream_agent` 内快慢脑共用——
  **手册未提及，v2 协议必须继承**。
- legacy 任务运行时：仲裁轮询、can_deliver 三条件、静默掐死重试预算、播出后打断
  =终态、交付前 note 回收（pop_message 单出口保证）。
- 确定性安全网 auto_cancel / auto_delegate（2026-10-30 A/B 定型）。
- 回归资产：agent_smoke(24)、agent_e2e(11)、selftalk(12)、empty_stream(15)、
  pvad_stream(24)、cancel_repro、frc_mode_test；reconnect_test 会重启进程，禁当单测跑。

### 实际缺口（G1-G15，详见审查记录）
G1 无任务版本/amend；G2 cancel 三层混淆+点名取消不清 agenda；G3 执行/交付状态合并
（异常按成功、重试耗尽算 delivered）；G4 无持久化、断线即全取消；G5 **grace 窗内
delegate_task 可提交**（工具执行在生成器轮末同步跑，早熟话轮留下用户从未听到受理的
任务）；G6 `_TagSplitter.flush` 可吐未闭合工具调用；G7 无幂等/请求登记；G8 无
needs_input；G9 无任务依赖；G10 auto_cancel 一律取消全部；G11 delegate 同步 3s 健康
检查+第 3 任务直接拒绝；G12 /agent_chat 分帧脆弱（单字节前缀、参数恒字符串、
temperature 0 当默认）；G13 can_deliver 非原子；G14 无 direct 越界承诺防护；
get_datetime 无 IANA 时区；G15 播放证据全为 estimated。

## 二、冲突点裁决（用户原则：尽量遵循手册；语音前端第一公民）

| 冲突 | 裁决 |
|---|---|
| 禁用 auto_cancel/auto_delegate | 按手册：enforce 模式下副作用禁用、保留为观测指标；legacy/shadow 模式原样保留。enforce 为非默认，demo 默认路径不受影响 |
| python_repl → calculate | 按手册：新 runtime 默认工具集 = web_search/fetch_page/get_datetime/calculate，python_repl 由 `AGENT_TOOLS_PYTHON_REPL=1` 开关恢复；**legacy 的 TOOL_SCHEMAS 原样保留**（legacy 行为不变） |
| 结果 note 改临时上下文 | 按手册 §13.3：新交付走临时结构化上下文（generate_with_history 新增可选 extra_context，注入会话副本不落持久历史）；静默掐死=重试、播出后打断=终态两条语义在新 delivery 状态机中等价复刻 |
| 逐 PCM 校验点 | 手册意图保留、位置调整以保护第一公民：校验做在生成器包装层 + 现有 stop_event（失效即置位，复用既有打断机制），**不改 `_speak_chunk` 内部** |
| 路由延迟 | 按手册：shadow 先行实测；router_deadline 10s、router_max_tokens 1024；白名单直达为 P3 后续项 |
| v2 协议 | 继承 `<|im_start|>` 禁 token 与 GEN_LOCK；`/chat` 采样不动；v2 专用 temperature=0 贪心；`_TagSplitter.flush` 拒绝未闭合工具块（legacy /agent_chat 同修——只防截断调用被执行，手册 §12 明确要求） |
| grace 提交门 | 按 §6.1 复用同一 grace 阈值（同一 setting/常量），判据在 adapter 侧读 `session.last_voice_ts`（与 `_run_speech_pipeline` 同一证据源）；**不复制第二套计时阈值，不改 grace 时机** |
| 任务表注入 | 控制轮恒看任务快照；表达轮保持 `_TASK_MENTION_RE` 门控（2026-10-30 闲聊不汇报实测不被破坏）；路由结果绝不作为尾部 note 注入表达轮（实测会压低触发率） |

## 三、实施顺序（逐模块提交）

1. **P0** 本文档。
2. **P1a** `agent/contracts.py`：操作/Receipt/快照/信封 Schema + 严格 plan 解析。
3. **P1b** `agent/store.py`：SQLite 事务/迁移/幂等/outbox/围栏。
4. **P1c** `agent/tools.py`：calculate（AST 白名单）、fetch_page SSRF 目的地址核验、
   get_datetime IANA、统一 ToolResult；legacy TOOL_SCHEMAS 不动。
5. **P1d** `agent/scheduler.py` + `agent/delivery.py` + `runtime.py` 新控制器：
   有界队列/run lease/依赖/取消三层/交付状态机（假慢脑全离线测试）。
6. **P2a** `services/qwen35_fused_server.py`：`/agent_chat_v2` NDJSON + flush 修复。
7. **P2b** `clients/llm_client.py` v2 客户端 + `agent/router.py` + `agent/context.py`。
8. **P3** `agent/frontend_adapter.py` + `app.py` 接入（AGENT_RUNTIME_MODE=
   legacy|shadow|enforce，默认 legacy=行为与今日逐字节等价）+ shadow 观测。
9. **P4** `agent/slow_brain.py` 版本化/needs_input/事件信封。
10. **P5** 收尾：修改日志、实施报告、memory、全量回归（agent_smoke --offline、
    selftalk、empty_stream、pvad_stream）、push。

**保护清单（本轮明确不改）**：`frontend/index.html`、`style.css`、`processor.js`、
`script.js`（task_update 仅增字段）、`pvad_*.py`、`speaker_sep.py`、`frc_engine.py`
音频逻辑、turn server、两个 TTS 服务、`_run_speech_pipeline` 内部（仅允许 None 默认
的加参）、`llm_client` 既有方法签名（仅加可选参数）。

**未验证声明**：本轮全部为代码+离线验证；不启动服务、不调真实模型、不做实机语音
回归（P5 的 baseline vs enforce 对比与路由延迟实测留待用户授权后执行）。
