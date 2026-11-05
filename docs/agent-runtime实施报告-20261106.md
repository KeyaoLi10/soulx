# agent-runtime 实施报告（P0–P5）

**日期**：2026-11-06
**依据**：`/home/research/krisl/agent-runtime-execution-manual.md`（下称"手册"）；
实施前审查 `docs/agent-runtime实施前审查-20261106.md`（commit dc1d0d9）。
**大原则**（用户指令）：语音前端——轮次判别、噪声抑制、grace、ASR、TTS、
播放——是第一公民，已调好的行为一律不动；本实施只**读**它们的结论，不写它们的
决策。

**总状态**：手册 P1–P5 全部落地，7 套离线测试 296 断言全绿，语音前端 4 套
不变量回归全绿。默认 `AGENT_RUNTIME_MODE=legacy`：V2 分支为死代码，语音路径与
实施前**字节等价**。本轮未重启任何服务、未做任何真实模型调用。

---

## 1. 修改文件清单与前端接点理由

| Commit | 文件 | 内容 | 为何必要 |
|---|---|---|---|
| dc1d0d9 | docs/审查报告 | P0 实施前审查与冲突裁决 | 手册 §20 要求先建保护清单 |
| e9db8e0 | `agent/contracts.py` | Plan/操作判别联合、严格 `parse_plan` | 手册 §3：模型控制输出进业务前完整校验 |
| cd1642a | `agent/store.py` | SQLite 持久层（10 张表） | §4：任务状态唯一事实源、幂等、outbox |
| e650ede | `agent/tools.py` | v2 工具层（web_search/fetch_page/get_datetime/calculate） | §8：只做真实基础工具；calculate AST 白名单替代 python_repl |
| a31a630 | `agent/scheduler.py`、`agent/delivery.py`、`agent/runtime.py` | 调度器 + 交付管理器 + `AgentRuntimeV2` | §5/§9：执行/交付分离、run 围栏、至少一次投递 |
| e107f14 | `services/qwen35_fused_server.py` | `/agent_chat_v2` NDJSON 端点 + G6 截断修复 | §12：控制通道需要可判定完成边界的传输 |
| 3ad4cb7 | `clients/llm_client.py`、`agent/router.py`、`agent/context.py` | 控制客户端 + 快脑路由器 + 快照 | §7：路由是控制平面的决策入口 |
| 8f2fc17 | `agent/frontend_adapter.py`、`app.py`、`agent/slow_brain.py`、`agent/contracts.py`、`agent/runtime.py`、`agent/context.py` | 窄接入层 + app 集成 + V2 慢脑 runner | §6/§10/§19 |
| 87f878d | `pvad_gate.py`、`scripts/empty_stream_replay.py` | 双轨回放取证收尾（见 §6 更正说明） | 补上一会话未提交的 pvad 侧切刀 |

**app.py 的前端接点（全部只读或旁路，共 5 处）**：

1. `_AGENT_RUNTIME_MODE` 环境开关（app.py:89）——legacy 时下面 4 处全部为死分支。
2. `_make_agent_v2()` 工厂（app.py:1015）——`_speak` 桥**复用**同一条
   `_run_speech_pipeline`（cut_pending_audio → _commit_pending_message →
   interrupt → reset → `_begin_generation`），三态返回值原样透传；grace 阈值取
   `settings["turn_grace_ms"]` 同一份设置；话权判断取现有 `_agent_can_deliver`；
   历史取 llm 会话副本。**不复制任何语音逻辑。**
3. 会话创建（app.py:1564-1580）——legacy AgentRuntime 的启动加
   `!= "enforce"` 门（防双出口双重播报）；V2 创建失败自动回落 legacy 并打日志。
4. `pipeline_worker`（app.py:1309-1321）——shadow 只调 `observe()`（后台线程，
   先等快脑空闲再路由，绝不抢 GEN_LOCK）；enforce 在 `gen_in_progress` 置位前
   `handle_utterance()` 后 `return`，旧生成路径本轮不跑。两者都以
   `settings["agent_enabled"]` 和 `session.agent_v2 is not None` 为前提。
5. websocket 关闭（app.py:1990）——`agent_v2.close()`：先落 `closed=1` 再关
   store 连接（顺序错误会导致标记永远写不上）。

**明确未动的**（前端保护清单核对，见 §6）：turn server 五道门、grace 宽限窗、
滑窗回声防御、假 turn 丢弃、自说自话过滤、pvad 流门控/迟滞、两遍式 ASR 修正、
VoxCPM 音色与 checkpoint、`/chat` 与 `/agent_chat` 的生成路径与采样行为、
`_TagSplitter` 对 `/chat` 的行为（`_stream_tagged` 独立，G6 修复只在
`/agent_chat_v2` 路径生效）。

## 2. Schema、迁移、状态图与版本化协议

- **数据库**（`checkpoints/agent_v2/{sha1(client_id)[:16]}.sqlite3`，每连接一个，
  全新库，无迁移问题）：`meta`、`sessions`（closed/closed_reason/timezone）、
  `requests`（request_id 幂等）、`operations`、`tasks`（execution_status +
  pending_question/pending_question_id）、`task_revisions`（revision 单调）、
  `runs`（lease token + 围栏）、`events`、`tool_calls`、`deliveries`（三态：
  played_confirmed 估计 / interrupted_partial 终态 / retry≤3）、`outbox`（至少
  一次）、`route_log`（shadow/enforce 观测留痕）。不删任何历史证据（§19）。
- **任务状态图**：queued → running → {completed | failed(outcome=empty/…) |
  waiting_input →(answer)→ queued | canceled}；blocked_dependency、
  cancel_requested、superseded 为过渡/终态。run 有 lease 围栏，僵尸 run 可被
  接管不双跑。
- **版本化协议**：plan `schema_version: 1`；快照 `snapshot_version` =
  Σrevision + 任务数（单调）；任务播报绑定 (task_id, revision)；NDJSON 帧
  `{"v":2,"seq":n,...}`，`done.finish_reason=stop` 才可提交，length/cancelled/
  error 均不提交。
- **工具线格式**：Qwen3.5 原生 `<tool_call>` XML（`_TOOL_FN_RE`/`_TOOL_PARAM_RE`
  解析），路由器单工具 `route_turn(plan_json)`。

## 3. 测试：实际运行的命令与结果

全部用 `/home/research/krisl/soulx/venv/bin/python -I`（pvad 用 venv-frc）：

```
scripts/agent_runtime_test.py   45 通过   （runtime/store 集成、幂等、围栏）
scripts/agent_store_test.py     53 通过   （事务/并发/outbox/lease）
scripts/agent_tools_test.py     63 通过   （AST 白名单/SSRF/时区）
scripts/agent_sched_test.py     40 通过   （调度/交付三态）
scripts/agent_v2_proto_test.py  22 通过   （NDJSON 帧/G6/or-bug）
scripts/agent_router_test.py    37 通过   （真实 socket 拆包/路由/快照）
scripts/agent_adapter_test.py   36 通过   （shadow/enforce/grace/续说/表达轮/runner）
                               ——共 296 断言
语音前端不变量回归：
scripts/agent_smoke.py --offline  17/17   （legacy 路径烟囱）
scripts/selftalk_replay.py       12/12   （自说自话过滤）
scripts/empty_stream_replay.py   15/15   （投递三连修，桩已对齐取证新签名）
scripts/pvad_stream_replay.py    24/24   （venv-frc；pvad 门控+影子分离）
```

**未运行项**（明确列出）：路由语义评测（§18.2 建议的 120 条人工标注集）；
路由延迟实测（快脑 ~25 tok/s、计划 80–150 token，预估 3–6s，需实测定基线）；
enforce 实机语音回归；任何真实模型调用与服务重启。离线测试中的路由 transport、
慢脑 `_generate`、`web_search` 均为测试替身，**不代表真实模型/工具的业务结果**。

## 4. 路由评测

未做（同上）。离线测试覆盖的是**确定性正确性层**（§18.2 第一层）：格式修复、
传输终态、G2 无目标取消拦截、快照焦点确定性等。语义层的逐类
precision/recall 与混淆矩阵待标注集与用户授权。

## 5. 真实工具演示记录

未做——需要真实模型调用（慢脑/快脑服务），本轮按 §0 约束全部离线。
离线最接近的等价物是 `agent_adapter_test.py` 场景 3/8（enforce 端到端：提交 →
模板回执上口 → 执行 → 交付；mixed 模式回执+表达轮不吞问题）与 B5
（web_search 打桩 + calculate **真实执行**，工具观察进二轮、sources 归集）。

## 6. 前端保护核对与取证更正

- 回滚方式：不设 `AGENT_RUNTIME_MODE`（或设 legacy）即回滚；已落库的任务数据
  保留不删。enforce 只在快脑服务重启后的维护窗口开启（新连接生效，不热切换
  拥有任务的两套 runtime）。
- 声明（§18.2 能力声明）：无播放 ACK，交付为 server-estimated
  (played_confirmed)；无恢复 token，会话为 connection-scoped；真实工具集成
  未测试。
- **取证更正**：commit 8f2fc17 的 app.py 部分意外捎带了上一会话已完工未提交的
  双轨回放 mark_cut 取证改动（`cut_pending_audio`/`run_break(since)`/
  `mark_cut(t, since)`）。该改动本身已由 empty_stream/pvad_stream 回归覆盖，
  pvad 入口一半在 87f878d 补齐；未改写历史，仅在此声明。

## 7. 下一步（待用户授权）

1. shadow 模式实测：设 `AGENT_RUNTIME_MODE=shadow` 重启 web app，收集
   route_log 的路由正确率与延迟分布（不扰语音）。
2. 路由标注集评测（§18.2）。
3. 指标口径接入 §18.1 的时间线字段（utterance_admitted → route_done → …）。
4. enforce 实机验证清单：grace 门与前端同阈、续说作废、回执模板音色一致、
   打断三态、enforce 下旧 auto_cancel/auto_delegate 确认不运行。
