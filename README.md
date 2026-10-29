# SoulX-Duplug 对话系统 — 部署与解读总览

[SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug)（dialogue-system 分支）全双工口语对话系统的完整部署。
核心思想：**一个 0.6B 的流式语义 VAD / 轮次状态预测模型（即插即用的 "Duplug"）挂在任意 LLM+TTS 级联前端上**，实时判断用户"说完没有 / 该不该接话 / 有没有被抢话"，实现全双工对话。

原方案的 Qwen2.5-7B-Instruct + IndexTTS/CosyVoice 在本机不可得，替换为本地已有资源：
**Qwen3.5-9B**（LLM）+ **FireRedTTS**（语音合成），线协议与原仓库完全兼容。

> 代码级深度解读见 **[docs/代码解读.md](docs/代码解读.md)**；
> 故障修复记录见 **[docs/修改日志.md](docs/修改日志.md)**
> （2026-10-26：修复 FireRedTTS 全局锁死锁导致 Web 对话几轮后永久沉默的问题）。

## 1. 系统架构

```
                         用户音频 16kHz float32，160ms/chunk
                                   │
        ┌──────────────────────────▼───────────────────────────┐
        │ turn server   ws://localhost:8000/turn      (GPU 5)   │
        │  SoulX-Duplug-0.6B：glm-4-voice 音频 tokenizer         │
        │  → projector → Qwen3-0.6B(扩词表+LoRA) 逐 chunk 推理   │
        │  状态机: idle / nonidle / speak(附 paraformer 转写)     │
        └──────────────────────────┬───────────────────────────┘
             每个会话独立 buffer / KV cache / 60s GC
                                   │ state="speak" 且带用户话轮文本
                                   │ （trigger: "complete"=立即 / "wait"=静音确认）
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ Web app :8090  ——  快脑 + Agent Runtime（确定性仲裁）   │
        │  pipeline_worker: 话轮门控→历史→生成→TTS→推送          │
        │  complete 触发话轮 → 0.9s 宽限保持窗（抢话修复，        │
        │    复语即静默丢弃整次生成）                              │
        │  快脑 = /agent_chat(:6007, thinking=off) + 委派工具:   │
        │    delegate_task / cancel_task（原生 tool calling）    │
        │  AgentRuntime: 任务表 / 议程 / 交付仲裁线程；           │
        │    结果在"用户空闲+无播放+无生成"的空档期经同一语音出口  │
        │    （任务状态仅当用户话轮提及任务时注入 → 闲聊不汇报）    │
        └───────────────┬──────────────────────┬───────────────┘
                        │ 简单对话/受理/汇报      │ 委派任务(后台线程)
                        ▼                       ▼
        ┌────────────────────────┐  ┌──────────────────────────┐
        │ LLM :6007 (GPU 6)      │  │ 慢脑 :6008 (GPU 1)       │
        │ Qwen3.5-9B thinking=off│  │ Qwen3.5-9B thinking=on   │
        │ /chat /agent_chat(S/R/T)│ │ agent 循环+工具: web_search│
        │ /v1/score(frc 语义EOT) │ │ /fetch_page/python_repl/  │
        └────────────────────────┘  │ get_datetime (agent/tools)│
                                    └──────────────────────────┘
                        │
                        ▼
   每段回复 → VoxCPM2 TTS(:8085) → 24kHz PCM
   mic 旁路 → PVAD sidecar(:8086,CPU)     ← 异步 feed,不进轮次路径
```

### 1.1 FRC 前端处理模式（turn_engine=frc，2026-10-30）

Settings 面板可把会话的轮次引擎从 soulx 切到 **frc**（FireRedChat 架构，
LLM/TTS 与 soulx 模式完全共用，仅轮次判别不同；LLM+TTS 级联以下不变）：

```
mic chunk → pvad sidecar :8086（零说话人嵌入 = 通用语音活动检测）
  持续 prob≥0.5 → 说话开始（2.5s 前滚防切头）；连续 0.4s <0.5 → 静音结束
  → 双通道 EOT 并行判定:
      ultraVAD :8082  （20s 滚动音频窗 + 上一条助手文本）
      :6007 /v1/score （FireRedASR 转写 + 助手文本 → 是/否首 token 概率）
    min 融合，双通道 ≥0.2 才快速提交；否则 hold 至 EOS+6s 封顶
  提交 → FireRedASR :8010 整句转写 → 同一条 pipeline_worker（含门控/agent）
```

与 soulx 路径的取舍：frc 无实时转写（整句后置转写）、依赖 3 个额外服务，
但语义 EOT 对"句中停顿不是说完"的判别远强于 0.6B 轮次模型（实测半句
uv=0.088/llm=0.018 稳定 hold，整句 0.90+ 快速提交）。切换即 reset，
soulx 路径代码零改动。回归：`scripts/frc_mode_test.py`。

## 2. 目录结构

```
soulx/
├── README.md                       # 本文件
├── docs/代码解读.md                 # 代码级深度解读
├── repo/                           # SoulX-Duplug dialogue-system 分支克隆
│   ├── server.py                   # turn server 入口（FastAPI websocket）
│   ├── service/                    # 引擎层：engine / session / TurnModel 状态机
│   ├── model/                      # State_Prediction_Model + paraformer ASR
│   ├── config/config.yaml          # 模型与服务配置
│   ├── dialogue_system/            # 应用层
│   │   ├── app.py                  # Web 实时对话应用
│   │   ├── offline_infer.py        # 离线推理（本次生成对话音频的入口）
│   │   ├── clients/                # vad / llm / tts 三个客户端
│   │   ├── eval/                   # 官方评测脚本（行为/停顿/平滑抢话/打断）
│   │   ├── modules/                # 原版 TTS/LLM 后端（IndexTTS/CosyVoice，未用）
│   │   └── frontend/               # Web 前端静态页
│   ├── example_client.py           # turn server 最小测试客户端
│   └── pretrained_models -> ../pretrained_models
├── pretrained_models/              # HF 权重（hf-mirror 分段并行下载，已校验）
│   ├── Qwen3-0.6B-expand_vocab_v2/ # 扩词表基座（含 LoRA，llm_dim=1024）
│   ├── SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth
│   └── glm-4-voice-tokenizer/      # WhisperVQ 音频 tokenizer
├── services/qwen35_fused_server.py # 自研：Qwen3.5-9B + FireRedTTS 融合服务
├── services/voxcpm_tts_server.py   # 自研：VoxCPM2 TTS 微服务（web 对话默认 TTS，:8085）
├── scripts/                        # 部署脚本集
│   ├── start_services.sh           # tmux 全量拉起
│   ├── install_deps.sh             # 依赖安装 + 导入自检
│   ├── fetch_wheels_parallel.sh    # 清华镜像分段并行 wheel 下载器
│   ├── download_models.sh / download_remaining.sh
│   ├── build_eval_inputs.py        # eot-bench-zh / xyz → 24 条评估输入
│   └── smoke_test.py               # 四服务冒烟测试
├── data/eval_inputs/               # {eotbench_zh,xyz}/<id>/input.wav（24 条）
├── modelscope_cache/               # paraformer ASR（首次启动自动拉取，~2GB）
├── venv/                           # py3.10：turn server / web / 离线推理
├── venv-voxcpm/                    # py3.10：VoxCPM2 TTS 服务（torch 2.6.0+cu124）
├── /mnt/disk3/krisl/soulx/VoxCPM2/ # VoxCPM2 权重（4.7GB，大盘）
├── logs/                           # 各服务与推理日志（tmux 同步 tee）
└── results/
    ├── soulx-dialogues-20261026.zip  # ★ 生成的对话音频交付包（81MB）
    └── soulx-dialogues-20261026/      # 解包目录（含 manifest.json）
```

## 3. 服务与端口（tmux 会话 `soulx`）

| tmux 窗口 | 服务 | GPU | 端口 | venv |
|---|---|---|---|---|
| `tts` | FireRedTTS（`full_duplex/frc_services/tts/server.py`，full_duplex FRC agent 共用，兼作 web 的 firered 回退） | 7 | 8081 | `full_duplex/venv` |
| `voxtts` | **VoxCPM2 TTS（web 对话默认 TTS，`services/voxcpm_tts_server.py`）** | 7 | 8085 | `soulx/venv-voxcpm` |
| `llm` | 融合 LLM 服务 = **快脑**（`services/qwen35_fused_server.py`，`/chat` + `/agent_chat` 工具调用流） | 6 | 6007 | `full_duplex/venv-frc-llm` |
| `slowllm` | **慢脑**（同款脚本第二实例，thinking=on，后台任务 agent 循环的生成后端） | 1 | 6008 | `full_duplex/venv-frc-llm` |
| `turn` | SoulX turn server（`repo/` uvicorn server:app） | 5 | 8000 | `soulx/venv` |
| `ultravad` | ultraVAD EOT 音频通道（frc 模式专用，`full_duplex/frc_services/ultravad/server.py`；加载 ~4min，>20s 音频会 500） | 5 | 8082 | `eot_bench/venv` |
| `frcasr` | FireRedASR 整句转写（frc 模式专用，`full_duplex/frc_services/asr`；需 tempo env 的 ffmpeg 在 PATH） | 5 | 8010 | `full_duplex/venv` |
| `pvad` | **PVAD 主说话人门控 sidecar（`services/pvad_server.py`，CPU；frc 模式下兼作通用语音活动检测）** | – | 8086 | `full_duplex/venv-frc` |
| `web` | Web 演示应用（`repo/dialogue_system/app.py`，`TTS_ENGINE=voxcpm\|firered`，会话级 `turn_engine=soulx\|frc`） | – | 8090 | `soulx/venv` |

`tmux attach -t soulx` 查看；日志在 `logs/*.log`。TTS 回滚：`TTS_ENGINE=firered`
重启 web 窗口即可。voxtts 启动需 `LD_LIBRARY_PATH=/mnt/disk1/miniconda3/envs/tempo/lib`
(torchcodec 要 ffmpeg 库)，见 `start_services.sh` 内注释。frc 模式依赖
:8082/:8010/:8086 三服务在线，任一宕机该会话 frc 轮次判别降级（soulx 模式不受影响）。

## 4. 对话音频生成结果

输入构建（`scripts/build_eval_inputs.py`）：eot-bench 中文 validation（344 行）与
小宇宙 xyz-eot-data（466 行）按时长过滤 [8,90]s 后均匀采样各 12 条，共 **24 条**，
重采样 16kHz 存为 `data/eval_inputs/<来源>/<ID>/input.wav`。

离线推理（`repo/dialogue_system` 下）：

```bash
/home/research/krisl/soulx/venv/bin/python offline_infer.py \
  --config_path ../config/config.yaml --response_tail 2 \
  --eval_dirs /home/research/krisl/soulx/data/eval_inputs/eotbench_zh \
              /home/research/krisl/soulx/data/eval_inputs/xyz
```

每条样本在输入旁产出：`output.wav`（助手侧）、`dialogue.wav`（混合）、
`states.json`（轮次状态时间线）、`infer_time_info.json`（延迟指标）。

**结果**（`results/soulx-dialogues-20261026.zip`，24 样本 + manifest.json）：

- 24/24 成功；22 条含实质助手语音（活跃 12–47%）
- 2 条 xyz steve_472 样本（t0004 / t0018）助手静音：连续播客语音中轮次检测器
  未判定出用户话轮结束，属系统真实行为而非故障
- 首包延迟（LLM+TTS 融合流）1.1–3.6s

## 5. 与原仓库的差异

1. **LLM**：Qwen2.5-7B-Instruct → 本机 Qwen3.5-9B。`services/qwen35_fused_server.py`
   按原 `modules/qwen_llm/llm_server.py` 线协议实现 `/chat`、`/chat_indextts`，
   分句策略一致（标点切分 + 最小 3 token 段长）。运行于 `venv-frc-llm`
   （transformers 5.x，无 accelerate，须 `.to("cuda:0")` 加载）。
2. **TTS**：IndexTTS/CosyVoice → FireRedTTS → **VoxCPM2（2026-10-27 起 web 对话默认）**
   → **2026-10-29 加自训练 LoRA + 参考音色锚定**。`clients/tts_client.py` 含
   `FireRedTTS` 与 `VoxCPM_TTS` 两个等接口类，公共文本清洗在
   `_clean_tts_text`；`app.py` 由 `TTS_ENGINE` env 选择（默认 voxcpm，
   回滚设 firered）。VoxCPM2 走独立微服务 `services/voxcpm_tts_server.py`
   （:8085，venv-voxcpm）：48kHz 生成经 soxr 流式重采样到 24kHz 裸 PCM 流。
   **跨段音色一致性**：zero-shot 模式每次调用从新高斯噪声采样（无 seed），
   每个 LLM 分段各是一个随机音色（实测 ECAPA 段间余弦最低 0.22）——修复为
   参考克隆模式：启动时锚定一个参考音色（`data/voxcpm_anchor.wav`，持久化，
   删文件即重摇；`VOXCPM_REF_WAV` 可显式指定），`build_prompt_cache` 一次、
   所有分段复用（修复后段间余弦 0.65-0.75）。**自训练 LoRA**
   （`/vepfs-cnbjc4e12c6d7e1f/krisl/step_0000500`，r=32）经
   `VOXCPM_LORA_PATH` 加载，加载校验失败即拒绝启动。实测：首块 80ms，
   RTF 0.32。详见 `docs/修改日志.md` 2026-10-29 条。
3. **离线推理补丁**：`offline_infer.py` 新增 `--response_tail <秒>`——原版
   `output.wav` 按输入等长裁剪，而这批输入是完整用户话轮、助手回复大多落在
   输入结束之后；补丁渲染前把音轨补静音到助手响应结束，保留完整对话。
4. **Web UI（2026-10-28 深色录音室控制台版）**：全宽横向布局（舞台 +
   400px 右栏，16:9 无两侧留空）、深炭底霓虹青配色、声呐扩散环语音球、
   整宽波形监视条、会话计时器、滚动对话日志（用户气泡随 ASR 实时更新、
   助手气泡流式追加）、无外部 CDN 依赖、无品牌字样。
   Settings 面板（`update_config` WS 事件，实时生效）：LLM 人设/剧本
   prompt（含预设）、TTS 引擎按会话切换（voxcpm/firered）、VoxCPM2 音色参数
   （cfg/quality steps/voice description）、记忆轮数、首音频分段长度。
   详见 `docs/修改日志.md` 2026-10-28 条。
5. **主说话人门控 + 浏览器回声消除（2026-10-29）**：接入 FireRedChat 的
   fireredchat-pvad 模块（ONNX 个性化 VAD + ECAPA 说话人嵌入），作为
   `services/pvad_server.py`（:8086，CPU，运行于 full_duplex/venv-frc ——
   唯一带 speechbrain/onnxruntime 栈的环境，跨 venv 复用模式同 eot_bench
   的 ultraVAD）。**架构约束：完全不碰轮次控制器路径**——mic chunk 经
   `feed()`（队列 put）异步旁路送入 sidecar，pvad 概率仅作 UI 实时遥测；
   身份门控用 ECAPA 余弦（实测 pvad 概率在硬负例上无法区分说话人），
   只在两个决策点（抢话、话轮完成）以 worker 线程询问，信号未知一律
   fail-open（服务宕机验证过：对话完全不受影响）。首个通过门控的话轮
   自动注册主说话人；三态模式 off/shadow/enforce + 相似度阈值可在
   Settings 实时调。回声/降噪走浏览器 WebRTC 处理链
   （echoCancellation + noiseSuppression + autoGainControl，与 FireRedChat
   官方 demo 同款——它没有服务端 AEC 模型）。详见
   `docs/修改日志.md` 2026-10-29 条。
   **2026-10-30 起默认 off（直连 SoulX 轮次判别，零 pvad 开销）**：
   实测浏览器 WebRTC 链路下同人 ECAPA 余弦仅 0.15–0.57，低于按干净
   音频标定的 0.5 阈值，enforce 会持续误拒注册用户本人（"30 秒后
   一直不说话"第四起）；且防污染修复后弱参考无法自愈。代码全部保留，
   Settings 可随时切回 shadow/enforce。详见
   `docs/修改日志.md` 2026-10-30 PVAD 默认下线条。
6. **快慢脑 Agent 架构（2026-10-29）**：在级联底座上加入"快脑即时交互 /
   慢脑异步任务 / 确定性 Runtime 仲裁"双循环（设计来自 task.md 参考方案 +
   本系统落地改造）。快脑 = 原 :6007 实例 thinking=off，新增 `/agent_chat`
   端点（S/R/T 三通道标签流：可说文本 / 推理 / 完整工具调用 JSON）+
   原生 tool calling（`delegate_task`/`cancel_task`）；慢脑 = :6008 第二实例
   thinking=on，agent 循环带 4 个工具（DDG 搜索经 curl——实测 requests 的
   TLS 指纹被 DDG 反爬挑战、curl 同刻通过；DDG 被限流时自动回落百度；
   网页抓取；python_repl 子进程沙箱；时间）。**2026-10-30 起慢脑默认
   thinking=off + temperature 0.3（demo 提速：思考轮时长重尾分布 5–100s+，
   关闭后天气类任务 15–25s → 6–9s）；`SLOW_BRAIN_THINKING=1` 切回深度
   推理模式。**`agent/` 包：tools.py / slow_brain.py / runtime.py
   （任务表 + 议程 + 交付仲裁线程）。交付时机 = 用户空闲 + 无播放 + 无生成
   的空档期，经与用户话轮完全相同的语音出口（`_run_speech_pipeline` 抽象，
   打断/截断/待提交语义一致）；结果以 user 角色系统提示注入历史。**改口
   取消是 Runtime 确定性执行的**（快脑实测会说"已取消"却不调工具，
   `auto_cancel` 正则在生成前先行取消并注入确认提示）。工具轮次
   不进持久历史。前端新增 Background Tasks 面板（任务卡片 + Agent 开关，
   `task_update` WS 事件）。详见 `docs/修改日志.md` 2026-10-29 快慢脑条。
7. **打断风暴修复（2026-10-29 同日）**：播放期噪声/回声被 turn server 判为
   nonidle，而说话人门控因"注册永远凑不齐 3s"而 fail-open → 每块都打断
   播放（TTS 卡顿、speaking/listening 跳变）。修复分层：注册音频库
   （bank，短话轮跨轮累积，第 2 轮前必注册）、抢话迟滞（3 块/480ms + 每
   episode 单次打断）、播放期 fail-closed（不可验证的抢话一律拒绝）、
   streak 身份评分、播放期 LISTENING 状态门控；ECAPA 窗口全部改为自然
   连续音频 + 边缘静音剪除（拼接注册与静音填充窗口均被实测证明垃圾分），
   注册改为每话轮滚动更新（对抗同人音色漂移）。回归：
   `scripts/bargein_storm_test.py`（回声 -20dB 回灌 0 打断）、
   `scripts/ecapa_window_calib.py`（窗口校准表）。详见
   `docs/修改日志.md` 2026-10-29 打断风暴条。
8. **播放期话轮门控（2026-10-29 同日，两度调整）**：先引入"播放期宽容"
   （低于阈值但非自对话的话轮放行，当时归因浏览器 AEC 畸变播放期的注册
   用户），同日经用户决定**移除**——用户实测全程戴耳机（AEC 无回声可消、
   不产生畸变），播放期低分的真实原因是参考被他人语音拉偏，而宽容恰好
   放行了旁人（他人说话得到 LLM 回复）。现行行为：播放与空闲同一严格
   阈值；保留的播放期特例是不可评分的短回声起始段靠自对话文本检查
   （召回率 ≥0.6 模糊匹配）拒绝。历史方案（召回率匹配、打分窗按话轮
   切分）继续生效。详见 `docs/修改日志.md` 2026-10-29 各条。
9. **毛刺话轮过滤（2026-10-30，同日下线）**：针对安静环境"幻听"（paraformer
   在回声/噪声残留上幻觉出"怎么样""是不是""平时"短碎片）曾上线 chunk 计数
   过滤，线上实测两条判据全部失效后**整体下线**（`_BLIPTURN_FILTER=False`
   总开关，代码保留）：真人 WebRTC 语音的 turn-server 状态逐块 nonidle/idle
   交替（永远无 3 连续，整句被误杀），而幻听碎片 episode 与真人同形态、
   计数不可分。防幻听待另找判据（pvad 门控加强或音频取证方向）。完整
   分析见 `docs/修改日志.md` 2026-10-30 条及其追记。
10. **抢话修复：complete 触发宽限窗（2026-10-30）**：turn server 的
   `<|user_complete|>` 路径一有预测立即接话、零静音确认，中文句中停顿
   常被误判为说完 → 抢话。修复：speak 返回新增 `trigger` 字段，
   complete 触发的话轮在 app 侧获得 0.9s 宽限保持窗（LLM 照常流式生成
   但一切对用户不可见；窗口内持续复语即静默丢弃整次生成——未说出的
   回复不进历史，用户续说经 llm_client groupby 并入同一 utterance）。
   wait（静音确认）触发与后台交付路径不受影响。该修复同时消除了
   E2E 阶段 1 的委派失败根因（TTS 句间停顿把一句话切成 4 个 complete
   话轮，截断残段污染历史让模型误以为已委派）。详见
   `docs/修改日志.md` 2026-10-30 抢话修复条。
11. **任务状态按提及注入（2026-10-30）**：慢脑任务运行期间，快脑每轮
   无条件注入任务状态导致反复主动汇报进度、无法自由闲聊。改为
   `_TASK_MENTION_RE` 触及任务/进度/结果关键词才注入（auto_cancel
   确认始终注入），系统提示改为【闲聊不汇报】规则。
12. **FRC 前端处理模式（2026-10-30）**：见 §1.1。`frc_engine.py` 移植
   full_duplex 双 EOT worker 架构（pvad 通用 VAD 端点检测 + ultraVAD/
   Qwen 语义双通道 min 融合 + FireRedASR 整句转写），提交后进同一条
   pipeline_worker。`:6007` 新增 `/v1/score` 首 token 概率打分端点。
13. **配置**：`config/config.yaml` 按原 README 基准 `max_wait_num: 5`、
   `far_field_threshold: 0`。

## 6. 环境与网络

- `soulx/venv`（py3.10）：torch 2.6.0+cu124、transformers 4.52.1、peft 0.16.0、
  pytorch-lightning 2.5.2、WeTextProcessing、funasr 1.2.6 + modelscope 1.28.2、
  datasets 3.6.0（钉 huggingface-hub==0.36.0）。
- 网络路由：GitHub→ghfast.top 镜像；HF→hf-mirror.com（分段并行）；PyPI→清华镜像
  （单流 ~340KB/s，需 `fetch_wheels_parallel.sh` 分段并行 ~5MB/s）；paraformer→modelscope。
- 已知非问题：turn 模型加载时 `load_state_dict strict=False` 重试
  （`embed_tokens_func.weight` 多余键）为仓库自身预期行为。

## 7. 常用操作

```bash
# 冒烟测试（四服务）
/home/research/krisl/soulx/venv/bin/python -I scripts/smoke_test.py

# 重启某服务（示例：turn server）
tmux send-keys -t soulx:turn C-c
tmux send-keys -t soulx:turn 'cd /home/research/krisl/soulx/repo && CUDA_VISIBLE_DEVICES=5 \
  MODELSCOPE_CACHE=/home/research/krisl/soulx/modelscope_cache \
  /home/research/krisl/soulx/venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000 \
  2>&1 | tee -a /home/research/krisl/soulx/logs/turn.log' Enter

# 全量拉起（冷启动）
bash scripts/start_services.sh

# 重建评估输入
/home/research/krisl/soulx/venv/bin/python -I scripts/build_eval_inputs.py

# PVAD 主说话人门控测试（服务已拉起时）
/home/research/krisl/soulx/venv/bin/python -I scripts/pvad_service_test.py   # sidecar 13 项
/home/research/krisl/soulx/venv/bin/python -I scripts/pvad_ws_test.py       # web 门控行为 11 项
/home/research/krisl/soulx/venv/bin/python -I scripts/bargein_storm_test.py  # 打断风暴回归（回声/噪声/真人抢话）
/home/research/krisl/soulx/venv/bin/python -I scripts/blip_turn_test.py     # 毛刺话轮过滤回归（幻听碎片丢弃/尾段不误杀）

# 快慢脑 Agent 测试（服务已拉起时）
/home/research/krisl/soulx/venv/bin/python -I scripts/agent_smoke.py          # 工具层 + 慢脑端到端 + Runtime 交付仲裁
/home/research/krisl/soulx/venv/bin/python -I scripts/agent_smoke.py --offline # 仅离线部分（不碰 LLM）
/home/research/krisl/soulx/venv/bin/python -I scripts/agent_e2e_test.py       # 全链路 E2E：委派/闲聊中交付/取消（约 10 分钟）
/home/research/krisl/soulx/venv/bin/python -I scripts/cancel_repro.py        # 快脑取消调用率探针（auto_cancel 设计依据）

# FRC 前端模式 E2E（turn_engine=frc；需 :8082/:8010/:8086 在线）
/home/research/krisl/soulx/venv/bin/python -I scripts/frc_mode_test.py       # 单话轮 + 句中停顿不切分，7 项

# 实时设置面板回归（update_config 边界 + 一次真实话轮）
/home/research/krisl/soulx/venv/bin/python -I scripts/config_ws_test.py

# PVAD 门控音频取证（复现幻听/误拒后）
# logs/pvad_forensics/<session8>/<时刻>_<reason>/{context.wav,window.wav,meta.json}
# context.wav = 房间里实际的声音（6s 原始）；window.wav = ECAPA 实际打分的窗口

# VoxCPM2 跨段音色一致性（LoRA + 锚定克隆；产出试听 wav）
/home/research/krisl/soulx/venv/bin/python -I scripts/voxcpm_consistency_test.py
```
