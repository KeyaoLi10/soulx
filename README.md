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
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ LLM+TTS 融合服务  http://localhost:6007      (GPU 6)   │
        │  Qwen3.5-9B 流式生成 → 按标点切段 ─┐                    │
        │  POST /chat           纯文本流    │ 每段调用           │
        │  POST /chat_indextts  S/B 分块流 ◄┘ FireRedTTS(:8081)  │
        │  b"S"+UTF-8 文本 / b"B"+int16 PCM@24kHz               │
        └──────────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   Web 实时对话 app (:8090)                offline_infer.py 离线推理
   VADModelPool(10 路 TurnTaking            1x 实时流式回放 input.wav，
   ws 客户端池)+SessionManager+             打断模拟/音频对齐渲染，
   pipeline_worker                          产出对话音频与状态时间线
   每段回复 → VoxCPM2 TTS(:8085) → 24kHz PCM
   mic 旁路 → PVAD sidecar(:8086,CPU)     ← 异步 feed,不进轮次路径
```

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
| `llm` | 融合 LLM 服务（`services/qwen35_fused_server.py`） | 6 | 6007 | `full_duplex/venv-frc-llm` |
| `turn` | SoulX turn server（`repo/` uvicorn server:app） | 5 | 8000 | `soulx/venv` |
| `pvad` | **PVAD 主说话人门控 sidecar（`services/pvad_server.py`，CPU）** | – | 8086 | `full_duplex/venv-frc` |
| `web` | Web 演示应用（`repo/dialogue_system/app.py`，`TTS_ENGINE=voxcpm\|firered`） | – | 8090 | `soulx/venv` |

`tmux attach -t soulx` 查看；日志在 `logs/*.log`。TTS 回滚：`TTS_ENGINE=firered`
重启 web 窗口即可。voxtts 启动需 `LD_LIBRARY_PATH=/mnt/disk1/miniconda3/envs/tempo/lib`
(torchcodec 要 ffmpeg 库)，见 `start_services.sh` 内注释。

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
2. **TTS**：IndexTTS/CosyVoice → FireRedTTS → **VoxCPM2（2026-10-27 起 web 对话默认）**。
   `clients/tts_client.py` 含 `FireRedTTS` 与 `VoxCPM_TTS` 两个等接口类，公共文本
   清洗在 `_clean_tts_text`；`app.py` 由 `TTS_ENGINE` env 选择（默认 voxcpm，
   回滚设 firered）。VoxCPM2 走独立微服务 `services/voxcpm_tts_server.py`（:8085，
   venv-voxcpm）：48kHz 生成经 soxr 流式重采样到 24kHz 裸 PCM 流，实测首块
   0.06s、RTF 0.32；音色由 `VOXCPM_VOICE_DESC` env 配置（voice design 文本前缀）。
   FireRedTTS 服务本身未动（full_duplex 共用）。详见 `docs/修改日志.md` 2026-10-27 条。
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
6. **配置**：`config/config.yaml` 按原 README 基准 `max_wait_num: 5`、
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
```
