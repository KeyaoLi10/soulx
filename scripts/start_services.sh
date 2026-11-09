#!/bin/bash
# Start all SoulX-Duplug dialogue-system services in tmux session "soulx".
#
#   GPU 5  turn-taking server  :8000   (SoulX-Duplug 0.6B + paraformer, soulx/venv)
#   GPU 3  LLM fused server    :6007   (Qwen3.5-9B, venv-frc-llm; /chat /agent_chat)
#   GPU 1  slow brain LLM      :6008   (Qwen3.5-9B thinking=on, venv-frc-llm)
#   GPU 6  router LLM          :6009   (Qwen3.5-9B greedy, venv-frc-llm)
#   GPU 7  VoxCPM2 TTS         :8085   (soulx/venv-voxcpm; dual LoRA checkpoints,
#                                       see services/voxcpm_tts_server.py)
#   -      web app             :8090   (soulx/venv; 启动须内联
#                                       AGENT_RUNTIME_MODE=enforce
#                                       AGENT_CONTROL_ENDPOINT=http://localhost:6009/agent_chat_v2)
#   -      pvad sidecar        :8086   (CPU; full_duplex/venv-frc — the only env
#                                       with speechbrain/onnxruntime; code+weights
#                                       are in-repo: services/pvad_proc.py +
#                                       models/pvad/)
#
# 2026-11-09 退役：FireRedTTS (:8081)、frc 双 EOT 引擎（ultraVAD :8082、
# FireRedASR :8010）——SoulX 是唯一 turn engine，VoxCPM2 是唯一 TTS。
#
# Logs under /home/research/krisl/soulx/logs/
set -u
SOULX=/home/research/krisl/soulx
FD=/home/research/krisl/full_duplex
LOGS=$SOULX/logs

tmux has-session -t soulx 2>/dev/null && { echo "tmux session 'soulx' already exists"; exit 1; }

tmux new-session -d -s soulx -n voxtts
# LD_LIBRARY_PATH: torchcodec needs ffmpeg 4/5/6/7 libs; the box has none
# system-wide, tempo conda env ships them (torchcodec 0.1.1 pairs with torch 2.6/cu124)
# VOXCPM_LORA_PATH_500/400: self-trained VoxCPM2 LoRA (r=32) checkpoints, both
# loaded, selected per request — see services/voxcpm_tts_server.py header
tmux send-keys -t soulx:voxtts "CUDA_VISIBLE_DEVICES=7 VOXCPM_MODEL_PATH=/mnt/disk3/krisl/soulx/VoxCPM2 VOXCPM_LORA_PATH_500=/vepfs-cnbjc4e12c6d7e1f/krisl/step_0000500 VOXCPM_LORA_PATH_400=/vepfs-cnbjc4e12c6d7e1f/krisl/step_0000400 LD_LIBRARY_PATH=/mnt/disk1/miniconda3/envs/tempo/lib $SOULX/venv-voxcpm/bin/python $SOULX/services/voxcpm_tts_server.py --port 8085 2>&1 | tee $LOGS/voxcpm_tts.log" C-m

tmux new-window -t soulx -n llm
tmux send-keys -t soulx:llm "CUDA_VISIBLE_DEVICES=3 $FD/venv-frc-llm/bin/python $SOULX/services/qwen35_fused_server.py --port 6007 2>&1 | tee $LOGS/llm.log" C-m

# Slow brain: second Qwen3.5-9B instance (thinking=on) for background tasks
tmux new-window -t soulx -n slowllm
tmux send-keys -t soulx:slowllm "CUDA_VISIBLE_DEVICES=1 $FD/venv-frc-llm/bin/python $SOULX/services/qwen35_fused_server.py --port 6008 2>&1 | tee $LOGS/slowllm.log" C-m

# Router: third instance (greedy) for agent-V2 route_turn
tmux new-window -t soulx -n router
tmux send-keys -t soulx:router "CUDA_VISIBLE_DEVICES=6 $FD/venv-frc-llm/bin/python $SOULX/services/qwen35_fused_server.py --port 6009 2>&1 | tee $LOGS/router.log" C-m

tmux new-window -t soulx -n turn
tmux send-keys -t soulx:turn "cd $SOULX/repo && CUDA_VISIBLE_DEVICES=5 MODELSCOPE_CACHE=$SOULX/modelscope_cache $SOULX/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 2>&1 | tee $LOGS/turn.log" C-m

tmux new-window -t soulx -n pvad
# HF_HUB_OFFLINE: the ECAPA weights resolve from the in-repo models/pvad dir;
# going online makes speechbrain try an HF fetch.
tmux send-keys -t soulx:pvad "HF_HUB_OFFLINE=1 $FD/venv-frc/bin/python $SOULX/services/pvad_server.py --port 8086 2>&1 | tee $LOGS/pvad.log" C-m

tmux new-window -t soulx -n web
tmux send-keys -t soulx:web "cd $SOULX/repo/dialogue_system && AGENT_RUNTIME_MODE=enforce AGENT_CONTROL_ENDPOINT=http://localhost:6009/agent_chat_v2 $SOULX/venv/bin/python app.py 2>&1 | tee $LOGS/web.log" C-m

echo "started tmux session 'soulx' (windows: voxtts llm slowllm router turn pvad web)"
echo "wait for models to load, then check: curl -s localhost:8085/health; curl -s localhost:6007/health; curl -s localhost:6008/health; curl -s localhost:8086/health"
