#!/bin/bash
# Start all SoulX-Duplug dialogue-system services in tmux session "soulx".
#
#   GPU 5  turn-taking server  :8000   (SoulX-Duplug 0.6B + paraformer, soulx/venv)
#   GPU 5  ultraVAD EOT        :8082   (eot_bench/venv — transformers 4.57.1; only
#                                       used by the turn_engine="frc" frontend mode)
#   GPU 5  FireRedASR          :8010   (full_duplex/venv; ditto, frc mode only)
#   GPU 6  LLM+TTS fused server:6007   (Qwen3.5-9B + FireRedTTS client, venv-frc-llm;
#                                       /chat /agent_chat /v1/score — score is the
#                                       frc dual-EOT semantic channel)
#   GPU 7  FireRedTTS          :8081   (full_duplex/venv; kept for the full_duplex
#                                       FRC agent and as TTS_ENGINE=firered rollback)
#   GPU 7  VoxCPM2 TTS         :8085   (soulx/venv-voxcpm; web app default TTS)
#   -      web app             :8090   (soulx/venv; TTS_ENGINE=voxcpm|firered)
#   -      pvad sidecar        :8086   (CPU; full_duplex/venv-frc — the only env
#                                       with the fireredchat-pvad plugin's
#                                       speechbrain/onnxruntime stack; cross-venv
#                                       pattern like ultraVAD in eot_bench)
#
# Logs under /home/research/krisl/soulx/logs/
set -u
SOULX=/home/research/krisl/soulx
FD=/home/research/krisl/full_duplex
LOGS=$SOULX/logs

tmux has-session -t soulx 2>/dev/null && { echo "tmux session 'soulx' already exists"; exit 1; }

tmux new-session -d -s soulx -n tts
tmux send-keys -t soulx:tts "CUDA_VISIBLE_DEVICES=7 $FD/venv/bin/python $FD/frc_services/tts/server.py --port 8081 2>&1 | tee $LOGS/tts.log" C-m

tmux new-window -t soulx -n voxtts
# LD_LIBRARY_PATH: torchcodec needs ffmpeg 4/5/6/7 libs; the box has none
# system-wide, tempo conda env ships them (torchcodec 0.1.1 pairs with torch 2.6/cu124)
# VOXCPM_LORA_PATH: self-trained VoxCPM2 LoRA (r=32); voice anchored by
# reference cloning — see services/voxcpm_tts_server.py header
tmux send-keys -t soulx:voxtts "CUDA_VISIBLE_DEVICES=7 VOXCPM_MODEL_PATH=/mnt/disk3/krisl/soulx/VoxCPM2 VOXCPM_LORA_PATH=/vepfs-cnbjc4e12c6d7e1f/krisl/step_0000500 LD_LIBRARY_PATH=/mnt/disk1/miniconda3/envs/tempo/lib $SOULX/venv-voxcpm/bin/python $SOULX/services/voxcpm_tts_server.py --port 8085 2>&1 | tee $LOGS/voxcpm_tts.log" C-m

tmux new-window -t soulx -n llm
tmux send-keys -t soulx:llm "CUDA_VISIBLE_DEVICES=6 $FD/venv-frc-llm/bin/python $SOULX/services/qwen35_fused_server.py --port 6007 2>&1 | tee $LOGS/llm.log" C-m

# Slow brain: second Qwen3.5-9B instance (thinking=on) for background tasks
tmux new-window -t soulx -n slowllm
tmux send-keys -t soulx:slowllm "CUDA_VISIBLE_DEVICES=1 $FD/venv-frc-llm/bin/python $SOULX/services/qwen35_fused_server.py --port 6008 2>&1 | tee $LOGS/slowllm.log" C-m

tmux new-window -t soulx -n turn
tmux send-keys -t soulx:turn "cd $SOULX/repo && CUDA_VISIBLE_DEVICES=5 MODELSCOPE_CACHE=$SOULX/modelscope_cache $SOULX/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 2>&1 | tee $LOGS/turn.log" C-m

tmux new-window -t soulx -n ultravad
# frc-frontend-mode EOT audio channel. Same cross-venv pattern as the pvad
# sidecar: the eot_bench venv is the only one whose transformers (4.57.1)
# runs the fixie-ai/ultraVAD custom code.
tmux send-keys -t soulx:ultravad "cd $FD && CUDA_VISIBLE_DEVICES=5 HF_HOME=/home/research/krisl/eot/eot_bench/hf HF_HUB_OFFLINE=1 /home/research/krisl/eot/eot_bench/venv/bin/python $FD/frc_services/ultravad/server.py --port 8082 2>&1 | tee $LOGS/ultravad.log" C-m

tmux new-window -t soulx -n frcasr
# frc-frontend-mode utterance transcription (file-upload API, called once
# per committed turn). Port 8010: 8000 is the SoulX turn server and 8003
# belongs to another user's long-running proxy. PATH: load_audio shells out
# to ffmpeg and the box has no system ffmpeg — the tempo conda env has one
# (same env that provides the voxtts ffmpeg libs).
tmux send-keys -t soulx:frcasr "cd $FD/frc_services/asr && PATH=/mnt/disk1/miniconda3/envs/tempo/bin:\$PATH FIREREDASR_PATH=$FD/frc_services/asr/fireredasr MODEL_DIR=$FD/frc_services/asr/models CUDA_VISIBLE_DEVICES=5 $FD/venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8010 2>&1 | tee $LOGS/frcasr.log" C-m

tmux new-window -t soulx -n pvad
# HF_HUB_OFFLINE: the plugin's ECAPA weights resolve from the local
# resources dir; going online makes speechbrain try an HF fetch.
tmux send-keys -t soulx:pvad "HF_HUB_OFFLINE=1 $FD/venv-frc/bin/python $SOULX/services/pvad_server.py --port 8086 2>&1 | tee $LOGS/pvad.log" C-m

tmux new-window -t soulx -n web
tmux send-keys -t soulx:web "cd $SOULX/repo/dialogue_system && $SOULX/venv/bin/python app.py 2>&1 | tee $LOGS/web.log" C-m

echo "started tmux session 'soulx' (windows: tts voxtts llm slowllm turn pvad web)"
echo "wait for models to load, then check: curl -s localhost:8081/health; curl -s localhost:8085/health; curl -s localhost:6007/health; curl -s localhost:6008/health; curl -s localhost:8086/health"
