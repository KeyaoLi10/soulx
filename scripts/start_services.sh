#!/bin/bash
# Start all SoulX-Duplug dialogue-system services in tmux session "soulx".
#
#   GPU 5  turn-taking server  :8000   (SoulX-Duplug 0.6B + paraformer, soulx/venv)
#   GPU 6  LLM+TTS fused server:6007   (Qwen3.5-9B + FireRedTTS client, venv-frc-llm)
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
tmux send-keys -t soulx:voxtts "CUDA_VISIBLE_DEVICES=7 VOXCPM_MODEL_PATH=/mnt/disk3/krisl/soulx/VoxCPM2 LD_LIBRARY_PATH=/mnt/disk1/miniconda3/envs/tempo/lib $SOULX/venv-voxcpm/bin/python $SOULX/services/voxcpm_tts_server.py --port 8085 2>&1 | tee $LOGS/voxcpm_tts.log" C-m

tmux new-window -t soulx -n llm
tmux send-keys -t soulx:llm "CUDA_VISIBLE_DEVICES=6 $FD/venv-frc-llm/bin/python $SOULX/services/qwen35_fused_server.py --port 6007 2>&1 | tee $LOGS/llm.log" C-m

tmux new-window -t soulx -n turn
tmux send-keys -t soulx:turn "cd $SOULX/repo && CUDA_VISIBLE_DEVICES=5 MODELSCOPE_CACHE=$SOULX/modelscope_cache $SOULX/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1 2>&1 | tee $LOGS/turn.log" C-m

tmux new-window -t soulx -n pvad
# HF_HUB_OFFLINE: the plugin's ECAPA weights resolve from the local
# resources dir; going online makes speechbrain try an HF fetch.
tmux send-keys -t soulx:pvad "HF_HUB_OFFLINE=1 $FD/venv-frc/bin/python $SOULX/services/pvad_server.py --port 8086 2>&1 | tee $LOGS/pvad.log" C-m

tmux new-window -t soulx -n web
tmux send-keys -t soulx:web "cd $SOULX/repo/dialogue_system && $SOULX/venv/bin/python app.py 2>&1 | tee $LOGS/web.log" C-m

echo "started tmux session 'soulx' (windows: tts voxtts llm turn pvad web)"
echo "wait for models to load, then check: curl -s localhost:8081/health; curl -s localhost:8085/health; curl -s localhost:6007/health; curl -s localhost:8086/health"
