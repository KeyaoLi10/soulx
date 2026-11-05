script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname "$script_dir")

cd ${root_dir}/modules/index_tts_vllm
python api_server.py \
    --host 0.0.0.0 \
    --port 6006 \
    --model_dir ${root_dir}/../pretrained_models/Index-TTS-1.5-vLLM

# cd ${root_dir}/modules/CosyVoice/async_cosyvoice/runtime/async_grpc
# python server.py --load_jit --load_trt --fp16 \
#     --port 6006 \
#     --model_dir ${root_dir}/../pretrained_models/CosyVoice2-0.5B