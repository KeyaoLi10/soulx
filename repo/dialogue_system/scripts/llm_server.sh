script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname "$script_dir")

cd ${root_dir}/modules/qwen_llm
python llm_server.py \
    --host 0.0.0.0 \
    --port 6007 \
    --model_dir ${root_dir}/../pretrained_models/Qwen2.5-7B-Instruct