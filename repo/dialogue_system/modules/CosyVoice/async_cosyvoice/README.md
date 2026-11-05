<div align="center">

# Async CosyVoice
</div>

## 项目概述
Async CosyVoice 项目用于加速 cosyvoice2 的推理过程，当前仅支持 Linux 系统，并且依赖 vllm 库。以下是该项目的主要特性：
1. **LLM 推理加速**：借助 vllm 对 LLM 部分的推理进行加速。
2. **Flow 推理情况**：Flow 部分的推理采用官方的 `load_jit` 或 `load_trt` 模式，并使用[hexisyztem](https://github.com/hexisyztem)提供的多estimator实例加速。
3. **加速表现**：
   - 单任务推理的 RTF 从原来的 0.25 - 0.30，经过 vllm 加速后可达到 0.1 - 0.15。
   - 单任务流式推理场景下，首包延迟约在 150 - 250ms 之间。
   - 并发推理时，在rtf<1的前提下，4070能够支持 20 个非流式并发，或者 10 个流式并发。

## 环境准备
请使用 Python 3.10.16 版本，按以下步骤创建并激活 Conda 环境：
```bash
conda create -n cosyvoice2 python=3.10.16 -y
conda activate cosyvoice2
```

## 使用步骤
### 1. 克隆 CosyVoice 项目
```bash
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 安装系统依赖
conda install -y -c conda-forge pynini==2.1.5
sudo apt-get update
sudo apt-get install sox libsox-dev -y
```

### 2. 克隆本项目
在 CosyVoice 项目路径下，克隆本项目：
```bash
git clone https://github.com/qi-hua/async_cosyvoice.git
```

### 3. 安装依赖
进入 `async_cosyvoice` 目录，安装所有依赖（不用再安装原CosyVoice项目依赖）：
```bash
cd async_cosyvoice
pip install -r requirements.txt
```

### 4. 下载模型文件
从 [这里](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B/) 下载 CosyVoice2 - 0.5B 模型文件，并将其保存至 `CosyVoice/pretrained_models` 目录。

### 5. 复制文件
将 `async_cosyvoice/CosyVoice2 - 0.5B` 文件夹下的文件复制到下载的 CosyVoice2 - 0.5B 模型文件夹中：
```bash
cp async_cosyvoice/CosyVoice2-0.5B/* pretrained_models/CosyVoice2-0.5B/
```

### 6. 配置参数
可在 `config.py` 文件中设置 vllm 的 `AsyncEngineArgs` 参数和 `SamplingParams` 参数，以及 ESTIMATOR_COUNT：
```python
ENGINE_ARGS = {
    # 根据实际情况设置 
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 1024,
    "max_model_len": 2048,
    "max_num_seqs": 256,
}

SAMPLING_PARAMS = {
    "max_tokens": 2048,
}

ESTIMATOR_COUNT = 2
```

### 7. GRPC 服务使用
在 `runtime/async_grpc` 目录下，运行以下命令启动 GRPC 服务：
```bash
# 1. 由 proto 文件生成依赖代码
cd runtime/async_grpc
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
python server.py --load_jit --load_trt --fp16
```
测试 GRPC 服务
```bash
python client.py
```

### 8. FastAPI 服务使用
在 `runtime/fastapi` 目录下，运行以下命令启动 FastAPI 服务：
```bash
cd runtime/fastapi
python server.py --load_jit --load_trt --fp16
```

### 9. spk2info 说明
`spk2info.pt` 文件中保存了 `prompt_text` 及其 token、embedding 数据，可用于 `sft`、`inference_zero_shot_by_spk_id`、`inference_instruct2_by_spk_id`，使用时无需传递参考 `prompt_text` 及音频数据，直接传递 `spk_id` 即可，并可以跳过对参考音频的预处理步骤，直接进入推理步骤。

可通过 `frontend.generate_spk_info` 函数生成新的 `spk2info`，需要传入的参数为：`spk_id: str`、`prompt_text: str`、`prompt_speech_16k: torch.Tensor`、`resample_rate: int = 24000`、`name: str = None`。


## 注意事项
1. 本项目使用的是 vllm == 0.7.3，并开启了 `VLLM_USE_V1 = '1'`。
2. 当前 CUDA 环境为 12.4，部分依赖文件使用的版本为 `vllm==0.7.3`、`torch==2.5.1`、`onnxruntime-gpu==1.19.0`。
3. 启动 cosyvoice2 实例后，需要进行 **预热**，即进行 10 次以上的推理任务，预热 trt 模型。