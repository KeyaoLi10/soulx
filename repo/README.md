<div align="center">
    <h1>
    SoulX-Duplug
    </h1>
    <p>
    Official code for enabling full-duplex speech interaction with<br>
    <b><em>SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation</em></b>
    </p>
    <p>
    <img src="assets/SoulX-Duplug-logo.png" alt="SoulX-Duplug Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <!-- <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/Python-3.10-blue" alt="version"></a> -->
    <a href="https://soulx-duplug.sjtuxlance.com/"><img src="https://img.shields.io/badge/🌐%20Online-Demo-blue" alt="Online Demo"></a>
    <a href="https://arxiv.org/abs/2603.14877"><img src="https://img.shields.io/badge/arXiv-2603.14877-B31B1B?logo=arxiv&logoColor=white.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="HF-Model"></a>
    <a href="https://huggingface.co/datasets/Soul-AILab/SoulX-Duplug-Eval"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Eval-yellow" alt="HF-Eval"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Duplug"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="Apache-2.0"></a>
</div>


# Full-Duplex Spoken Dialogue System

*This branch contains the implementation of a full-duplex spoken dialogue system based on SoulX-Duplug.*

## ⚙️ Preparations
Change directory to `SoulX-Duplug/dialogue_system` and make sure all the model weights are downloaded to the `SoulX-Duplug/pretrained_models` folder.


### Environment Setup
```bash
conda create -n dialogue-system -y python=3.10.16
conda activate dialogue-system
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt
```


### SoulX-Duplug
Install SoulX-Duplug according to the instructions in the [main README](https://github.com/Soul-AILab/SoulX-Duplug/blob/main/README.md).


### LLM
We utilize Qwen2.5-7B-Instruct as the LLM. Please download the model weights to `pretrained_models/Qwen2.5-7B-Instruct`.

```bash
huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir ../pretrained_models/Qwen2.5-7B-Instruct

# If you are in mainland China
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ../pretrained_models/Qwen2.5-7B-Instruct
```


### TTS
Currently, we utilize two excellent open-source projects as our TTS models: [IndexTTS-vLLM](https://github.com/Ksuriuri/index-tts-vllm) and [Async CosyVoice](https://github.com/qi-hua/async_cosyvoice).


- For IndexTTS-vLLM, please refer to [IndexTTS-vLLM](https://github.com/Ksuriuri/index-tts-vllm) for environment setup and model download.

    ```bash
    conda create -n index-tts-vllm python=3.12
    conda activate index-tts-vllm
    pip install -r modules/index_tts_vllm/requirements.txt
    modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ../pretrained_models/Index-TTS-1.5-vLLM
    ```

- For Async CosyVoice, you can refer to [Async CosyVoice](https://github.com/qi-hua/async_cosyvoice) for environment setup and model download.

    ```bash
    conda create -n cosyvoice2 python=3.10.16 -y
    conda activate cosyvoice2
    conda install -y -c conda-forge pynini==2.1.5
    pip install -r modules/CosyVoice/async_cosyvoice/requirements.txt
    huggingface-cli download --resume-download swulling/CosyVoice2-0.5B-vllm --local-dir ../pretrained_models/CosyVoice2-0.5B
    cp -r modules/CosyVoice/async_cosyvoice/CosyVoice2-0.5B/* ../pretrained_models/CosyVoice2-0.5B/
    cd modules/CosyVoice/async_cosyvoice/runtime/async_grpc
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
    ```


## 🚀 Run the Service

### Launch TTS Server
```bash
conda activate ...  # index-tts-vllm for IndexTTS-vLLM, cosyvoice2 for Async CosyVoice
bash scripts/tts_server.sh
```

### Launch LLM Server
```bash
conda activate soulx-duplug
bash scripts/llm_server.sh
```

### Launch VAD Server
```bash
conda activate soulx-duplug
bash scripts/vad_server.sh
```

### Launch Dialogue System
```bash
conda activate dialogue-system
bash deploy.sh
```

Visit `http://localhost:8090` to chat with the full-duplex spoken dialogue system.

You can also replace the LLM+TTS component with any other half-duplex spoken dialogue model according to your needs.


## 🧪 Evaluation

We provide offline inference code for simulating real-time interaction in `dialogue_system/offline_infer.py`, along with evaluation scripts in `dialogue_system/eval`.

The evaluation scripts are largely consistent with the [official Full-Duplex-Bench repository](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/bb595759b84934f81f1522e222e3976b51c94ff0), with the addition of a `get_transcript/asr_zh.py` script. The GPT-4o score is not evaluated for the *user_interruption* setting, as we focus primarily on turn management capability.

For benchmarking, we use the following config:
- `infer_config.max_wait_num: 5`
- `infer_config.far_field_threshold: 0`
- For Chinese evaluation, we use `paraformer`. The prompt for TTS is `dialogue_system/modules/index_tts_vllm/assets/ada-female.wav`
- For English evaluation, we use `sensevoice en`. The prompt for TTS is `dialogue_system/modules/index_tts_vllm/assets/john-male.wav`


## 📌 TODOs
- [x] Publish the technical report.
- [x] Release evaluation scripts.
- [x] Release training scripts.


## 🔖 Citation
If you find this work useful in your research, please consider citing:

```bibtex
@misc{yan2026soulxduplug,
      title={SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation}, 
      author={Ruiqi Yan and Wenxi Chen and Zhanxun Liu and Ziyang Ma and Haopeng Lin and Hanlin Wen and Hanke Xie and Jun Wu and Yuzhe Liang and Yuxiang Zhao and Pengchao Feng and Jiale Qian and Hao Meng and Yuhang Dai and Shunshun Yin and Ming Tao and Lei Xie and Kai Yu and Xinsheng Wang and Xie Chen},
      year={2026},
      eprint={2603.14877},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.14877}, 
}
```

## 📜 License
This project is licensed under the [Apache 2.0 License](LICENSE).


## 🙏 Acknowledgment
We thank the following open-source projects for their contributions:

- [QwenLM](https://github.com/QwenLM)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Async CosyVoice](https://github.com/qi-hua/async_cosyvoice)
- [IndexTTS](https://github.com/index-tts/index-tts)
- [IndexTTS-vLLM](https://github.com/Ksuriuri/index-tts-vllm)
- [ChatTTS](https://github.com/2noise/ChatTTS) 
- [X-Talk](https://github.com/xcc-zach/xtalk)
