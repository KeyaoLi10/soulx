# Full-Duplex Spoken Dialogue System

*This directory contains the implementation of a full-duplex spoken dialogue system based on SoulX-Duplug.*

## Preparations
Make sure you are in the `SoulX-Duplug/dialogue_system` directory and all the model weights are downloaded to the `SoulX-Duplug/pretrained_models` folder.


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


## Run the Service

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

Visit http://localhost:8090 to chat with the full-duplex spoken dialogue system.

You can also replace the LLM+TTS component with any other half-duplex spoken dialogue model according to your needs.

## Code Layout

The web app (`app.py`, run by `deploy.sh`) is the orchestrator; cohesive layers
live in their own modules:

| Module | Responsibility |
|---|---|
| `app.py` | FastAPI wiring, WebSocket endpoint, turn-controller receive loop, speech pipeline (`pipeline_worker` / `_run_speech_pipeline`), session lifecycle |
| `settings.py` | `Config`, `DEFAULT_SETTINGS` (web UI panel defaults), `sanitize_updates()` validation |
| `emitter.py` | `emit_to_room()` — thread-safe WebSocket push onto the main event loop |
| `pvad_gate.py` | Main-speaker gate (ECAPA identity vs enrolled speaker) consulted at barge-in / turn completion |
| `frc_engine.py` | Optional FireRedChat-architecture turn engine (`turn_engine=frc`) |
| `agent/` | Fast/slow brain: `runtime.py` (AgentRuntime, delivery arbiter, auto-cancel/auto-delegate backstops), `slow_brain.py` (background agent loop), `tools.py` (web_search / fetch_page / python_repl / get_datetime) |
| `clients/` | HTTP/WS clients for the LLM, TTS, turn server (VAD) and pvad sidecar services |
| `frontend/` | Static web UI (no build step) |

`emitter` and `pvad_gate` must not import `app` back (circular); app wires the
two hooks they need (`emitter.set_session_lookup`, `pvad_gate.get_history`)
right after creating its singletons.



## Acknowledgment

We thank the following projects for their open-source contributions:

- [QwenLM](https://github.com/QwenLM)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Async CosyVoice](https://github.com/qi-hua/async_cosyvoice)
- [IndexTTS](https://github.com/index-tts/index-tts)
- [IndexTTS-vLLM](https://github.com/Ksuriuri/index-tts-vllm)
- [ChatTTS](https://github.com/2noise/ChatTTS)
- [X-Talk](https://github.com/xcc-zach/xtalk)
