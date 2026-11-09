"""Self-contained pvad processor for the SoulX repo (2026-11-09 整合).

FireRedChat 的 fireredchat-pvad livekit 插件里，本项目实际用到的只有
两块：personal-VAD 的 ONNX 会话（pvad.onnx，逐 10ms 帧出"是注册说话人
在说话"的概率）和 speechbrain ECAPA 说话人嵌入（spkrec-ecapa-voxceleb）。
本模块把这两块从 livekit 运行时里剥出来（原实现：full_duplex/
fireredchat_ref/FireRedChat/agents/fireredchat-plugins/
livekit-plugins-fireredchat-pvad/.../vad.py 的 SpeakerEmbExtractor 与
VadProcessor），权重搬进本仓库 models/pvad/（gitignored），从此 soulx
不再依赖 full_duplex 的插件目录树。

数值行为与原实现逐位一致：同样的 ONNX SessionOptions（关 spinning、
4 线程、SEQUENTIAL）、同样的输入形状与输出口、同样的 ECAPA L2 归一化。

运行环境仍用 full_duplex/venv-frc——它是本机唯一带 onnxruntime +
speechbrain 的环境（torch CPU 栈）；这是解释器依赖而非代码依赖，权重
与逻辑都已在本仓库内。PVAD_MODEL_DIR 可覆盖权重目录（默认
<soulx>/models/pvad/）。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from speechbrain.inference.speaker import EncoderClassifier

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "pvad"


class SpeakerEmbExtractor:
    """speechbrain ECAPA（voxceleb）说话人嵌入，输出 L2 归一化的 (1,192)。"""

    def __init__(self, ckpt_path=None, device="cpu"):
        if ckpt_path is None:
            ckpt_path = str(_DEFAULT_MODEL_DIR / "spkrec-ecapa-voxceleb") + "/"
        self.device = device
        if device == "cpu":
            device_str = "cpu"
        else:
            device_str = f"cuda:{device}"
        self.classifier = EncoderClassifier.from_hparams(
            source=ckpt_path, savedir=ckpt_path, run_opts={"device": device_str}
        )

    def get_embedding(self, input_signal):
        embeddings = self.classifier.encode_batch(input_signal)[0][0].detach()
        embeddings = embeddings / embeddings.norm(p=2, dim=0, keepdim=True)
        return embeddings.cpu().unsqueeze(0).numpy()


class PVADProcessor:
    """personal-VAD ONNX 逐帧推理（与原插件的 VadProcessor 同构）。

    状态就是三个 numpy 缓冲（mel/gru/spkemb）——会话本身无状态，宿主
    （pvad_server）在调用前后换入换出即可实现多会话时间片，与原实现
    完全一致。
    """

    def __init__(self, model_dir=None):
        model_dir = Path(model_dir or os.environ.get("PVAD_MODEL_DIR")
                         or _DEFAULT_MODEL_DIR)
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_dir / "pvad.onnx"),
            providers=["CPUExecutionProvider"], sess_options=opts)
        self._spk_extractor = SpeakerEmbExtractor(
            str(model_dir / "spkrec-ecapa-voxceleb"))
        self.window_size_samples = 160
        self.mel_buffer = np.zeros((1, 80, 15), dtype=np.float32)
        self.gru_buffer = np.zeros((2, 1, 256), dtype=np.float32)
        self.spkemb = np.zeros((1, 192), dtype=np.float32)
        self.sample_rate = 16000

    def reset(self):
        self.mel_buffer = np.zeros((1, 80, 15), dtype=np.float32)
        self.gru_buffer = np.zeros((2, 1, 256), dtype=np.float32)
        self.spkemb = np.zeros((1, 192), dtype=np.float32)

    def __call__(self, wav_np):
        if wav_np.shape[0] != 160:
            return 0.0
        wav_np = wav_np.reshape((1, 160))

        outputs = self._session.run(None, {
            'input_audio': wav_np,
            'spkemb': self.spkemb,
            'mel_buffer': self.mel_buffer,
            'gru_buffer': self.gru_buffer
        })
        raw_prob = outputs[1][0].tolist()[0]
        self.mel_buffer = outputs[2]   # (1, 80, 15)
        self.gru_buffer = outputs[3]   # (2, 1, 256)
        return raw_prob
