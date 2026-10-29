#!/bin/bash
# Install the non-torch dependencies for the SoulX turn server + web app +
# offline_infer into soulx/venv (torch 2.6.0 installed separately).
set -e
V=/home/research/krisl/soulx/venv/bin/python
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_HTTP_TIMEOUT=60 UV_HTTP_MAX_RETRIES=10

uv pip install --python $V -i $IDX \
  "numpy==1.26.4" \
  "transformers==4.52.1" \
  "peft==0.16.0" \
  "pytorch-lightning==2.5.2" \
  "omegaconf==2.3.0" \
  "fastapi" "uvicorn" "websockets==12.0" \
  "websocket-client==1.8.0" \
  "soxr" "soundfile" "pyarrow" "requests" \
  "pynini==2.1.5" "WeTextProcessing==1.0.3" \
  "pypinyin" "zhon" "grpcio" \
  "modelscope==1.28.2" "funasr==1.2.6"

# quick import sanity check
$V -c "
import numpy, torch, transformers, peft, pytorch_lightning, omegaconf
import soxr, soundfile, pyarrow, websocket, requests
import modelscope, funasr
from tn.chinese.normalizer import Normalizer
print('imports OK: torch', torch.__version__, '| transformers', transformers.__version__)
"
