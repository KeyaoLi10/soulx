# -*- coding: utf-8 -*-
"""VoxLock 离线混音评测（2026-11-10 说话人分离方案替换）.

背景：MossFormer2+ECAPA 影子方案失败——分离后说话人内方差仍大于说话
人间方差，分数带完全重叠（docs/PVAD背景视频失效调查报告-20261105.md
§5）。本评测用同样的 ECAPA 打分协议检验 VoxLock 是否真的拉开带宽。

素材（全部真实音频，ground truth 按构造成立）：
  - target（主说话人）: soulx 会话 c41a43ce 的用户长句
    （02:28:33-02:28:48, pcm offset 84-99s，web.log ASR 交叉验证）
  - enroll/seed 参考: 同一用户 offset 84-94s（10s 干净语音）
  - interferer A: 内核恐慌播客 kp_72 的 1 号说话人段（男声）
  - interferer B: 同会话里 Elva TTS 的回声（offset 17-31s 自我介绍，
    生产场景的真实背景人声形态）

时间轴（66s）：
  0-4 静音 | 4-14 干扰A only | 16-26 target only | 28-48 重叠
  50-60 干扰B only | 62-66 静音
两路各 RMS 归一到 0.06 后相加。

两步（numpy/onnx 与 torch 分进程，遵循本机 libomp 纪律）：
  step1 (soulx/venv):    构造混音 + VoxLock 隔离 → results/voxlock_mix_eval/*.npy
  step2 (venv-frc):      ECAPA 滑窗打分 mix vs isolated vs clean，输出带宽报告

Run:
  /home/research/krisl/soulx/venv/bin/python -I scripts/voxlock_mix_eval.py isolate
  /home/research/krisl/full_duplex/venv-frc/bin/python -I scripts/voxlock_mix_eval.py score
"""
import json
import sys
from pathlib import Path

import numpy as np

SR = 16000
SID = "c41a43ce-2284-4c3b-ab87-34effcfba7c6"
OUT = Path(__file__).resolve().parents[1] / "results" / "voxlock_mix_eval"

# --- sources (pcm offsets, seconds) ---
USER_REF = (83.0, 91.0)      # enroll/seed: clean user speech (8s)
USER_SPK = (91.0, 101.0)     # target speech in the mix — disjoint from the
                             # ref window so scoring isn't circular
ELVA_ECHO = (17.0, 27.0)     # interferer B: TTS echo in-session
POD_SEGS = [                 # interferer A: podcast speaker-1 segments
    (74.2, 84.2),
    (734.1, 744.1),
]
POD_PCM = Path("/home/research/krisl/eot/xyz_work/pcm/kp_72.npy")

# --- mix timeline (seconds) ---
T = [
    (4.0, 14.0, "interfererA"),   # podcast only
    (16.0, 26.0, "target"),       # user only
    (28.0, 38.0, "overlap"),      # user + podcast
    (42.0, 52.0, "interfererB"),  # Elva echo only
]
TOTAL_S = 56.0
RMS_TARGET = 0.06


def _rms(w):
    return float(np.sqrt((w.astype(np.float64) ** 2).mean())) if len(w) else 0.0


def _norm(w):
    r = _rms(w)
    return w / (r / RMS_TARGET) if r > 1e-4 else w


def _span(arr, a, b):
    return arr[int(a * SR): int(b * SR)]


def build_streams():
    """target / interferer streams on the shared timeline + class labels."""
    sid_pcm = np.fromfile(
        OUT.parents[1] / "logs" / "audio" / f"{SID}.mic.pcm",
        dtype=np.int16).astype(np.float32) / 32768.0
    pod = np.load(POD_PCM, mmap_mode="r")
    n = int(TOTAL_S * SR)
    target = np.zeros(n, np.float32)
    interf = np.zeros(n, np.float32)
    for a, b, kind in T:
        if kind == "interfererA":
            s = POD_SEGS[0]
            interf[int(a * SR): int(b * SR)] = _norm(_span(pod, *s))
        elif kind == "target":
            target[int(a * SR): int(b * SR)] = _norm(_span(sid_pcm, *USER_SPK))
        elif kind == "overlap":
            target[int(a * SR): int(b * SR)] = _norm(_span(sid_pcm, *USER_SPK))
            s = POD_SEGS[1]
            interf[int(a * SR): int(b * SR)] = _norm(_span(pod, *s))
        elif kind == "interfererB":
            interf[int(a * SR): int(b * SR)] = _norm(_span(sid_pcm, *ELVA_ECHO))
    ref = _norm(_span(sid_pcm, *USER_REF))
    mix = np.clip(target + interf, -1.0, 1.0).astype(np.float32)
    return ref, target, interf, mix


def window_class(t0, t1):
    """Timeline class of a scoring window [t0, t1)."""
    best, best_ov = "silent", 0.0
    for a, b, kind in T:
        ov = min(t1, b) - max(t0, a)
        if ov > best_ov:
            best, best_ov = kind, ov
    return best if best_ov >= 0.6 else "mixed"


def step_isolate():
    import voxlock

    ref, target, interf, mix = build_streams()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "ref.npy", ref)
    np.save(OUT / "target.npy", target)
    np.save(OUT / "interf.npy", interf)
    np.save(OUT / "mix.npy", mix)

    vl = voxlock.VoxLock(threads=4)
    stream = vl.stream(ref)          # seeded: isolates from the first block
    out = np.zeros_like(mix)
    n_chunks = len(mix) // 512
    import time
    t_start = time.time()
    for i in range(n_chunks):
        y = stream.push(mix[i * 512: (i + 1) * 512])
        if len(y):
            lo = i * 512
            out[lo: lo + len(y)] = y[: max(0, min(len(y), len(out) - lo))]
    # flush any residue
    tail = stream.push(np.zeros(0, np.float32)) if hasattr(stream, "push") else None
    dt = time.time() - t_start
    np.save(OUT / "isolated.npy", out)
    print(f"isolated {n_chunks * 512 / SR:.1f}s of audio in {dt:.1f}s "
          f"(RTF {dt / (n_chunks * 512 / SR):.2f})")
    print(f"out/in samples: {int((out != 0).sum())} nonzero / {len(mix)} "
          f"(stream object: {type(stream).__name__}, "
          f"locked={getattr(stream, 'locked', '?')})")


def step_score():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services"))
    from pvad_proc import PVADProcessor

    proc = PVADProcessor()

    def embed(wav):
        w = np.ascontiguousarray(wav[-80000:])
        e = proc._spk_extractor.get_embedding(torch.from_numpy(w).unsqueeze(0))
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(e)
        return e / n if n > 0 else e

    ref = np.load(OUT / "ref.npy")
    ref_e = embed(ref)
    streams = {name: np.load(OUT / f"{name}.npy")
               for name in ("mix", "isolated", "target")}
    WIN, STEP = 1.2, 0.8
    rows = []
    for name, wav in streams.items():
        for t0 in np.arange(0.0, TOTAL_S - WIN, STEP):
            w = _span(wav, t0, t0 + WIN)
            if _rms(w) < 0.005:
                continue                     # silent window: unscoreable
            cos = float(np.dot(embed(w), ref_e))
            rows.append({"stream": name, "t": round(t0, 2),
                         "class": window_class(t0, t0 + WIN),
                         "cos": round(cos, 3)})
    print(f"\n{'class':<14}{'stream':<10}{'n':>3}  {'mean':>6}  {'min':>6}  {'max':>6}")
    verdict = {}
    for cls in ("interfererA", "interfererB", "overlap", "target"):
        line = {}
        for name in ("mix", "isolated", "target"):
            xs = [r["cos"] for r in rows
                  if r["stream"] == name and r["class"] == cls]
            if not xs:
                continue
            line[name] = (len(xs), float(np.mean(xs)),
                          float(np.min(xs)), float(np.max(xs)))
            print(f"{cls:<14}{name:<10}{len(xs):>3}  "
                  f"{np.mean(xs):6.3f}  {min(xs):6.3f}  {max(xs):6.3f}")
        verdict[cls] = {k: [round(v, 3) for v in vv] for k, vv in line.items()}
        print()
    (OUT / "score_rows.json").write_text(
        json.dumps({"rows": rows, "summary": verdict}, ensure_ascii=False, indent=1))
    print("saved", OUT / "score_rows.json")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    if step == "isolate":
        step_isolate()
    elif step == "score":
        step_score()
    else:
        sys.exit(f"unknown step {step!r} (isolate|score)")
