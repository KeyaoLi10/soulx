from io import BytesIO
import numpy as np
import torch
import torchaudio

def convert_audio_bytes_to_tensor(raw_audio: bytes) -> torch.Tensor:
    """同步音频转换方法"""
    return torch.from_numpy(np.array(np.frombuffer(raw_audio, dtype=np.float32))).unsqueeze(0)

def convert_audio_tensor_to_bytes(tensor: torch.Tensor, format: str = None, sample_rate=24000) -> bytes:
    """将音频Tensor转换为指定格式的字节流

    Args:
        tensor: 输入音频Tensor，形状需为 (channels, samples) 或 (samples,)
        format: 目标格式，支持 'wav', 'pcm', 'mp3'
        sample_rate: 采样率（默认16000）

    Returns:
        bytes: 编码后的音频字节流
    """
    if not format:
        return tensor.numpy().astype(np.float32).tobytes()
    # 统一Tensor形状为 (channels, samples)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        if tensor.size(0) > tensor.size(1):  # 假设输入为 (samples, channels)
            tensor = tensor.permute(1, 0)
    else:
        raise ValueError("Invalid tensor shape")

    match format:
        case "wav":
            # bits_per_sample: PCM/WAV的量化位数（16或32）
            return _encode_wav(tensor, sample_rate, 16)
        case "pcm":
            return _encode_pcm(tensor, 16)
        case "mp3":
            return _encode_mp3(tensor, sample_rate)
        case _:
            raise ValueError(f"Unsupported format: {format}")

def _encode_wav(tensor: torch.Tensor, sr: int, bits: int) -> bytes:
    """编码WAV格式"""
    if bits == 16:
        encoding = "PCM_S"
        bits_depth = 16
    elif bits == 32:
        encoding = "PCM_F"
        bits_depth = 32
    else:
        raise ValueError("Only 16/32-bit WAV supported")
    buffer = BytesIO()
    torchaudio.save(
        buffer,
        tensor,
        sr,
        format="wav",
        encoding=encoding,
        bits_per_sample=bits_depth,
    )
    buffer.seek(0)
    return buffer.getvalue()

def _encode_pcm(tensor: torch.Tensor, bits: int) -> bytes:
    """编码原始PCM数据"""
    assert tensor.dtype == torch.float32 or tensor.dtype == torch.float64, "输入张量应为浮点类型"
    np_array = tensor.cpu().numpy()
    np_array = np.clip(np_array, -1.0, 1.0)

    # 量化到目标位深
    if bits == 16:
        np_array = (np_array * 32767.0).astype(np.int16)
    elif bits == 32:
        np_array = (np_array * 2147483647.0).astype(np.int32)
    else:
        raise ValueError("Only 16/32-bit PCM supported")
    return np_array.tobytes()

def _encode_mp3(tensor: torch.Tensor, sr: int) -> bytes:
    """编码MP3格式"""
    buffer = BytesIO()

    # 注意：需要安装支持MP3编码的后端（如ffmpeg, libsox）
    torchaudio.save(
        buffer,
        tensor,
        sr,
        format="mp3",
        encoding="MP3",
    )
    buffer.seek(0)
    return buffer.getvalue()