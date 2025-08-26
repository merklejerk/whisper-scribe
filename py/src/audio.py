from __future__ import annotations

import base64
import math
from typing import Tuple

import numpy as np
from scipy.signal import resample_poly

TARGET_SR = 16000


def _decode_to_float32(buf: bytes, channels: int, sample_width: int) -> np.ndarray:
    """Decode raw PCM buffer to float32 mono array in -1..1.

    Supports:
    - 8-bit unsigned PCM (sample_width=1)
    - 16-bit signed little-endian PCM (sample_width=2)
    - 32-bit float little-endian PCM (sample_width=4)
    - 32-bit signed little-endian PCM (sample_width=4, heuristically detected)
    """
    if sample_width == 1:
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(buf, dtype='<i2').astype(np.float32) / 32768.0
    elif sample_width == 4:
        # Try float32 first
        f = np.frombuffer(buf, dtype='<f4').astype(np.float32)
        # Heuristic: if values are way outside [-1, 1], treat as int32 PCM
        max_abs = float(np.max(np.abs(f))) if f.size else 0.0
        if max_abs > 16.0:
            i = np.frombuffer(buf, dtype='<i4').astype(np.float32)
            arr = i / 2147483648.0
        else:
            arr = f
    else:
        raise ValueError(f"unsupported sample_width: {sample_width}")

    if channels <= 0:
        raise ValueError(f"invalid channels: {channels}")
    if channels == 1:
        return arr
    # Interleaved channels -> mono
    if arr.size % channels != 0:
        # Truncate incomplete frame to avoid reshape error
        arr = arr[: arr.size - (arr.size % channels)]
    frames = arr.reshape((-1, channels))
    mono = frames.mean(axis=1)
    return mono.astype(np.float32, copy=False)


def _resample_float32(x: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    if src_sr <= 0:
        raise ValueError(f"invalid sample rate: {src_sr}")
    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    y = resample_poly(x, up, down).astype(np.float32, copy=False)
    return y


def normalize_to_mono16k(data: bytes, sr: int, channels: int, sample_width: int) -> bytes:
    """Decode PCM and convert to mono 16kHz PCM16 little-endian bytes.

    Raises ValueError on unsupported formats.
    """
    x = _decode_to_float32(data, channels=channels, sample_width=sample_width)
    x = _resample_float32(x, src_sr=sr, dst_sr=TARGET_SR)
    # Clip to [-1,1] and convert to int16 little-endian
    y = np.clip(x, -1.0, 1.0)
    pcm16 = (y * 32768.0).astype('<i2', copy=False).tobytes()
    return pcm16
