from __future__ import annotations
"""Silero VAD wrapper.

Encapsulates model loading, device placement, and 32 ms (512-sample @ 16 kHz)
frame-wise speech probability estimation.
"""
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import torch
from silero_vad import load_silero_vad

from .devices import resolve_device


SAMPLE_RATE = 16000
FRAME_MS = 32
FRAME_SAMPLES = int(SAMPLE_RATE * (FRAME_MS / 1000.0))  # 512


class InsufficientSamplesError(Exception):
    """Raised when the provided buffer doesn't contain enough samples for one 32 ms frame."""
    pass


@dataclass
class SileroVAD:
    """Thin convenience wrapper around silero-vad.

    Usage: create once per process and call max_prob on PCM16 arrays.
    """

    device: str
    _model: Any  # silero_vad may return various callable wrappers (PyTorch/ONNX)

    @classmethod
    def create(cls, device_name: Optional[str] = None) -> "SileroVAD":
        dev = device_name or resolve_device().name
        model = load_silero_vad()
        if hasattr(model, "to"):
            model = model.to(dev)  # type: ignore[attr-defined]
        return cls(device=dev, _model=model)


    def analyze(
        self,
        pcm16: np.ndarray,
        sample_rate: int,
        window_s: float,
        threshold: float,
        *,
        keep_context_ms: int = 96,
        min_consecutive: int = 3,
    ) -> tuple[int, float]:
        """Unified analysis in one pass: (drop_leading_samples, max_prob_tail_window).

        - Frames the entire buffer into 32 ms, computes per-frame probabilities once.
        - Leading-silence drop is based on consecutive low-prob frames from start,
          keeping `keep_context_ms` worth of frames.
        - Tail max is the max probability over the last `window_s` seconds.

        Raises:
        - TypeError/ValueError for invalid inputs
        - InsufficientSamplesError if the tail window contains no complete frame
        """
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        if pcm16.ndim != 1:
            raise TypeError("pcm16 must be a 1-D array")
        if not (pcm16.dtype.kind == 'i' and pcm16.dtype.itemsize == 2):
            raise TypeError("pcm16 must be 16-bit signed integer (int16)")
        if pcm16.size == 0:
            # no frames -> tail window will be insufficient
            raise InsufficientSamplesError("insufficient samples for one 32 ms frame")
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE}, got {sample_rate}")

        norm = pcm16.astype(np.float32) / 32768.0
        n_frames_total = norm.shape[0] // FRAME_SAMPLES
        if n_frames_total <= 0:
            raise InsufficientSamplesError("insufficient samples for one 32 ms frame")

        window_frames = max(1, int((window_s * SAMPLE_RATE) // FRAME_SAMPLES))
        tail_frames = min(n_frames_total, window_frames)
        start_tail_idx = n_frames_total - tail_frames
        if tail_frames <= 0:
            # Defensive, though start_tail_idx logic above prevents this
            raise InsufficientSamplesError("insufficient samples for one 32 ms frame in tail window")

        keep_ctx_frames = max(1, int(round(keep_context_ms / FRAME_MS)))
        leading_low = 0
        found_voice = False
        max_p_tail = 0.0

        with torch.no_grad():
            for i in range(n_frames_total):
                frame = torch.from_numpy(norm[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]).to(self.device)
                p = float(self._model(frame, SAMPLE_RATE).item())

                # Leading silence measurement
                if not found_voice:
                    if p < threshold:
                        leading_low += 1
                    else:
                        found_voice = True

                # Tail window max
                if i >= start_tail_idx:
                    if p > max_p_tail:
                        max_p_tail = p

        if leading_low < min_consecutive:
            drop_frames = 0
        else:
            drop_frames = max(0, leading_low - keep_ctx_frames)
        return (drop_frames * FRAME_SAMPLES, max_p_tail)



__all__ = [
    "SileroVAD",
    "SAMPLE_RATE",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "InsufficientSamplesError",
]
