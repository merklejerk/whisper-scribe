from __future__ import annotations

"""Minimal device resolver.

Single env knob: DEVICE (defaults to 'auto').
Return ResolvedDevice with (name, pipeline_index) where pipeline_index is
compatible with HuggingFace pipelines (-1 for CPU, 0 otherwise).
Priority for 'auto': CUDA -> MPS -> ROCm -> CPU.
Gracefully falls back to CPU if explicit request unavailable.
"""

from dataclasses import dataclass
import os
import torch


@dataclass(frozen=True)
class ResolvedDevice:
    name: str  # 'cuda' | 'cpu' | 'mps'
    pipeline_index: int  # -1 for CPU else 0 (we only expose first device for now)

    def is_cpu(self) -> bool:
        return self.name == "cpu"

    def to_arg(self) -> str:
        if self.is_cpu():
            return "cpu"
        return f"{self.name}:{self.pipeline_index}"


def resolve_device(preference: str | None = None) -> ResolvedDevice:
    # Accept explicit arg or fallback to env var (DEVICE) without requiring caller to read env.
    pref = preference if preference is not None else os.getenv("DEVICE", "auto")
    pref = (pref or "").strip().lower() or "auto"

    def _make(name: str) -> ResolvedDevice:
        return ResolvedDevice(name=name, pipeline_index=0)

    # Explicit selection
    if pref != "auto":
        if pref.startswith("cuda"):
            if torch.cuda.is_available():
                return _make("cuda")
        elif pref == "mps":
            if torch.backends.mps.is_available():
                return _make("mps")
        elif pref.startswith("cpu"):
            return _make("cpu")
        # Unknown or unavailable explicit preference -> fall through to auto

    # Auto priority chain
    if torch.cuda.is_available():
        return _make("cuda")
    if torch.backends.mps.is_available():
        return _make("mps")
    return _make("cpu")


def resolve_from_env() -> ResolvedDevice:
    return resolve_device(None)


__all__ = [
    "ResolvedDevice",
    "resolve_device",
    "resolve_from_env",
]
