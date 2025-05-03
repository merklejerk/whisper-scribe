import numpy as np
from scipy.signal import resample_poly, butter, lfilter
from typing import Callable

def has_sound(
    waveform: np.ndarray,
    *,
    threshold=1e-4,
    sample_rate=16000,
    window_ms=50,
    stride_ms=10,
    total_sound_ms=250,
) -> bool:
    """ Checks if the waveform contains sound above a certain threshold. """
    window_size = int(sample_rate * window_ms / 1000) 
    stride = int(sample_rate * stride_ms / 1000)

    active_samples = 0

    for i in range(0, len(waveform) - window_size + 1, stride):
        window = waveform[i:i + window_size]
        energy = np.sqrt(np.mean(window**2))
        if energy > threshold:
            active_samples += window_size

    return active_samples >= int(total_sound_ms / 1000 * sample_rate)

def pcm16_to_norm_waveform(
        audio: bytes,
        source_sr: int | None = None,
        target_sr: int | None = None,
        pre_resample: Callable[[np.ndarray, int | None], np.ndarray] | None = None,
    ) -> np.ndarray:
    """Converts PCM int16 audio bytes to a normalized waveform."""
    samples = np.frombuffer(audio, dtype=np.int16).reshape(-1, 2)
    mono = np.mean(samples.astype(np.float32), axis=1)
    normed = (mono / 32768.0).clip(-1.0, 1.0).astype(np.float32)
    if pre_resample:
        normed = pre_resample(normed, source_sr)
    if source_sr and target_sr and source_sr != target_sr:
        return resample_poly(normed, target_sr, source_sr)
    return normed

def resample(
    signal: np.ndarray,
    source_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resamples the signal to the target sample rate."""
    if source_sr != target_sr:
        return resample_poly(signal, target_sr, source_sr)
    return signal.copy()

def bandpass_filter(
    signal: np.ndarray,
    sr: int,
    lowcut: float = 250.0,
    highcut: float = 3300.0,
    order: int = 5
) -> np.ndarray:
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal)

def pre_emphasis(signal: np.ndarray, coeff: float = 0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def rms_normalize(signal: np.ndarray, target_db: float = -20.0):
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return signal.copy()  # silent
    target_rms = 10 ** (target_db / 20.0)
    return signal * (target_rms / rms)

def get_duration(signal: np.ndarray, sr: int) -> float:
    return len(signal) / sr