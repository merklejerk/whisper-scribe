import numpy as np
from scipy.signal import resample_poly

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

def pcm16_to_norm_waveform(audio: bytes, source_sr: int, target_sr: int) -> np.ndarray:
    """Converts PCM int16 audio bytes to a normalized waveform."""
    samples = np.frombuffer(audio, dtype=np.int16).reshape(-1, 2)
    mono = np.mean(samples.astype(np.float32), axis=1)
    if source_sr != target_sr:
        resampled = resample_poly(mono, target_sr, source_sr)
    else:
        resampled = mono
    return np.clip(resampled, -32768, 32767).astype(np.float32) / 32768.0
